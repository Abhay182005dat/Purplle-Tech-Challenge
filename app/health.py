"""
Health endpoint.
GET /health — system health including database, stores, and camera feed status.
"""

import json
import logging
import os
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import func, distinct, text

from app.db import EventRow

logger = logging.getLogger(__name__)

# Cache store/camera definitions
_store_cameras_cache = {}


def _load_store_cameras():
    """Load store and camera definitions from store_layout.json."""
    global _store_cameras_cache
    if _store_cameras_cache:
        return _store_cameras_cache

    layout_path = os.environ.get("STORE_LAYOUT_PATH", "data/store_layout.json")
    try:
        with open(layout_path, "r") as f:
            data = json.load(f)
        for store in data.get("stores", []):
            _store_cameras_cache[store["store_id"]] = store.get("cameras", [])
    except Exception as e:
        logger.error(f"Error loading store cameras: {e}")

    return _store_cameras_cache


def get_health(db: Session) -> dict:
    """
    System health check.
    Works even if zero events have been ingested.
    Returns structured error on database failure — no stack traces.
    """
    now = datetime.utcnow()
    ten_minutes_ago = now - timedelta(minutes=10)

    try:
        # Test database connectivity with a simple query
        db.execute(text("SELECT 1"))
    except Exception:
        return {
            "status": "unhealthy",
            "checked_at": now.isoformat(),
            "database": "disconnected",
            "stores": {},
        }

    try:
        # Get all known stores and their cameras
        store_cameras = _load_store_cameras()

        # Also check stores that have events but aren't in layout
        event_stores = (
            db.query(distinct(EventRow.store_id))
            .all()
        )
        for (sid,) in event_stores:
            if sid not in store_cameras:
                # Get cameras from events
                cams = (
                    db.query(distinct(EventRow.camera_id))
                    .filter(EventRow.store_id == sid)
                    .all()
                )
                store_cameras[sid] = [c[0] for c in cams]

        stores_health = {}
        all_feeds_live = True
        any_feed_stale = False
        warnings = []

        for store_id, cameras in store_cameras.items():
            # Get last event for this store
            last_store_event = (
                db.query(EventRow)
                .filter(EventRow.store_id == store_id)
                .order_by(EventRow.timestamp.desc())
                .first()
            )

            if last_store_event is None:
                stores_health[store_id] = {
                    "last_event_timestamp": None,
                    "last_event_age_seconds": 0,
                    "feed_status": "NO_DATA",
                    "cameras": {
                        cam: {"last_event": None, "status": "NO_DATA"}
                        for cam in cameras
                    },
                }
                all_feeds_live = False
                continue

            last_ts = last_store_event.timestamp
            age_seconds = int((now - last_ts).total_seconds())
            store_feed_status = "LIVE" if last_ts >= ten_minutes_ago else "STALE"

            if store_feed_status == "STALE":
                any_feed_stale = True
                all_feeds_live = False
                warnings.append(f"STALE_FEED: {store_id} last event {age_seconds}s ago")
            elif store_feed_status != "LIVE":
                all_feeds_live = False

            # Camera-level status
            camera_health = {}
            for cam in cameras:
                last_cam_event = (
                    db.query(EventRow)
                    .filter(
                        EventRow.store_id == store_id,
                        EventRow.camera_id == cam,
                    )
                    .order_by(EventRow.timestamp.desc())
                    .first()
                )

                if last_cam_event is None:
                    camera_health[cam] = {
                        "last_event": None,
                        "status": "NO_DATA",
                    }
                    all_feeds_live = False
                else:
                    cam_ts = last_cam_event.timestamp
                    cam_status = "LIVE" if cam_ts >= ten_minutes_ago else "STALE"
                    if cam_status == "STALE":
                        any_feed_stale = True
                        all_feeds_live = False
                        warnings.append(f"STALE_FEED: {store_id}/{cam} last event {int((now - cam_ts).total_seconds())}s ago")
                    camera_health[cam] = {
                        "last_event": cam_ts.isoformat(),
                        "status": cam_status,
                    }

            stores_health[store_id] = {
                "last_event_timestamp": last_ts.isoformat(),
                "last_event_age_seconds": age_seconds,
                "feed_status": store_feed_status,
                "cameras": camera_health,
            }

        # Determine overall status
        if not store_cameras:
            status = "healthy"  # No stores configured = healthy (empty state)
        elif all_feeds_live:
            status = "healthy"
        elif any_feed_stale:
            status = "degraded"
        else:
            status = "healthy"

        result = {
            "status": status,
            "checked_at": now.isoformat(),
            "database": "connected",
            "stores": stores_health,
        }

        if warnings:
            result["warnings"] = warnings

        return result

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "checked_at": now.isoformat(),
            "database": "disconnected",
            "stores": {},
        }
