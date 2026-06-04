"""
Anomalies endpoint.
GET /stores/{store_id}/anomalies — 4 anomaly detectors with severity and suggested actions.
"""

import json
import logging
import os
import uuid
from datetime import datetime, date, timedelta, time

from sqlalchemy.orm import Session
from sqlalchemy import func, distinct

from app.db import EventRow, SessionRow

logger = logging.getLogger(__name__)

# Cache store open/close times
_store_hours_cache = {}


def _load_store_hours():
    """Load store open/close times from store_layout.json."""
    global _store_hours_cache
    if _store_hours_cache:
        return _store_hours_cache

    layout_path = os.environ.get("STORE_LAYOUT_PATH", "data/store_layout.json")
    try:
        with open(layout_path, "r") as f:
            data = json.load(f)
        for store in data.get("stores", []):
            sid = store["store_id"]
            open_h, open_m = map(int, store["open_time"].split(":"))
            close_h, close_m = map(int, store["close_time"].split(":"))
            _store_hours_cache[sid] = {
                "open_time": time(open_h, open_m),
                "close_time": time(close_h, close_m),
                "cameras": store.get("cameras", []),
            }
    except Exception as e:
        logger.error(f"Error loading store hours: {e}")

    return _store_hours_cache


def get_anomalies(store_id: str, db: Session) -> dict:
    """
    Run all 4 anomaly detectors and return active anomalies.
    Empty anomalies list = [] not null.
    """
    now = datetime.utcnow()
    anomalies = []

    try:
        # ANOMALY 1: BILLING_QUEUE_SPIKE
        spike = _check_billing_queue_spike(store_id, db, now)
        if spike:
            anomalies.append(spike)

        # ANOMALY 2: CONVERSION_DROP
        conv_drop = _check_conversion_drop(store_id, db, now)
        if conv_drop:
            anomalies.append(conv_drop)

        # ANOMALY 3: DEAD_ZONE
        dead_zone = _check_dead_zone(store_id, db, now)
        if dead_zone:
            anomalies.append(dead_zone)

        # ANOMALY 4: STALE_FEED
        stale_feeds = _check_stale_feed(store_id, db, now)
        anomalies.extend(stale_feeds)

    except Exception as e:
        logger.error(f"Error checking anomalies for {store_id}: {e}")

    return {
        "store_id": store_id,
        "checked_at": now.isoformat(),
        "anomalies": anomalies,
    }


def _check_billing_queue_spike(store_id: str, db: Session, now: datetime) -> dict | None:
    """
    ANOMALY 1: BILLING_QUEUE_SPIKE
    Condition: queue_depth > 5 for 3+ consecutive minutes
    Check last 10 BILLING_QUEUE_JOIN events.
    """
    try:
        recent_queue_events = (
            db.query(EventRow)
            .filter(
                EventRow.store_id == store_id,
                EventRow.event_type == "BILLING_QUEUE_JOIN",
            )
            .order_by(EventRow.timestamp.desc())
            .limit(10)
            .all()
        )

        if len(recent_queue_events) < 3:
            return None

        # Check if queue_depth > 5 for 3+ consecutive minutes
        high_depth_events = []
        for evt in recent_queue_events:
            metadata = evt.metadata_json or {}
            depth = metadata.get("queue_depth", 0) or 0
            if depth > 5:
                high_depth_events.append(evt)

        if len(high_depth_events) < 3:
            return None

        # Check if they span at least 3 minutes
        timestamps = sorted([e.timestamp for e in high_depth_events])
        if len(timestamps) >= 2:
            time_span = (timestamps[-1] - timestamps[0]).total_seconds()
            if time_span >= 180:  # 3 minutes
                return {
                    "anomaly_id": str(uuid.uuid4()),
                    "type": "BILLING_QUEUE_SPIKE",
                    "severity": "WARN",
                    "detected_at": now.isoformat(),
                    "description": f"Billing queue depth exceeded 5 for {int(time_span // 60)} consecutive minutes",
                    "suggested_action": "Deploy additional billing staff immediately",
                    "metadata": {
                        "events_checked": len(recent_queue_events),
                        "high_depth_count": len(high_depth_events),
                        "time_span_seconds": int(time_span),
                    },
                }

    except Exception as e:
        logger.error(f"Error checking billing queue spike: {e}")

    return None


def _check_conversion_drop(store_id: str, db: Session, now: datetime) -> dict | None:
    """
    ANOMALY 2: CONVERSION_DROP
    Condition: today's conversion_rate < 7-day average * 0.7
    Skip if fewer than 3 days of history.
    """
    try:
        today = date.today()
        today_start = datetime.combine(today, datetime.min.time())
        today_end = datetime.combine(today, datetime.max.time())

        # Today's conversion rate
        total_today = (
            db.query(func.count(SessionRow.session_id))
            .filter(
                SessionRow.store_id == store_id,
                SessionRow.is_staff == False,
                SessionRow.entry_time >= today_start,
                SessionRow.entry_time <= today_end,
            )
            .scalar()
        ) or 0

        converted_today = (
            db.query(func.count(SessionRow.session_id))
            .filter(
                SessionRow.store_id == store_id,
                SessionRow.is_staff == False,
                SessionRow.converted == True,
                SessionRow.entry_time >= today_start,
                SessionRow.entry_time <= today_end,
            )
            .scalar()
        ) or 0

        if total_today == 0:
            return None

        today_rate = converted_today / total_today

        # 7-day average (excluding today)
        seven_days_ago = datetime.combine(today - timedelta(days=7), datetime.min.time())
        yesterday_end = datetime.combine(today - timedelta(days=1), datetime.max.time())

        # Check we have at least 3 days of history
        distinct_days = (
            db.query(func.count(distinct(func.date(SessionRow.entry_time))))
            .filter(
                SessionRow.store_id == store_id,
                SessionRow.is_staff == False,
                SessionRow.entry_time >= seven_days_ago,
                SessionRow.entry_time <= yesterday_end,
            )
            .scalar()
        ) or 0

        if distinct_days < 3:
            return None

        total_7d = (
            db.query(func.count(SessionRow.session_id))
            .filter(
                SessionRow.store_id == store_id,
                SessionRow.is_staff == False,
                SessionRow.entry_time >= seven_days_ago,
                SessionRow.entry_time <= yesterday_end,
            )
            .scalar()
        ) or 0

        converted_7d = (
            db.query(func.count(SessionRow.session_id))
            .filter(
                SessionRow.store_id == store_id,
                SessionRow.is_staff == False,
                SessionRow.converted == True,
                SessionRow.entry_time >= seven_days_ago,
                SessionRow.entry_time <= yesterday_end,
            )
            .scalar()
        ) or 0

        if total_7d == 0:
            return None

        avg_7d_rate = converted_7d / total_7d

        # Determine severity
        severity = None
        if today_rate < avg_7d_rate * 0.5:
            severity = "CRITICAL"
        elif today_rate < avg_7d_rate * 0.7:
            severity = "WARN"

        if severity:
            return {
                "anomaly_id": str(uuid.uuid4()),
                "type": "CONVERSION_DROP",
                "severity": severity,
                "detected_at": now.isoformat(),
                "description": (
                    f"Today's conversion rate ({today_rate:.1%}) is significantly below "
                    f"7-day average ({avg_7d_rate:.1%})"
                ),
                "suggested_action": "Review floor staff engagement and promotional displays",
                "metadata": {
                    "today_rate": round(today_rate * 100, 1),
                    "seven_day_avg_rate": round(avg_7d_rate * 100, 1),
                    "days_of_history": distinct_days,
                },
            }

    except Exception as e:
        logger.error(f"Error checking conversion drop: {e}")

    return None


def _check_dead_zone(store_id: str, db: Session, now: datetime) -> dict | None:
    """
    ANOMALY 3: DEAD_ZONE
    Condition: no ZONE_ENTER event for any zone in last 30 minutes during store open hours.
    Only trigger during open hours.
    """
    try:
        store_hours = _load_store_hours()
        store_info = store_hours.get(store_id)
        if not store_info:
            return None

        current_time = now.time()
        if current_time < store_info["open_time"] or current_time > store_info["close_time"]:
            return None  # Not during open hours

        thirty_min_ago = now - timedelta(minutes=30)
        recent_zone_enters = (
            db.query(func.count(EventRow.event_id))
            .filter(
                EventRow.store_id == store_id,
                EventRow.event_type == "ZONE_ENTER",
                EventRow.timestamp >= thirty_min_ago,
            )
            .scalar()
        ) or 0

        if recent_zone_enters == 0:
            return {
                "anomaly_id": str(uuid.uuid4()),
                "type": "DEAD_ZONE",
                "severity": "INFO",
                "detected_at": now.isoformat(),
                "description": "No zone activity detected in the last 30 minutes during store hours",
                "suggested_action": "Check camera feed and zone visibility",
                "metadata": {
                    "last_check_window_minutes": 30,
                    "store_open_time": str(store_info["open_time"]),
                    "store_close_time": str(store_info["close_time"]),
                },
            }

    except Exception as e:
        logger.error(f"Error checking dead zone: {e}")

    return None


def _check_stale_feed(store_id: str, db: Session, now: datetime) -> list:
    """
    ANOMALY 4: STALE_FEED
    Condition: no events from a specific camera_id in last 10 minutes.
    Returns a list (one anomaly per stale camera).
    """
    anomalies = []
    try:
        store_hours = _load_store_hours()
        store_info = store_hours.get(store_id)
        cameras = store_info["cameras"] if store_info else []

        if not cameras:
            # Fallback: get distinct cameras from events
            camera_results = (
                db.query(distinct(EventRow.camera_id))
                .filter(EventRow.store_id == store_id)
                .all()
            )
            cameras = [r[0] for r in camera_results]

        ten_min_ago = now - timedelta(minutes=10)

        for camera_id in cameras:
            recent_events = (
                db.query(func.count(EventRow.event_id))
                .filter(
                    EventRow.store_id == store_id,
                    EventRow.camera_id == camera_id,
                    EventRow.timestamp >= ten_min_ago,
                )
                .scalar()
            ) or 0

            if recent_events == 0:
                # Check if we've ever received events from this camera
                any_events = (
                    db.query(func.count(EventRow.event_id))
                    .filter(
                        EventRow.store_id == store_id,
                        EventRow.camera_id == camera_id,
                    )
                    .scalar()
                ) or 0

                if any_events > 0:
                    # Stale — had events before but none recently
                    anomalies.append({
                        "anomaly_id": str(uuid.uuid4()),
                        "type": "STALE_FEED",
                        "severity": "WARN",
                        "detected_at": now.isoformat(),
                        "description": f"No events received from camera {camera_id} in the last 10 minutes",
                        "suggested_action": "Check camera connection and pipeline health",
                        "metadata": {
                            "camera_id": camera_id,
                            "window_minutes": 10,
                        },
                    })

    except Exception as e:
        logger.error(f"Error checking stale feeds: {e}")

    return anomalies
