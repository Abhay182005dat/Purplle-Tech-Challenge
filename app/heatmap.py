"""
Heatmap endpoint.
GET /stores/{store_id}/heatmap — zone activity with normalised visit frequency and dwell.
"""

import json
import logging
import os
from datetime import datetime, date

from sqlalchemy.orm import Session
from sqlalchemy import func, distinct

from app.db import EventRow, SessionRow

logger = logging.getLogger(__name__)

# Cache store layout zones
_store_zones_cache = {}


def _load_store_zones():
    """Load zone definitions from store_layout.json."""
    global _store_zones_cache
    if _store_zones_cache:
        return _store_zones_cache

    layout_path = os.environ.get("STORE_LAYOUT_PATH", "data/store_layout.json")
    try:
        with open(layout_path, "r") as f:
            data = json.load(f)

        for store in data.get("stores", []):
            store_id = store["store_id"]
            _store_zones_cache[store_id] = {
                zone["zone_id"]: zone["zone_name"]
                for zone in store.get("zones", [])
            }
    except Exception as e:
        logger.error(f"Error loading store layout: {e}")

    return _store_zones_cache


def get_heatmap(store_id: str, db: Session) -> dict:
    """
    Zone activity heatmap with normalised metrics.
    - visit_frequency: normalised 0-100 (raw count / max count * 100)
    - avg_dwell_seconds: average dwell time in seconds
    - normalised_dwell: normalised 0-100 across zones
    - data_confidence: True if session_count >= 20
    """
    today = date.today()
    today_start = datetime.combine(today, datetime.min.time())
    today_end = datetime.combine(today, datetime.max.time())

    try:
        # Get session count for data confidence
        session_count = (
            db.query(func.count(SessionRow.session_id))
            .filter(
                SessionRow.store_id == store_id,
                SessionRow.is_staff == False,
                SessionRow.entry_time >= today_start,
                SessionRow.entry_time <= today_end,
            )
            .scalar()
        ) or 0

        # Get zone visit counts from ZONE_ENTER events
        zone_visits = (
            db.query(
                EventRow.zone_id,
                func.count(EventRow.event_id),
            )
            .filter(
                EventRow.store_id == store_id,
                EventRow.event_type == "ZONE_ENTER",
                EventRow.is_staff == False,
                EventRow.timestamp >= today_start,
                EventRow.timestamp <= today_end,
            )
            .group_by(EventRow.zone_id)
            .all()
        )

        # Get average dwell per zone from ZONE_DWELL events
        zone_dwells = (
            db.query(
                EventRow.zone_id,
                func.avg(EventRow.dwell_ms),
            )
            .filter(
                EventRow.store_id == store_id,
                EventRow.event_type == "ZONE_DWELL",
                EventRow.is_staff == False,
                EventRow.timestamp >= today_start,
                EventRow.timestamp <= today_end,
            )
            .group_by(EventRow.zone_id)
            .all()
        )

        # Build raw data maps
        raw_counts = {zone_id: count for zone_id, count in zone_visits if zone_id}
        raw_dwells = {
            zone_id: (avg_ms or 0) / 1000
            for zone_id, avg_ms in zone_dwells
            if zone_id
        }

        # Normalise visit_frequency
        max_count = max(raw_counts.values()) if raw_counts else 1
        # Normalise dwell
        max_dwell = max(raw_dwells.values()) if raw_dwells else 1

        # Load zone names from layout
        store_zones = _load_store_zones()
        zone_names = store_zones.get(store_id, {})

        # Build zone list — include all zones with visits
        all_zone_ids = set(raw_counts.keys()) | set(raw_dwells.keys()) | set(zone_names.keys())

        zones = []
        for zone_id in sorted(all_zone_ids):
            visit_count = raw_counts.get(zone_id, 0)
            avg_dwell = raw_dwells.get(zone_id, 0.0)

            zones.append({
                "zone_id": zone_id,
                "zone_name": zone_names.get(zone_id, zone_id),
                "visit_frequency": round(visit_count / max_count * 100) if max_count > 0 else 0,
                "avg_dwell_seconds": round(avg_dwell, 1),
                "normalised_dwell": round(avg_dwell / max_dwell * 100) if max_dwell > 0 else 0,
            })

        return {
            "store_id": store_id,
            "session_count": session_count,
            "data_confidence": session_count >= 20,
            "zones": zones,
        }

    except Exception as e:
        logger.error(f"Error computing heatmap for {store_id}: {e}")
        return {
            "store_id": store_id,
            "session_count": 0,
            "data_confidence": False,
            "zones": [],
        }
