"""
Metrics endpoint.
GET /stores/{store_id}/metrics — real-time store metrics, never cached.
"""

import logging
from datetime import datetime, date, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import func, distinct

from app.db import EventRow, SessionRow

logger = logging.getLogger(__name__)


def get_metrics(store_id: str, db: Session) -> dict:
    """
    Query metrics fresh on each request.
    ALWAYS exclude is_staff=True from all counts.
    NEVER return null for numeric fields — zero traffic = zeros.
    """
    today = date.today()
    today_start = datetime.combine(today, datetime.min.time())
    today_end = datetime.combine(today, datetime.max.time())

    try:
        # Unique visitors: distinct visitor_id from sessions where is_staff=False and entry today
        unique_visitors = (
            db.query(func.count(distinct(SessionRow.visitor_id)))
            .filter(
                SessionRow.store_id == store_id,
                SessionRow.is_staff == False,
                SessionRow.entry_time >= today_start,
                SessionRow.entry_time <= today_end,
            )
            .scalar()
        ) or 0

        # Total sessions today (non-staff)
        total_sessions = (
            db.query(func.count(SessionRow.session_id))
            .filter(
                SessionRow.store_id == store_id,
                SessionRow.is_staff == False,
                SessionRow.entry_time >= today_start,
                SessionRow.entry_time <= today_end,
            )
            .scalar()
        ) or 0

        # Converted sessions today (non-staff)
        converted_sessions = (
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

        # Conversion rate
        conversion_rate = 0.0
        if total_sessions > 0:
            conversion_rate = round(converted_sessions / total_sessions * 100, 1)

        # Average dwell per zone from ZONE_DWELL events
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

        avg_dwell_per_zone = {}
        for zone_id, avg_dwell in zone_dwells:
            if zone_id:
                avg_dwell_per_zone[zone_id] = round((avg_dwell or 0) / 1000, 1)

        # Current queue depth: most recent queue_depth from BILLING_QUEUE_JOIN in last 10 minutes
        ten_minutes_ago = datetime.utcnow() - timedelta(minutes=10)
        recent_queue_event = (
            db.query(EventRow)
            .filter(
                EventRow.store_id == store_id,
                EventRow.event_type == "BILLING_QUEUE_JOIN",
                EventRow.timestamp >= ten_minutes_ago,
            )
            .order_by(EventRow.timestamp.desc())
            .first()
        )

        current_queue_depth = 0
        if recent_queue_event and recent_queue_event.metadata_json:
            current_queue_depth = recent_queue_event.metadata_json.get("queue_depth", 0) or 0

        # Abandonment rate: BILLING_QUEUE_ABANDON / BILLING_QUEUE_JOIN
        queue_joins = (
            db.query(func.count(EventRow.event_id))
            .filter(
                EventRow.store_id == store_id,
                EventRow.event_type == "BILLING_QUEUE_JOIN",
                EventRow.is_staff == False,
                EventRow.timestamp >= today_start,
                EventRow.timestamp <= today_end,
            )
            .scalar()
        ) or 0

        queue_abandons = (
            db.query(func.count(EventRow.event_id))
            .filter(
                EventRow.store_id == store_id,
                EventRow.event_type == "BILLING_QUEUE_ABANDON",
                EventRow.is_staff == False,
                EventRow.timestamp >= today_start,
                EventRow.timestamp <= today_end,
            )
            .scalar()
        ) or 0

        abandonment_rate = 0.0
        if queue_joins > 0:
            abandonment_rate = round(queue_abandons / queue_joins * 100, 1)

        return {
            "store_id": store_id,
            "date": str(today),
            "unique_visitors": unique_visitors,
            "conversion_rate": conversion_rate,
            "avg_dwell_per_zone": avg_dwell_per_zone,
            "current_queue_depth": current_queue_depth,
            "abandonment_rate": abandonment_rate,
        }

    except Exception as e:
        logger.error(f"Error computing metrics for {store_id}: {e}")
        # Return zeros on error — never null
        return {
            "store_id": store_id,
            "date": str(today),
            "unique_visitors": 0,
            "conversion_rate": 0.0,
            "avg_dwell_per_zone": {},
            "current_queue_depth": 0,
            "abandonment_rate": 0.0,
        }
