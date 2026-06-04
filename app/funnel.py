"""
Funnel endpoint.
GET /stores/{store_id}/funnel — session-based conversion funnel.
Re-entries must NOT double-count a visitor.
"""

import logging
from datetime import datetime, date

from sqlalchemy.orm import Session
from sqlalchemy import func, distinct

from app.db import EventRow, SessionRow

logger = logging.getLogger(__name__)


def get_funnel(store_id: str, db: Session) -> dict:
    """
    Session-based funnel: ENTRY → ZONE_VISIT → BILLING_QUEUE → PURCHASE.
    Each stage is a SUBSET of the previous stage.
    Re-entries count the visitor ONCE, not once per visit.
    """
    today = date.today()
    today_start = datetime.combine(today, datetime.min.time())
    today_end = datetime.combine(today, datetime.max.time())

    try:
        # ENTRY count: distinct visitor_ids with ENTRY event today, is_staff=False
        entry_visitors = set()
        entry_results = (
            db.query(distinct(EventRow.visitor_id))
            .filter(
                EventRow.store_id == store_id,
                EventRow.event_type == "ENTRY",
                EventRow.is_staff == False,
                EventRow.timestamp >= today_start,
                EventRow.timestamp <= today_end,
            )
            .all()
        )
        entry_visitors = {r[0] for r in entry_results}
        entry_count = len(entry_visitors)

        # ZONE_VISIT count: of those, who visited at least one non-billing zone
        zone_visit_visitors = set()
        if entry_visitors:
            # Get sessions for these visitors that have zones_visited
            sessions = (
                db.query(SessionRow)
                .filter(
                    SessionRow.store_id == store_id,
                    SessionRow.is_staff == False,
                    SessionRow.visitor_id.in_(entry_visitors),
                    SessionRow.entry_time >= today_start,
                    SessionRow.entry_time <= today_end,
                )
                .all()
            )

            for session in sessions:
                zones = session.zones_visited or []
                # Check for non-billing zones
                non_billing_zones = [z for z in zones if z and z.upper() != "BILLING"]
                if non_billing_zones:
                    zone_visit_visitors.add(session.visitor_id)

        zone_visit_count = len(zone_visit_visitors)

        # BILLING_QUEUE count: of zone visitors, who have BILLING_QUEUE_JOIN event
        billing_visitors = set()
        if zone_visit_visitors:
            billing_results = (
                db.query(distinct(EventRow.visitor_id))
                .filter(
                    EventRow.store_id == store_id,
                    EventRow.event_type == "BILLING_QUEUE_JOIN",
                    EventRow.is_staff == False,
                    EventRow.visitor_id.in_(zone_visit_visitors),
                    EventRow.timestamp >= today_start,
                    EventRow.timestamp <= today_end,
                )
                .all()
            )
            billing_visitors = {r[0] for r in billing_results}

        billing_count = len(billing_visitors)

        # PURCHASE count: of billing visitors, who have session.converted = True
        purchase_visitors = set()
        if billing_visitors:
            converted_sessions = (
                db.query(distinct(SessionRow.visitor_id))
                .filter(
                    SessionRow.store_id == store_id,
                    SessionRow.is_staff == False,
                    SessionRow.converted == True,
                    SessionRow.visitor_id.in_(billing_visitors),
                    SessionRow.entry_time >= today_start,
                    SessionRow.entry_time <= today_end,
                )
                .all()
            )
            purchase_visitors = {r[0] for r in converted_sessions}

        purchase_count = len(purchase_visitors)

        # Calculate drop-off percentages
        def drop_off(current, previous):
            if previous == 0:
                return 0.0
            return round((previous - current) / previous * 100, 1)

        # Overall conversion rate
        conversion_rate = 0.0
        if entry_count > 0:
            conversion_rate = round(purchase_count / entry_count * 100, 1)

        return {
            "store_id": store_id,
            "window": "today",
            "stages": [
                {
                    "stage": "ENTRY",
                    "count": entry_count,
                    "drop_off_pct": 0.0,
                },
                {
                    "stage": "ZONE_VISIT",
                    "count": zone_visit_count,
                    "drop_off_pct": drop_off(zone_visit_count, entry_count),
                },
                {
                    "stage": "BILLING_QUEUE",
                    "count": billing_count,
                    "drop_off_pct": drop_off(billing_count, zone_visit_count),
                },
                {
                    "stage": "PURCHASE",
                    "count": purchase_count,
                    "drop_off_pct": drop_off(purchase_count, billing_count),
                },
            ],
            "conversion_rate": conversion_rate,
        }

    except Exception as e:
        logger.error(f"Error computing funnel for {store_id}: {e}")
        return {
            "store_id": store_id,
            "window": "today",
            "stages": [
                {"stage": "ENTRY", "count": 0, "drop_off_pct": 0.0},
                {"stage": "ZONE_VISIT", "count": 0, "drop_off_pct": 0.0},
                {"stage": "BILLING_QUEUE", "count": 0, "drop_off_pct": 0.0},
                {"stage": "PURCHASE", "count": 0, "drop_off_pct": 0.0},
            ],
            "conversion_rate": 0.0,
        }
