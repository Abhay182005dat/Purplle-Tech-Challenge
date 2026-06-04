"""
Event ingestion endpoint.
POST /events/ingest — validates, deduplicates, stores events and manages sessions.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import List

from sqlalchemy.orm import Session

from app.models import Event, IngestRequest, IngestResponse, VALID_EVENT_TYPES
from app.db import EventRow, SessionRow, PosTransactionRow, get_db_session

logger = logging.getLogger(__name__)


def ingest_events(request: IngestRequest, db: Session) -> IngestResponse:
    """
    Process a batch of events.
    - Validates each event
    - Deduplicates by event_id
    - Inserts valid events
    - Updates sessions incrementally
    - Returns partial success response (HTTP 200 even with some failures)
    """
    accepted = 0
    rejected = 0
    duplicate = 0
    errors: List[dict] = []

    for event in request.events:
        try:
            # Check duplicate by event_id
            existing = db.query(EventRow).filter_by(event_id=event.event_id).first()
            if existing:
                duplicate += 1
                continue

            # Set ingested_at to current UTC time
            event.ingested_at = datetime.utcnow()

            # Insert event into events table
            event_row = EventRow(
                event_id=event.event_id,
                store_id=event.store_id,
                camera_id=event.camera_id,
                visitor_id=event.visitor_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                ingested_at=event.ingested_at,
                zone_id=event.zone_id,
                dwell_ms=event.dwell_ms,
                is_staff=event.is_staff,
                confidence=event.confidence,
                is_face_hidden=event.is_face_hidden,
                group_id=event.group_id,
                group_size=event.group_size,
                metadata_json={
                    "queue_depth": event.metadata.queue_depth,
                    "queue_position_at_join": event.metadata.queue_position_at_join,
                    "wait_seconds": event.metadata.wait_seconds,
                    "zone_hotspot_x": event.metadata.zone_hotspot_x,
                    "zone_hotspot_y": event.metadata.zone_hotspot_y,
                    "sku_zone": event.metadata.sku_zone,
                    "session_seq": event.metadata.session_seq,
                },
            )
            db.add(event_row)

            # Update sessions incrementally
            _update_session(event, db)

            # Flush after each event so that subsequent events in the same batch
            # can find the session (e.g. ZONE_ENTER after ENTRY)
            db.flush()

            accepted += 1

        except Exception as e:
            rejected += 1
            errors.append({
                "event_id": getattr(event, "event_id", "unknown"),
                "reason": str(e),
            })
            logger.warning(f"Event rejected: {e}")

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database commit failed: {e}")
        raise

    return IngestResponse(
        accepted=accepted,
        rejected=rejected,
        duplicate=duplicate,
        errors=errors,
    )


def _update_session(event: Event, db: Session):
    """
    Incrementally update sessions table based on event type.

    SESSION MANAGEMENT LOGIC:
    - ENTRY: create new session (or new session with is_reentry=True if prior closed)
    - ZONE_ENTER: append zone_id to zones_visited
    - EXIT: close session, run POS correlation
    - REENTRY: create new session with is_reentry=True
    """
    if event.event_type == "ENTRY":
        # Check if visitor has an open session for this store
        open_session = (
            db.query(SessionRow)
            .filter_by(
                store_id=event.store_id,
                visitor_id=event.visitor_id,
            )
            .filter(SessionRow.exit_time.is_(None))
            .first()
        )

        if open_session is None:
            # Check if there's a prior closed session
            prior_session = (
                db.query(SessionRow)
                .filter_by(
                    store_id=event.store_id,
                    visitor_id=event.visitor_id,
                )
                .filter(SessionRow.exit_time.isnot(None))
                .first()
            )

            is_reentry = prior_session is not None

            new_session = SessionRow(
                session_id=str(uuid.uuid4()),
                store_id=event.store_id,
                visitor_id=event.visitor_id,
                entry_time=event.timestamp,
                is_reentry=is_reentry,
                zones_visited=[],
                converted=False,
                is_staff=event.is_staff,
            )
            db.add(new_session)

    elif event.event_type == "ZONE_ENTER":
        # Find open session for visitor
        open_session = (
            db.query(SessionRow)
            .filter_by(
                store_id=event.store_id,
                visitor_id=event.visitor_id,
            )
            .filter(SessionRow.exit_time.is_(None))
            .first()
        )

        if open_session and event.zone_id:
            zones = list(open_session.zones_visited or [])
            if event.zone_id not in zones:
                zones.append(event.zone_id)
                open_session.zones_visited = zones

    elif event.event_type == "EXIT":
        # Find open session for visitor
        open_session = (
            db.query(SessionRow)
            .filter_by(
                store_id=event.store_id,
                visitor_id=event.visitor_id,
            )
            .filter(SessionRow.exit_time.is_(None))
            .first()
        )

        if open_session:
            open_session.exit_time = event.timestamp
            # Run POS correlation
            _correlate_pos(open_session, event, db)

    elif event.event_type == "REENTRY":
        # Create new session with is_reentry=True
        new_session = SessionRow(
            session_id=str(uuid.uuid4()),
            store_id=event.store_id,
            visitor_id=event.visitor_id,
            entry_time=event.timestamp,
            is_reentry=True,
            zones_visited=[],
            converted=False,
            is_staff=event.is_staff,
        )
        db.add(new_session)

    elif event.event_type == "BILLING_QUEUE_JOIN":
        # Also update session zones_visited if zone_id present
        open_session = (
            db.query(SessionRow)
            .filter_by(
                store_id=event.store_id,
                visitor_id=event.visitor_id,
            )
            .filter(SessionRow.exit_time.is_(None))
            .first()
        )

        if open_session and event.zone_id:
            zones = list(open_session.zones_visited or [])
            if event.zone_id not in zones:
                zones.append(event.zone_id)
                open_session.zones_visited = zones


def _correlate_pos(session: SessionRow, exit_event: Event, db: Session):
    """
    POS CORRELATION (runs on EXIT):
    - Find most recent BILLING_QUEUE_JOIN or ZONE_ENTER to billing zone for this visitor
    - Look for POS transaction at same store where order_time is within 5 minutes after billing_time
    - If found: session.converted = True
    """
    # Find billing time — most recent BILLING_QUEUE_JOIN or billing zone ZONE_ENTER
    billing_event = (
        db.query(EventRow)
        .filter(
            EventRow.store_id == session.store_id,
            EventRow.visitor_id == session.visitor_id,
            EventRow.event_type.in_(["BILLING_QUEUE_JOIN", "ZONE_ENTER"]),
        )
        .order_by(EventRow.timestamp.desc())
        .first()
    )

    if billing_event is None:
        session.converted = False
        return

    billing_time = billing_event.timestamp

    # Look for POS transaction within 5-minute window after billing_time
    pos_match = (
        db.query(PosTransactionRow)
        .filter(
            PosTransactionRow.store_id == session.store_id,
            PosTransactionRow.order_time >= billing_time,
            PosTransactionRow.order_time <= billing_time + timedelta(minutes=5),
        )
        .first()
    )

    session.converted = pos_match is not None
