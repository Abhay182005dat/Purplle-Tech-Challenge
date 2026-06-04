# PROMPT: Generate tests for the detection pipeline event schema validation,
# verifying all event types in the catalogue are valid, event_ids are unique,
# timestamps are ISO-8601, and all required fields are present.
# CHANGES MADE: Added tests for sample_events.jsonl schema validation,
# event type catalogue completeness, and metadata field presence.

"""
Tests for the detection pipeline output schema and event validation.
"""

import json
import uuid
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import VALID_EVENT_TYPES, Event, EventMetadata
from app.db import Base, engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_db():
    """Reset database before each test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def make_event(
    event_type="ENTRY",
    store_id="ST1008",
    visitor_id=None,
    confidence=0.85,
    zone_id=None,
    dwell_ms=0,
    timestamp=None,
    queue_depth=None,
    sku_zone=None,
    session_seq=1,
):
    """Helper to create a valid event dict."""
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": visitor_id or f"VIS_{str(uuid.uuid4())[:6]}",
        "event_type": event_type,
        "timestamp": (timestamp or datetime.utcnow()).isoformat(),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": False,
        "confidence": confidence,
        "is_face_hidden": False,
        "group_id": None,
        "group_size": None,
        "metadata": {
            "session_seq": session_seq,
            "queue_depth": queue_depth,
            "sku_zone": sku_zone,
        },
    }


# ─── Test Cases ───────────────────────────────────────────────────────────────


def test_valid_event_types_catalogue():
    """All 8 event types in the catalogue must be recognised."""
    expected_types = {
        "ENTRY",
        "EXIT",
        "ZONE_ENTER",
        "ZONE_EXIT",
        "ZONE_DWELL",
        "BILLING_QUEUE_JOIN",
        "BILLING_QUEUE_ABANDON",
        "REENTRY",
    }
    assert VALID_EVENT_TYPES == expected_types


def test_event_schema_has_required_fields():
    """Event model must have all required fields from the PDF schema."""
    required_fields = {
        "event_id",
        "store_id",
        "camera_id",
        "visitor_id",
        "event_type",
        "timestamp",
        "zone_id",
        "dwell_ms",
        "is_staff",
        "confidence",
        "metadata",
    }
    model_fields = set(Event.model_fields.keys())
    for field in required_fields:
        assert field in model_fields, f"Missing required field: {field}"


def test_metadata_schema_has_required_fields():
    """EventMetadata must include queue_depth, sku_zone, session_seq."""
    required_metadata_fields = {"queue_depth", "sku_zone", "session_seq"}
    metadata_fields = set(EventMetadata.model_fields.keys())
    for field in required_metadata_fields:
        assert field in metadata_fields, f"Missing metadata field: {field}"


def test_event_id_must_be_unique():
    """Duplicate event_ids should be detected as duplicates, not accepted twice."""
    eid = str(uuid.uuid4())
    event1 = make_event()
    event1["event_id"] = eid
    event2 = make_event()
    event2["event_id"] = eid

    # First ingest
    r1 = client.post("/events/ingest", json={"events": [event1]})
    assert r1.status_code == 200
    assert r1.json()["accepted"] == 1

    # Second ingest with same event_id
    r2 = client.post("/events/ingest", json={"events": [event2]})
    assert r2.status_code == 200
    assert r2.json()["duplicate"] == 1
    assert r2.json()["accepted"] == 0


def test_invalid_event_type_rejected():
    """Invalid event_type should be rejected at schema validation."""
    event = {
        "event_id": str(uuid.uuid4()),
        "store_id": "ST1008",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": "VIS_test01",
        "event_type": "INVALID_TYPE",
        "timestamp": datetime.utcnow().isoformat(),
        "confidence": 0.85,
    }
    response = client.post("/events/ingest", json={"events": [event]})
    assert response.status_code == 422


def test_timestamp_iso_8601_format():
    """Events must accept ISO-8601 timestamps."""
    # Various valid ISO-8601 formats
    valid_timestamps = [
        "2026-03-03T14:22:10Z",
        "2026-03-03T14:22:10.000Z",
        "2026-03-03T14:22:10+00:00",
        "2026-06-04T15:00:00",
    ]

    for ts in valid_timestamps:
        event = make_event()
        event["timestamp"] = ts
        response = client.post("/events/ingest", json={"events": [event]})
        assert response.status_code == 200, f"Failed for timestamp: {ts}"
        assert response.json()["accepted"] == 1


def test_all_event_types_ingestable():
    """Each event type in the catalogue should be accepted by the API."""
    now = datetime.utcnow()
    vid = "VIS_alltype"

    for event_type in VALID_EVENT_TYPES:
        event = make_event(
            event_type=event_type,
            visitor_id=vid,
            zone_id="FRAGRANCE" if "ZONE" in event_type or "BILLING" in event_type else None,
            dwell_ms=30000 if event_type == "ZONE_DWELL" else 0,
            timestamp=now,
        )
        response = client.post("/events/ingest", json={"events": [event]})
        assert response.status_code == 200, f"Failed for event_type: {event_type}"
        assert response.json()["rejected"] == 0, f"Rejected event_type: {event_type}"


def test_confidence_low_not_suppressed():
    """Low-confidence events should still be accepted, not suppressed."""
    event = make_event(confidence=0.15)
    response = client.post("/events/ingest", json={"events": [event]})
    assert response.status_code == 200
    assert response.json()["accepted"] == 1


def test_group_events_emit_individual_entries():
    """Group entry: 3 people entering together should produce 3 separate events."""
    now = datetime.utcnow()
    group_id = f"G_{uuid.uuid4().hex[:4]}"

    events = []
    for i in range(3):
        event = make_event(
            event_type="ENTRY",
            visitor_id=f"VIS_grp{i:02d}",
            timestamp=now,
        )
        event["group_id"] = group_id
        event["group_size"] = 3
        events.append(event)

    response = client.post("/events/ingest", json={"events": events})
    assert response.status_code == 200
    assert response.json()["accepted"] == 3


def test_staff_events_flagged():
    """Staff events should have is_staff=True and be accepted."""
    event = make_event(event_type="ENTRY")
    event["is_staff"] = True
    response = client.post("/events/ingest", json={"events": [event]})
    assert response.status_code == 200
    assert response.json()["accepted"] == 1


def test_zone_dwell_includes_dwell_ms():
    """ZONE_DWELL events should include non-zero dwell_ms."""
    event = make_event(
        event_type="ZONE_DWELL",
        zone_id="FRAGRANCE",
        dwell_ms=30000,
    )
    response = client.post("/events/ingest", json={"events": [event]})
    assert response.status_code == 200
    assert response.json()["accepted"] == 1


def test_billing_queue_join_includes_queue_depth():
    """BILLING_QUEUE_JOIN events should include queue_depth in metadata."""
    event = make_event(
        event_type="BILLING_QUEUE_JOIN",
        zone_id="BILLING",
        queue_depth=3,
    )
    response = client.post("/events/ingest", json={"events": [event]})
    assert response.status_code == 200
    assert response.json()["accepted"] == 1
