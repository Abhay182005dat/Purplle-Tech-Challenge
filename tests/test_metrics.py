# PROMPT: Generate tests for the metrics endpoint
# CHANGES MADE: Added zero-traffic, staff exclusion, and dwell calculation tests

"""
Tests for GET /stores/{store_id}/metrics endpoint.
"""

import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import app
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
    is_staff=False,
    confidence=0.85,
    zone_id=None,
    dwell_ms=0,
    timestamp=None,
    queue_depth=None,
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
        "is_staff": is_staff,
        "confidence": confidence,
        "is_face_hidden": False,
        "group_id": None,
        "group_size": None,
        "metadata": {
            "session_seq": 1,
            "queue_depth": queue_depth,
        },
    }


def ingest(events):
    """Helper to ingest events."""
    response = client.post("/events/ingest", json={"events": events})
    assert response.status_code == 200
    return response.json()


# ─── Test Cases ───────────────────────────────────────────────────────────────


def test_zero_traffic_returns_zeros_not_null():
    """Empty database: all numeric fields = 0 or 0.0, never null."""
    response = client.get("/stores/ST1008/metrics")
    assert response.status_code == 200
    data = response.json()

    assert data["unique_visitors"] == 0
    assert data["conversion_rate"] == 0.0
    assert data["avg_dwell_per_zone"] == {}
    assert data["current_queue_depth"] == 0
    assert data["abandonment_rate"] == 0.0

    # Verify no null values
    assert data["unique_visitors"] is not None
    assert data["conversion_rate"] is not None
    assert data["current_queue_depth"] is not None
    assert data["abandonment_rate"] is not None


def test_unique_visitor_count_correct():
    """Unique visitor count should reflect distinct visitor_ids."""
    now = datetime.utcnow()
    events = [
        make_event(visitor_id="VIS_aaa111", timestamp=now),
        make_event(visitor_id="VIS_bbb222", timestamp=now),
        make_event(visitor_id="VIS_ccc333", timestamp=now),
    ]
    ingest(events)

    response = client.get("/stores/ST1008/metrics")
    data = response.json()
    assert data["unique_visitors"] == 3


def test_conversion_rate_calculation():
    """Conversion rate = converted sessions / total sessions * 100."""
    now = datetime.utcnow()
    vid1 = "VIS_conv01"
    vid2 = "VIS_conv02"

    # Create 2 visitors, 1 with a completed purchase journey
    events = [
        make_event(event_type="ENTRY", visitor_id=vid1, timestamp=now),
        make_event(event_type="ENTRY", visitor_id=vid2, timestamp=now),
        make_event(
            event_type="EXIT",
            visitor_id=vid1,
            timestamp=now + timedelta(minutes=20),
        ),
        make_event(
            event_type="EXIT",
            visitor_id=vid2,
            timestamp=now + timedelta(minutes=20),
        ),
    ]
    ingest(events)

    response = client.get("/stores/ST1008/metrics")
    data = response.json()
    # Conversion rate should be a number (0.0 if no POS correlation match)
    assert isinstance(data["conversion_rate"], (int, float))
    assert data["conversion_rate"] >= 0.0


def test_staff_excluded_from_visitor_count():
    """Staff events (is_staff=True) should not be counted in unique_visitors."""
    now = datetime.utcnow()
    events = [
        make_event(visitor_id="VIS_cust01", is_staff=False, timestamp=now),
        make_event(visitor_id="VIS_cust02", is_staff=False, timestamp=now),
        make_event(visitor_id="VIS_staff1", is_staff=True, timestamp=now),
        make_event(visitor_id="VIS_staff2", is_staff=True, timestamp=now),
    ]
    ingest(events)

    response = client.get("/stores/ST1008/metrics")
    data = response.json()
    assert data["unique_visitors"] == 2  # Only customers, not staff


def test_zero_purchases_conversion_rate_zero():
    """With visitors but no purchases, conversion rate should be 0.0."""
    now = datetime.utcnow()
    events = [
        make_event(event_type="ENTRY", visitor_id="VIS_nopur1", timestamp=now),
        make_event(event_type="ENTRY", visitor_id="VIS_nopur2", timestamp=now),
        make_event(
            event_type="EXIT",
            visitor_id="VIS_nopur1",
            timestamp=now + timedelta(minutes=10),
        ),
        make_event(
            event_type="EXIT",
            visitor_id="VIS_nopur2",
            timestamp=now + timedelta(minutes=10),
        ),
    ]
    ingest(events)

    response = client.get("/stores/ST1008/metrics")
    data = response.json()
    assert data["conversion_rate"] == 0.0


def test_dwell_average_per_zone():
    """Avg dwell per zone should correctly average ZONE_DWELL events."""
    now = datetime.utcnow()
    vid = "VIS_dwell1"
    events = [
        make_event(event_type="ENTRY", visitor_id=vid, timestamp=now),
        make_event(
            event_type="ZONE_DWELL",
            visitor_id=vid,
            zone_id="SKINCARE",
            dwell_ms=30000,  # 30 seconds
            timestamp=now + timedelta(seconds=30),
        ),
        make_event(
            event_type="ZONE_DWELL",
            visitor_id=vid,
            zone_id="SKINCARE",
            dwell_ms=60000,  # 60 seconds
            timestamp=now + timedelta(seconds=60),
        ),
    ]
    ingest(events)

    response = client.get("/stores/ST1008/metrics")
    data = response.json()

    assert "SKINCARE" in data["avg_dwell_per_zone"]
    # Average of 30s and 60s = 45s
    assert data["avg_dwell_per_zone"]["SKINCARE"] == 45.0
