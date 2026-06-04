# PROMPT: Generate tests for the funnel endpoint
# CHANGES MADE: Added re-entry deduplication, subset enforcement, and drop-off calculation tests

"""
Tests for GET /stores/{store_id}/funnel endpoint.
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
    timestamp=None,
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
        "dwell_ms": 0,
        "is_staff": is_staff,
        "confidence": confidence,
        "is_face_hidden": False,
        "group_id": None,
        "group_size": None,
        "metadata": {"session_seq": 1},
    }


def ingest(events):
    """Helper to ingest events."""
    response = client.post("/events/ingest", json={"events": events})
    assert response.status_code == 200
    return response.json()


# ─── Test Cases ───────────────────────────────────────────────────────────────


def test_reentry_not_double_counted():
    """
    Same visitor_id: EXIT then REENTRY then EXIT.
    Funnel should count the visitor ONCE, not twice.
    """
    now = datetime.utcnow()
    vid = "VIS_reent1"

    events = [
        # First visit
        make_event(event_type="ENTRY", visitor_id=vid, timestamp=now),
        make_event(
            event_type="ZONE_ENTER",
            visitor_id=vid,
            zone_id="SKINCARE",
            timestamp=now + timedelta(minutes=5),
        ),
        make_event(
            event_type="EXIT",
            visitor_id=vid,
            timestamp=now + timedelta(minutes=15),
        ),
        # Re-entry
        make_event(
            event_type="REENTRY",
            visitor_id=vid,
            timestamp=now + timedelta(minutes=30),
        ),
        make_event(
            event_type="ENTRY",
            visitor_id=vid,
            timestamp=now + timedelta(minutes=30, seconds=1),
        ),
        make_event(
            event_type="EXIT",
            visitor_id=vid,
            timestamp=now + timedelta(minutes=45),
        ),
    ]
    ingest(events)

    response = client.get("/stores/ST1008/funnel")
    data = response.json()

    # Should count visitor only once in ENTRY stage
    entry_stage = next(s for s in data["stages"] if s["stage"] == "ENTRY")
    assert entry_stage["count"] == 1  # Not 2


def test_funnel_stages_are_subsets():
    """
    Each funnel stage count must be <= previous stage count.
    ZONE_VISIT <= ENTRY, BILLING <= ZONE_VISIT, PURCHASE <= BILLING.
    """
    now = datetime.utcnow()

    # Create visitors at various funnel stages
    events = []
    for i in range(10):
        vid = f"VIS_fun{i:03d}"
        events.append(make_event(event_type="ENTRY", visitor_id=vid, timestamp=now))

        # Only some visit zones
        if i < 7:
            events.append(
                make_event(
                    event_type="ZONE_ENTER",
                    visitor_id=vid,
                    zone_id="SKINCARE",
                    timestamp=now + timedelta(minutes=2),
                )
            )

        # Only some reach billing
        if i < 4:
            events.append(
                make_event(
                    event_type="BILLING_QUEUE_JOIN",
                    visitor_id=vid,
                    zone_id="BILLING",
                    timestamp=now + timedelta(minutes=10),
                )
            )

    ingest(events)

    response = client.get("/stores/ST1008/funnel")
    data = response.json()

    stages = {s["stage"]: s["count"] for s in data["stages"]}

    assert stages["ZONE_VISIT"] <= stages["ENTRY"]
    assert stages["BILLING_QUEUE"] <= stages["ZONE_VISIT"]
    assert stages["PURCHASE"] <= stages["BILLING_QUEUE"]


def test_drop_off_percentages_correct():
    """Drop-off percentages should be calculated correctly."""
    now = datetime.utcnow()

    # 4 enter, 2 visit zones, 1 reaches billing
    events = []
    for i in range(4):
        vid = f"VIS_drop{i:02d}"
        events.append(make_event(event_type="ENTRY", visitor_id=vid, timestamp=now))

        if i < 2:
            events.append(
                make_event(
                    event_type="ZONE_ENTER",
                    visitor_id=vid,
                    zone_id="HAIRCARE",
                    timestamp=now + timedelta(minutes=3),
                )
            )

        if i < 1:
            events.append(
                make_event(
                    event_type="BILLING_QUEUE_JOIN",
                    visitor_id=vid,
                    zone_id="BILLING",
                    timestamp=now + timedelta(minutes=8),
                )
            )

    ingest(events)

    response = client.get("/stores/ST1008/funnel")
    data = response.json()

    stages = {s["stage"]: s for s in data["stages"]}

    # ENTRY → ZONE_VISIT: (4-2)/4 = 50%
    assert stages["ZONE_VISIT"]["drop_off_pct"] == 50.0

    # ZONE_VISIT → BILLING: (2-1)/2 = 50%
    assert stages["BILLING_QUEUE"]["drop_off_pct"] == 50.0


def test_empty_store_funnel_returns_zeros():
    """Empty store should return all zero counts and 0.0 rates."""
    response = client.get("/stores/ST1008/funnel")
    assert response.status_code == 200
    data = response.json()

    for stage in data["stages"]:
        assert stage["count"] == 0
        assert stage["drop_off_pct"] == 0.0

    assert data["conversion_rate"] == 0.0
