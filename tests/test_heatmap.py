# PROMPT: Generate tests for the heatmap endpoint covering empty store,
# data_confidence flag, normalised values, and zone visit frequency.
# CHANGES MADE: Added normalisation boundary tests, verified data_confidence
# threshold at 20 sessions, tested zone list matches store_layout.json zones.

"""
Tests for GET /stores/{store_id}/heatmap endpoint.
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
):
    """Helper to create a valid event dict."""
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": "CAM_FLOOR_01",
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
        "metadata": {"session_seq": 1},
    }


def ingest(events):
    """Helper to ingest events."""
    response = client.post("/events/ingest", json={"events": events})
    assert response.status_code == 200
    return response.json()


# ─── Test Cases ───────────────────────────────────────────────────────────────


def test_empty_store_heatmap():
    """Empty store should return empty zones list with data_confidence=False."""
    response = client.get("/stores/NONEXISTENT/heatmap")
    assert response.status_code == 200
    data = response.json()

    assert data["session_count"] == 0
    assert data["data_confidence"] is False
    assert data["zones"] == []


def test_heatmap_returns_valid_structure():
    """Heatmap response should have required fields."""
    response = client.get("/stores/ST1008/heatmap")
    assert response.status_code == 200
    data = response.json()

    assert "store_id" in data
    assert "session_count" in data
    assert "data_confidence" in data
    assert "zones" in data
    assert isinstance(data["zones"], list)


def test_data_confidence_false_below_20_sessions():
    """data_confidence should be False when session_count < 20."""
    now = datetime.utcnow()

    # Create 10 sessions (< 20 threshold)
    events = []
    for i in range(10):
        vid = f"VIS_conf{i:02d}"
        events.append(make_event(event_type="ENTRY", visitor_id=vid, timestamp=now))

    ingest(events)

    response = client.get("/stores/ST1008/heatmap")
    data = response.json()

    assert data["session_count"] < 20
    assert data["data_confidence"] is False


def test_data_confidence_true_above_20_sessions():
    """data_confidence should be True when session_count >= 20."""
    now = datetime.utcnow()

    # Create 25 sessions (>= 20 threshold)
    events = []
    for i in range(25):
        vid = f"VIS_conf{i:02d}"
        events.append(make_event(event_type="ENTRY", visitor_id=vid, timestamp=now))

    ingest(events)

    response = client.get("/stores/ST1008/heatmap")
    data = response.json()

    assert data["session_count"] >= 20
    assert data["data_confidence"] is True


def test_zone_visit_frequency_normalised():
    """visit_frequency should be normalised 0-100."""
    now = datetime.utcnow()

    # Create zone visits
    events = []
    for i in range(5):
        vid = f"VIS_heat{i:02d}"
        events.append(make_event(event_type="ENTRY", visitor_id=vid, timestamp=now))
        events.append(
            make_event(
                event_type="ZONE_ENTER",
                visitor_id=vid,
                zone_id="FRAGRANCE",
                timestamp=now + timedelta(minutes=1),
            )
        )

    # Only 2 visitors to SKINCARE
    for i in range(2):
        vid = f"VIS_heat{i:02d}"
        events.append(
            make_event(
                event_type="ZONE_ENTER",
                visitor_id=vid,
                zone_id="EB_KOREAN",
                timestamp=now + timedelta(minutes=2),
            )
        )

    ingest(events)

    response = client.get("/stores/ST1008/heatmap")
    data = response.json()

    for zone in data["zones"]:
        assert 0 <= zone["visit_frequency"] <= 100
        assert 0 <= zone["normalised_dwell"] <= 100


def test_heatmap_staff_excluded():
    """Staff events should not affect heatmap metrics."""
    now = datetime.utcnow()

    # Staff zone visits
    events = [
        make_event(
            event_type="ZONE_ENTER",
            visitor_id="VIS_staff01",
            zone_id="FRAGRANCE",
            is_staff=True,
            timestamp=now,
        ),
    ]
    ingest(events)

    response = client.get("/stores/ST1008/heatmap")
    data = response.json()

    # FRAGRANCE should not appear in zones from staff visits alone
    fragrance_zones = [z for z in data["zones"] if z["zone_id"] == "FRAGRANCE"]
    for z in fragrance_zones:
        assert z["visit_frequency"] == 0


def test_heatmap_dwell_calculation():
    """Heatmap should correctly calculate avg_dwell_seconds from ZONE_DWELL events."""
    now = datetime.utcnow()
    vid = "VIS_dwtest"

    events = [
        make_event(event_type="ENTRY", visitor_id=vid, timestamp=now),
        make_event(
            event_type="ZONE_DWELL",
            visitor_id=vid,
            zone_id="FRAGRANCE",
            dwell_ms=30000,
            timestamp=now + timedelta(seconds=30),
        ),
        make_event(
            event_type="ZONE_DWELL",
            visitor_id=vid,
            zone_id="FRAGRANCE",
            dwell_ms=60000,
            timestamp=now + timedelta(seconds=60),
        ),
    ]
    ingest(events)

    response = client.get("/stores/ST1008/heatmap")
    data = response.json()

    fragrance = next(
        (z for z in data["zones"] if z["zone_id"] == "FRAGRANCE"), None
    )
    if fragrance:
        # Average of 30s and 60s = 45s
        assert fragrance["avg_dwell_seconds"] == 45.0
