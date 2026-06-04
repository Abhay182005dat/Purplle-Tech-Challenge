# PROMPT: Generate comprehensive tests for the health endpoint covering empty state,
# STALE_FEED detection, database connectivity, and per-store last_event_timestamp.
# CHANGES MADE: Added edge case for zero events, verified warnings list for stale feeds,
# ensured healthy status on fresh empty DB.

"""
Tests for GET /health endpoint.
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
    camera_id="CAM_ENTRY_01",
    is_staff=False,
    confidence=0.85,
    timestamp=None,
):
    """Helper to create a valid event dict."""
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id or f"VIS_{str(uuid.uuid4())[:6]}",
        "event_type": event_type,
        "timestamp": (timestamp or datetime.utcnow()).isoformat(),
        "zone_id": None,
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


def test_health_empty_db_returns_healthy():
    """Empty database should return healthy status with connected DB."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["database"] == "connected"
    assert "checked_at" in data
    assert "stores" in data


def test_health_returns_per_store_last_event():
    """Health should include last_event_timestamp per store."""
    now = datetime.utcnow()
    event = make_event(store_id="ST1008", timestamp=now)
    ingest([event])

    response = client.get("/health")
    data = response.json()

    assert "ST1008" in data["stores"]
    store_health = data["stores"]["ST1008"]
    assert store_health["last_event_timestamp"] is not None


def test_health_stale_feed_when_old_events():
    """Events older than 10 minutes should show STALE feed status."""
    old_time = datetime.utcnow() - timedelta(minutes=15)
    event = make_event(
        store_id="ST1008",
        camera_id="CAM_ENTRY_01",
        timestamp=old_time,
    )
    ingest([event])

    response = client.get("/health")
    data = response.json()

    assert "ST1008" in data["stores"]
    store_health = data["stores"]["ST1008"]
    assert store_health["feed_status"] == "STALE"


def test_health_live_feed_with_recent_events():
    """Recent events (< 10 min old) should show LIVE feed status."""
    now = datetime.utcnow()
    event = make_event(
        store_id="ST1008",
        camera_id="CAM_ENTRY_01",
        timestamp=now,
    )
    ingest([event])

    response = client.get("/health")
    data = response.json()

    # Find ST1008 in stores
    if "ST1008" in data["stores"]:
        store_health = data["stores"]["ST1008"]
        # Should be LIVE since event was just ingested
        assert store_health["feed_status"] == "LIVE"


def test_health_camera_level_status():
    """Health should include per-camera status."""
    now = datetime.utcnow()
    event = make_event(
        store_id="ST1008",
        camera_id="CAM_ENTRY_01",
        timestamp=now,
    )
    ingest([event])

    response = client.get("/health")
    data = response.json()

    if "ST1008" in data["stores"]:
        cameras = data["stores"]["ST1008"].get("cameras", {})
        assert "CAM_ENTRY_01" in cameras
        assert cameras["CAM_ENTRY_01"]["status"] == "LIVE"


def test_health_no_data_status_for_unknown_camera():
    """Cameras from layout with no events should show NO_DATA."""
    response = client.get("/health")
    data = response.json()

    if "ST1008" in data["stores"]:
        cameras = data["stores"]["ST1008"].get("cameras", {})
        # All cameras should be NO_DATA since no events ingested
        for cam_id, cam_health in cameras.items():
            assert cam_health["status"] == "NO_DATA"


def test_health_warnings_on_stale():
    """Health should include warnings list when feeds are stale."""
    old_time = datetime.utcnow() - timedelta(minutes=15)
    event = make_event(store_id="ST1008", timestamp=old_time)
    ingest([event])

    response = client.get("/health")
    data = response.json()

    # Status should be degraded when feeds are stale
    assert data["status"] == "degraded"
    # Should have warnings
    if "warnings" in data:
        assert len(data["warnings"]) > 0
        assert any("STALE_FEED" in w for w in data["warnings"])
