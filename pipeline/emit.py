"""
Event emitter — transforms tracked detections into the challenge event schema.
Handles:
- Entry/exit direction detection with hysteresis
- Group detection
- Zone enter/exit/dwell with occlusion tolerance
- Queue join/abandon with observation window
- Duplicate event prevention
"""

import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import requests

logger = logging.getLogger(__name__)

# Event type catalogue
EVENT_TYPES = [
    "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT",
    "ZONE_DWELL", "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY",
]

# ---------- CONFIGURABLE CONSTANTS ----------

# Entry threshold: fraction of frame height for the virtual threshold line
ENTRY_THRESHOLD_Y = 0.5

# Hysteresis margin: centroid must cross threshold by this fraction of frame height
ENTRY_HYSTERESIS_MARGIN = 0.05

# Group detection: entries within 2 seconds
GROUP_WINDOW_SECONDS = 2

# Dwell emission interval (seconds)
DWELL_INTERVAL = 30

# Zone exit timeout: seconds of absence before emitting ZONE_EXIT
ZONE_EXIT_TIMEOUT = 5.0

# Queue abandon observation window (seconds)
QUEUE_ABANDON_WINDOW_SECONDS = 120

# ---------- STATE ----------

# Direction detection: track_id → list of (frame_idx, y_position)
_track_y_history: Dict[int, List[Tuple[int, float]]] = {}

# Per-track entry/exit crossing flags to prevent duplicates
_track_has_entered: Set[int] = set()
_track_has_exited: Set[int] = set()

# Zone presence tracking: (visitor_id, zone_id) → entry_timestamp
_zone_presence: Dict[Tuple[str, str], datetime] = {}

# Zone last-seen tracking for occlusion tolerance: (visitor_id, zone_id) → last_seen_timestamp
_zone_last_seen: Dict[Tuple[str, str], datetime] = {}

# Visitor current zone: visitor_id → zone_id
_visitor_current_zone: Dict[str, str] = {}

# Dwell emission tracking: (visitor_id, zone_id) → last_dwell_emission_time
_last_dwell_emission: Dict[Tuple[str, str], datetime] = {}

# Session sequence counters: visitor_id → current sequence number
_session_seq: Dict[str, int] = {}

# Group tracking
_pending_entries: List[dict] = []

# Billing queue: set of visitor_ids currently in queue
_billing_queue: Set[str] = set()

# Queue observation window: visitor_id → {left_at, zone_id, camera_id, store_id}
_queue_observation: Dict[str, dict] = {}


def _next_session_seq(visitor_id: str) -> int:
    """Get and increment session sequence for a visitor."""
    if visitor_id not in _session_seq:
        _session_seq[visitor_id] = 0
    _session_seq[visitor_id] += 1
    return _session_seq[visitor_id]


def create_event(
    store_id: str,
    camera_id: str,
    visitor_id: str,
    event_type: str,
    timestamp: datetime,
    zone_id: str = None,
    dwell_ms: int = 0,
    is_staff: bool = False,
    confidence: float = 0.0,
    group_id: str = None,
    group_size: int = None,
    queue_depth: int = None,
    queue_position_at_join: int = None,
    wait_seconds: int = None,
    zone_hotspot_x: float = None,
    zone_hotspot_y: float = None,
    sku_zone: str = None,
) -> dict:
    """Create a single event in the challenge schema."""
    seq = _next_session_seq(visitor_id)

    return {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": timestamp.isoformat(),
        "ingested_at": datetime.utcnow().isoformat(),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": confidence,
        "is_face_hidden": confidence < 0.5,
        "group_id": group_id,
        "group_size": group_size,
        "metadata": {
            "queue_depth": queue_depth,
            "queue_position_at_join": queue_position_at_join,
            "wait_seconds": wait_seconds,
            "zone_hotspot_x": zone_hotspot_x,
            "zone_hotspot_y": zone_hotspot_y,
            "sku_zone": sku_zone,
            "session_seq": seq,
        },
    }


def detect_direction(
    track_id: int,
    centroid_y: float,
    frame_height: int,
    frame_idx: int,
) -> Optional[str]:
    """
    Detect ENTRY vs EXIT using centroid movement across virtual threshold line
    with hysteresis to prevent duplicate events from oscillation.

    Threshold: horizontal line at y = frame_height * ENTRY_THRESHOLD_Y
    Hysteresis: centroid must cross threshold ± margin before triggering.

    Returns "ENTRY", "EXIT", or None.
    """
    threshold = frame_height * ENTRY_THRESHOLD_Y
    margin = frame_height * ENTRY_HYSTERESIS_MARGIN

    if track_id not in _track_y_history:
        _track_y_history[track_id] = []

    _track_y_history[track_id].append((frame_idx, centroid_y))

    # Need at least 2 points to detect crossing
    history = _track_y_history[track_id]
    if len(history) < 2:
        return None

    prev_y = history[-2][1]
    curr_y = history[-1][1]

    # ENTRY: moving from above threshold to below (with hysteresis)
    if (prev_y < threshold - margin and curr_y >= threshold + margin
            and track_id not in _track_has_entered):
        _track_has_entered.add(track_id)
        _track_has_exited.discard(track_id)  # Allow re-exit after entry
        return "ENTRY"

    # EXIT: moving from below threshold to above (with hysteresis)
    elif (prev_y >= threshold + margin and curr_y < threshold - margin
            and track_id not in _track_has_exited):
        _track_has_exited.add(track_id)
        _track_has_entered.discard(track_id)  # Allow re-entry after exit
        return "EXIT"

    return None


def detect_groups(entries: List[dict]) -> List[dict]:
    """
    Group detection: if multiple tracks cross entry threshold within 2 seconds,
    assign same group_id. Each person still gets individual ENTRY event.
    """
    if not entries:
        return entries

    # Sort by timestamp
    sorted_entries = sorted(entries, key=lambda e: e["timestamp"])

    groups = []
    current_group = [sorted_entries[0]]

    for i in range(1, len(sorted_entries)):
        curr_ts = datetime.fromisoformat(sorted_entries[i]["timestamp"])
        prev_ts = datetime.fromisoformat(current_group[-1]["timestamp"])

        if (curr_ts - prev_ts).total_seconds() <= GROUP_WINDOW_SECONDS:
            current_group.append(sorted_entries[i])
        else:
            groups.append(current_group)
            current_group = [sorted_entries[i]]

    groups.append(current_group)

    # Assign group_id to groups of 2+
    for group in groups:
        if len(group) >= 2:
            gid = "GRP_" + str(uuid.uuid4())[:6]
            for event in group:
                event["group_id"] = gid
                event["group_size"] = len(group)

    return [e for g in groups for e in g]


def handle_zone_enter(
    visitor_id: str,
    zone_id: str,
    timestamp: datetime,
) -> None:
    """Track zone entry for dwell calculation."""
    key = (visitor_id, zone_id)
    _zone_presence[key] = timestamp
    _zone_last_seen[key] = timestamp
    _last_dwell_emission[key] = timestamp
    _visitor_current_zone[visitor_id] = zone_id


def handle_zone_exit(
    visitor_id: str,
    zone_id: str,
    timestamp: datetime,
) -> int:
    """
    Calculate dwell_ms on zone exit.
    Returns dwell_ms (time since ZONE_ENTER).
    """
    key = (visitor_id, zone_id)
    enter_time = _zone_presence.pop(key, None)
    _zone_last_seen.pop(key, None)
    _last_dwell_emission.pop(key, None)

    if _visitor_current_zone.get(visitor_id) == zone_id:
        del _visitor_current_zone[visitor_id]

    if enter_time:
        dwell_ms = int((timestamp - enter_time).total_seconds() * 1000)
        return max(0, dwell_ms)

    return 0


def update_zone_last_seen(visitor_id: str, zone_id: str, timestamp: datetime):
    """Update last-seen timestamp for a visitor in a zone (occlusion tolerance)."""
    key = (visitor_id, zone_id)
    if key in _zone_presence:
        _zone_last_seen[key] = timestamp


def check_zone_timeouts(timestamp: datetime) -> List[Tuple[str, str, int]]:
    """
    Check for visitors who haven't been seen in their zone for ZONE_EXIT_TIMEOUT.
    Returns list of (visitor_id, zone_id, dwell_ms) for expired zone presences.
    """
    expired = []
    keys_to_check = list(_zone_last_seen.keys())

    for key in keys_to_check:
        last_seen = _zone_last_seen.get(key)
        if last_seen and (timestamp - last_seen).total_seconds() >= ZONE_EXIT_TIMEOUT:
            visitor_id, zone_id = key
            dwell_ms = handle_zone_exit(visitor_id, zone_id, timestamp)
            expired.append((visitor_id, zone_id, dwell_ms))

    return expired


def check_zone_dwell(
    visitor_id: str,
    zone_id: str,
    timestamp: datetime,
) -> Optional[int]:
    """
    Check if visitor has been in zone for 30+ seconds.
    Emit ZONE_DWELL every 30 seconds of continued presence.
    Returns total dwell_ms if emission needed, None otherwise.
    """
    key = (visitor_id, zone_id)
    enter_time = _zone_presence.get(key)
    last_emission = _last_dwell_emission.get(key)

    if enter_time is None:
        return None

    total_dwell_seconds = (timestamp - enter_time).total_seconds()
    if total_dwell_seconds < DWELL_INTERVAL:
        return None

    time_since_last_emission = (timestamp - last_emission).total_seconds()
    if time_since_last_emission >= DWELL_INTERVAL:
        _last_dwell_emission[key] = timestamp
        return int(total_dwell_seconds * 1000)

    return None


def handle_queue_join(visitor_id: str):
    """Register a visitor joining the billing queue."""
    _billing_queue.add(visitor_id)
    # Clear any pending abandon observation
    _queue_observation.pop(visitor_id, None)


def handle_queue_leave(
    visitor_id: str,
    timestamp: datetime,
    zone_id: str,
    camera_id: str,
    store_id: str,
):
    """
    Start abandon observation window when a visitor leaves the billing zone.
    Does NOT immediately emit BILLING_QUEUE_ABANDON.
    """
    if visitor_id in _billing_queue:
        _queue_observation[visitor_id] = {
            "left_at": timestamp,
            "zone_id": zone_id,
            "camera_id": camera_id,
            "store_id": store_id,
        }


def check_queue_abandons(timestamp: datetime) -> List[dict]:
    """
    Check observation windows for queue abandonment.
    Returns list of abandon event dicts for visitors who:
    - Left the billing zone
    - Did NOT re-enter within QUEUE_ABANDON_WINDOW_SECONDS
    - Did NOT complete a purchase (no POS correlation check here — done at API level)
    """
    abandons = []
    expired = []

    for visitor_id, obs in _queue_observation.items():
        elapsed = (timestamp - obs["left_at"]).total_seconds()
        if elapsed >= QUEUE_ABANDON_WINDOW_SECONDS:
            # Observation window expired — emit abandon
            _billing_queue.discard(visitor_id)
            abandons.append({
                "visitor_id": visitor_id,
                "zone_id": obs["zone_id"],
                "camera_id": obs["camera_id"],
                "store_id": obs["store_id"],
                "wait_seconds": int(elapsed),
            })
            expired.append(visitor_id)

    for vid in expired:
        _queue_observation.pop(vid, None)

    return abandons


def get_current_queue_depth() -> int:
    """Get current billing queue depth (count of visitors in queue)."""
    return len(_billing_queue)


def process_and_emit(
    tracked_detections: list,
    store_id: str,
    camera_id: str,
    output_path: str = "data/events.jsonl",
    api_url: str = "http://localhost:8000",
) -> list:
    """
    Process tracked detections and emit events.
    Writes to JSONL file and POSTs to API.
    """
    events = []

    # Import reid, zone_mapper, and staff_detector here to avoid circular imports
    import numpy as np
    from pipeline.reid import get_or_assign_visitor_id, register_active_track, process_active_track_exits
    from pipeline.zone_mapper import get_zone, get_zone_category, load_zones, check_overlap_dedup
    from pipeline.staff_detector import (
        register_track_first_seen, register_track_seen,
        register_zone_visit, is_staff as check_is_staff,
    )

    # Ensure zones are loaded
    load_zones()

    # Frame dimensions (assume standard from first detection or default)
    frame_w = 1920
    frame_h = 1080

    # Get clip start time
    clip_start_time = None
    if tracked_detections:
        clip_start_time = min(datetime.fromisoformat(d["timestamp"]) for d in tracked_detections)

    # Pre-pass to find which tracks will trigger an actual ENTRY in this batch
    actual_entries = set()
    is_entry_cam = "entry" in camera_id.lower()
    if is_entry_cam:
        temp_y_history = {}
        temp_has_entered = set()
        for det in tracked_detections:
            tid = det["track_id"]
            cy = det["centroid"][1]
            threshold = frame_h * ENTRY_THRESHOLD_Y
            margin = frame_h * ENTRY_HYSTERESIS_MARGIN

            if tid not in temp_y_history:
                temp_y_history[tid] = []
            temp_y_history[tid].append(cy)

            if len(temp_y_history[tid]) >= 2:
                prev_y = temp_y_history[tid][-2]
                curr_y = temp_y_history[tid][-1]
                if (prev_y < threshold - margin and curr_y >= threshold + margin
                        and tid not in temp_has_entered):
                    temp_has_entered.add(tid)
                    actual_entries.add(tid)

    # Track which tracks have emitted an ENTRY event (real or synthetic)
    emitted_entries = set()

    # Track which visitors we've already seen in this batch (for zone tracking)
    seen_visitors: Set[str] = set()

    for det in tracked_detections:
        track_id = det["track_id"]
        cx, cy = det["centroid"]
        timestamp = datetime.fromisoformat(det["timestamp"])

        # Check active track exits on this camera based on the current detection's timestamp
        process_active_track_exits(camera_id, timestamp)

        # Get clothing color histogram if present
        hist_raw = det.get("colour_histogram")
        colour_hist = np.array(hist_raw) if hist_raw is not None else None

        # Get or assign visitor_id
        visitor_id, is_new = get_or_assign_visitor_id(
            track_id=track_id,
            camera_id=camera_id,
            centroid=(cx, cy),
            timestamp=timestamp,
            colour_histogram=colour_hist,
        )

        # Register track with staff detector
        register_track_first_seen(track_id, timestamp)
        register_track_seen(track_id, timestamp)

        # Register active track for cross-camera dedup
        register_active_track(
            camera_id=camera_id,
            track_id=track_id,
            visitor_id=visitor_id,
            timestamp=timestamp,
            centroid=(cx, cy),
            colour_histogram=colour_hist,
        )

        seen_visitors.add(visitor_id)

        # Check staff status
        staff_flag, staff_score, staff_signals = check_is_staff(
            track_id=track_id,
            store_id=store_id,
            colour_histogram=colour_hist,
            timestamp=timestamp,
        )

        # COLD START: Generate Synthetic Entry if track won't trigger a real entry and hasn't had one yet
        if track_id not in actual_entries and track_id not in emitted_entries:
            synthetic_event = create_event(
                store_id=store_id,
                camera_id=camera_id,
                visitor_id=visitor_id,
                event_type="ENTRY",
                timestamp=clip_start_time or timestamp,
                confidence=det["confidence"],
                is_staff=staff_flag,
            )
            events.append(synthetic_event)
            emitted_entries.add(track_id)

        # Check direction for ENTRY/EXIT at entry camera
        if is_entry_cam:
            direction = detect_direction(track_id, cy, frame_h, det["frame_idx"])
            if direction == "ENTRY":
                event = create_event(
                    store_id=store_id,
                    camera_id=camera_id,
                    visitor_id=visitor_id,
                    event_type="ENTRY",
                    timestamp=timestamp,
                    confidence=det["confidence"],
                    is_staff=staff_flag,
                )
                events.append(event)
                emitted_entries.add(track_id)

            elif direction == "EXIT":
                event = create_event(
                    store_id=store_id,
                    camera_id=camera_id,
                    visitor_id=visitor_id,
                    event_type="EXIT",
                    timestamp=timestamp,
                    confidence=det["confidence"],
                    is_staff=staff_flag,
                )
                events.append(event)

        # Check zone for all cameras
        zone_id, zone_name = get_zone(cx, cy, frame_w, frame_h, store_id, camera_id)

        if zone_id:
            # Get business category for sku_zone metadata
            category = get_zone_category(zone_id)

            # Register zone visit with staff detector
            register_zone_visit(track_id, zone_id, timestamp)

            key = (visitor_id, zone_id)

            if key not in _zone_presence:
                # Check if visitor was in a different zone → emit ZONE_EXIT for previous
                prev_zone = _visitor_current_zone.get(visitor_id)
                if prev_zone and prev_zone != zone_id:
                    prev_key = (visitor_id, prev_zone)
                    if prev_key in _zone_presence:
                        dwell_ms = handle_zone_exit(visitor_id, prev_zone, timestamp)
                        prev_category = get_zone_category(prev_zone)
                        exit_event = create_event(
                            store_id=store_id,
                            camera_id=camera_id,
                            visitor_id=visitor_id,
                            event_type="ZONE_EXIT",
                            timestamp=timestamp,
                            zone_id=prev_zone,
                            dwell_ms=dwell_ms,
                            confidence=det["confidence"],
                            is_staff=staff_flag,
                            sku_zone=prev_category,
                        )
                        events.append(exit_event)

                        # If leaving billing zone, start queue observation
                        if "billing" in prev_zone.lower():
                            handle_queue_leave(
                                visitor_id, timestamp, prev_zone,
                                camera_id, store_id,
                            )

                # Overlap dedup check
                if not check_overlap_dedup(
                    visitor_id, zone_id, camera_id, timestamp,
                    (cx / frame_w, cy / frame_h), store_id,
                ):
                    # Deduplicated — skip this zone event
                    update_zone_last_seen(visitor_id, zone_id, timestamp)
                    continue

                # ZONE_ENTER
                handle_zone_enter(visitor_id, zone_id, timestamp)

                event = create_event(
                    store_id=store_id,
                    camera_id=camera_id,
                    visitor_id=visitor_id,
                    event_type="ZONE_ENTER",
                    timestamp=timestamp,
                    zone_id=zone_id,
                    confidence=det["confidence"],
                    zone_hotspot_x=cx,
                    zone_hotspot_y=cy,
                    is_staff=staff_flag,
                    sku_zone=category,
                )
                events.append(event)

                # Check if this is billing zone for queue tracking
                if "billing" in zone_id.lower():
                    # Cancel any pending abandon observation (visitor returned)
                    _queue_observation.pop(visitor_id, None)

                    queue_depth = get_current_queue_depth()
                    handle_queue_join(visitor_id)

                    if queue_depth > 0:
                        queue_event = create_event(
                            store_id=store_id,
                            camera_id=camera_id,
                            visitor_id=visitor_id,
                            event_type="BILLING_QUEUE_JOIN",
                            timestamp=timestamp,
                            zone_id=zone_id,
                            confidence=det["confidence"],
                            queue_depth=queue_depth + 1,
                            queue_position_at_join=queue_depth + 1,
                            is_staff=staff_flag,
                            sku_zone=category,
                        )
                        events.append(queue_event)
            else:
                # Already in zone — update last seen and check for ZONE_DWELL
                update_zone_last_seen(visitor_id, zone_id, timestamp)

                dwell_ms = check_zone_dwell(visitor_id, zone_id, timestamp)
                if dwell_ms is not None:
                    event = create_event(
                        store_id=store_id,
                        camera_id=camera_id,
                        visitor_id=visitor_id,
                        event_type="ZONE_DWELL",
                        timestamp=timestamp,
                        zone_id=zone_id,
                        dwell_ms=dwell_ms,
                        confidence=det["confidence"],
                        zone_hotspot_x=cx,
                        zone_hotspot_y=cy,
                        is_staff=staff_flag,
                        sku_zone=category,
                    )
                    events.append(event)

    # --- Post-frame processing ---

    # Check zone exit timeouts (occlusion tolerance)
    if tracked_detections:
        last_timestamp = datetime.fromisoformat(tracked_detections[-1]["timestamp"])
        expired_zones = check_zone_timeouts(last_timestamp)
        for visitor_id, zone_id, dwell_ms in expired_zones:
            category = get_zone_category(zone_id)
            exit_event = create_event(
                store_id=store_id,
                camera_id=camera_id,
                visitor_id=visitor_id,
                event_type="ZONE_EXIT",
                timestamp=last_timestamp,
                zone_id=zone_id,
                dwell_ms=dwell_ms,
                sku_zone=category,
            )
            events.append(exit_event)

            # If leaving billing zone, start queue observation
            if "billing" in zone_id.lower():
                handle_queue_leave(
                    visitor_id, last_timestamp, zone_id,
                    camera_id, store_id,
                )

        # Check queue abandonment observation windows
        abandons = check_queue_abandons(last_timestamp)
        for abandon in abandons:
            category = get_zone_category(abandon["zone_id"])
            abandon_event = create_event(
                store_id=abandon["store_id"],
                camera_id=abandon["camera_id"],
                visitor_id=abandon["visitor_id"],
                event_type="BILLING_QUEUE_ABANDON",
                timestamp=last_timestamp,
                zone_id=abandon["zone_id"],
                wait_seconds=abandon["wait_seconds"],
                sku_zone=category,
            )
            events.append(abandon_event)

    # Group detection on ENTRY events
    entry_events = [e for e in events if e["event_type"] == "ENTRY"]
    if entry_events:
        grouped = detect_groups(entry_events)
        # Update group info in the original events list
        group_map = {e["event_id"]: e for e in grouped}
        for i, e in enumerate(events):
            if e["event_id"] in group_map:
                events[i] = group_map[e["event_id"]]

    # Write to JSONL file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "a") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    logger.info(f"Wrote {len(events)} events to {output_path}")

    # POST to API
    if api_url and events:
        try:
            batch_size = 500
            for i in range(0, len(events), batch_size):
                batch = events[i:i + batch_size]
                response = requests.post(
                    f"{api_url}/events/ingest",
                    json={"events": batch},
                    timeout=30,
                )
                if response.status_code == 200:
                    result = response.json()
                    logger.info(
                        f"Ingested batch: accepted={result.get('accepted', 0)}, "
                        f"rejected={result.get('rejected', 0)}, "
                        f"duplicate={result.get('duplicate', 0)}"
                    )
                else:
                    logger.warning(f"Ingest failed with status {response.status_code}: {response.text}")
        except Exception as e:
            logger.warning(f"Failed to POST events to API: {e}")

    return events


def reset_emitter():
    """Reset emitter state between clips."""
    global _track_y_history, _zone_presence, _zone_last_seen, _last_dwell_emission
    global _session_seq, _pending_entries, _billing_queue, _queue_observation
    global _track_has_entered, _track_has_exited, _visitor_current_zone
    _track_y_history = {}
    _track_has_entered = set()
    _track_has_exited = set()
    _zone_presence = {}
    _zone_last_seen = {}
    _visitor_current_zone = {}
    _last_dwell_emission = {}
    _session_seq = {}
    _pending_entries = []
    _billing_queue = set()
    _queue_observation = {}
