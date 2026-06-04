"""
Event transformer — converts raw pipeline events (sample_events.jsonl format)
to the API-compatible event schema.

The raw pipeline uses different field names:
  - id_token → visitor_id
  - store_code → store_id
  - event_timestamp / event_time → timestamp
  - entry/exit/zone_entered/zone_exited/queue_completed/queue_abandoned → ENTRY/EXIT/ZONE_ENTER/ZONE_EXIT/BILLING_QUEUE_JOIN/BILLING_QUEUE_ABANDON
"""

import json
import uuid
import sys
import os
from datetime import datetime
from typing import Optional

# Map raw event types to API event types
EVENT_TYPE_MAP = {
    "entry": "ENTRY",
    "exit": "EXIT",
    "zone_entered": "ZONE_ENTER",
    "zone_exited": "ZONE_EXIT",
    "zone_dwell": "ZONE_DWELL",
    "queue_completed": "BILLING_QUEUE_JOIN",
    "queue_abandoned": "BILLING_QUEUE_ABANDON",
    "reentry": "REENTRY",
}

# Store code mapping (raw store codes → API store IDs)
STORE_CODE_MAP = {
    "store_1076": "ST1076",
    "ST1076": "ST1076",
    "ST1008": "ST1008",
    "Store 1": "Store 1",
}


def transform_event(raw: dict) -> Optional[dict]:
    """
    Transform a single raw pipeline event to the API event schema.
    Returns None if the event cannot be transformed.
    """
    raw_type = raw.get("event_type", "").lower()
    api_type = EVENT_TYPE_MAP.get(raw_type)

    if api_type is None:
        return None

    # Get visitor_id
    visitor_id = raw.get("id_token") or raw.get("visitor_id") or f"VIS_{str(uuid.uuid4())[:6]}"

    # Get store_id
    store_code = raw.get("store_code") or raw.get("store_id") or "UNKNOWN"
    store_id = STORE_CODE_MAP.get(store_code, store_code)

    # Get camera_id
    camera_id = raw.get("camera_id", "CAM_UNKNOWN")

    # Get timestamp
    timestamp = (
        raw.get("event_timestamp")
        or raw.get("event_time")
        or raw.get("queue_join_ts")
        or datetime.utcnow().isoformat()
    )

    # Get zone_id
    zone_id = raw.get("zone_id")
    zone_name = raw.get("zone_name")

    # Get confidence (raw events may not have this)
    confidence = raw.get("confidence", 0.85)

    # Staff flag
    is_staff = raw.get("is_staff", False)

    # Face hidden
    is_face_hidden = raw.get("is_face_hidden", False)

    # Group info
    group_id = raw.get("group_id")
    group_size = raw.get("group_size")

    # Queue-specific metadata
    queue_depth = None
    queue_position_at_join = raw.get("queue_position_at_join")
    wait_seconds = raw.get("wait_seconds")
    zone_hotspot_x = raw.get("zone_hotspot_x")
    zone_hotspot_y = raw.get("zone_hotspot_y")

    # Dwell
    dwell_ms = 0
    if api_type == "BILLING_QUEUE_JOIN" and raw.get("queue_position_at_join"):
        queue_depth = raw["queue_position_at_join"]
    if api_type == "BILLING_QUEUE_ABANDON":
        is_staff = False  # Queue abandonment is always customer

    # Build API event
    event = {
        "event_id": raw.get("queue_event_id") or str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": api_type,
        "timestamp": timestamp,
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": confidence,
        "is_face_hidden": is_face_hidden,
        "group_id": group_id,
        "group_size": group_size,
        "metadata": {
            "queue_depth": queue_depth,
            "queue_position_at_join": queue_position_at_join,
            "wait_seconds": wait_seconds,
            "zone_hotspot_x": zone_hotspot_x,
            "zone_hotspot_y": zone_hotspot_y,
            "sku_zone": zone_name,
            "session_seq": 1,
        },
    }

    return event


def transform_jsonl(input_path: str, output_path: str = None) -> list:
    """
    Transform a JSONL file of raw events to API-compatible events.
    Returns list of transformed events.
    """
    transformed = []

    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                event = transform_event(raw)
                if event:
                    transformed.append(event)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            for event in transformed:
                f.write(json.dumps(event) + "\n")
        print(f"Wrote {len(transformed)} transformed events to {output_path}")

    return transformed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transform raw pipeline events to API schema")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", default=None, help="Output JSONL file")
    parser.add_argument("--ingest", action="store_true", help="Also POST to API")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")

    args = parser.parse_args()

    events = transform_jsonl(args.input, args.output)
    print(f"Transformed {len(events)} events")

    if args.ingest and events:
        import requests

        batch_size = 500
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            try:
                r = requests.post(
                    f"{args.api_url}/events/ingest",
                    json={"events": batch},
                    timeout=30,
                )
                if r.status_code == 200:
                    result = r.json()
                    print(f"Ingested: accepted={result['accepted']}, rejected={result['rejected']}, duplicate={result['duplicate']}")
                else:
                    print(f"Ingest failed: {r.status_code} {r.text}")
            except Exception as e:
                print(f"Error ingesting: {e}")
