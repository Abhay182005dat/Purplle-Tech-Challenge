"""
Tracker — maintains track history from ByteTrack output.

ByteTrack identity assignment is handled by YOLOv8s model.track().
This module receives ByteTrack-assigned track IDs and maintains:
- Track history (centroids over time)
- Track metadata (first/last seen, bbox history)
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Track history: track_id → list of centroids
track_history: Dict[int, List[Tuple[float, float]]] = {}

# Track metadata: track_id → {first_frame, last_frame, first_ts, last_ts}
track_metadata: Dict[int, dict] = {}


def compute_centroid(bbox: list) -> Tuple[float, float]:
    """Compute centroid from [x1, y1, x2, y2] bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def update_track_history(
    track_id: int,
    centroid: Tuple[float, float],
    frame_idx: int,
    timestamp: str,
):
    """
    Update track history with a new detection from ByteTrack.
    Called per-detection after ByteTrack assigns the track_id.
    """
    if track_id not in track_history:
        track_history[track_id] = []
        track_metadata[track_id] = {
            "first_frame": frame_idx,
            "last_frame": frame_idx,
            "first_ts": timestamp,
            "last_ts": timestamp,
        }

    track_history[track_id].append(centroid)
    track_metadata[track_id]["last_frame"] = frame_idx
    track_metadata[track_id]["last_ts"] = timestamp


def build_tracked_detections(
    detections: list,
    camera_id: str,
) -> list:
    """
    Accept detections that already have ByteTrack track_ids.
    Enrich with centroid, update history, and return unified format.

    Each input detection must have:
        track_id, bbox, frame_idx, timestamp, confidence

    Returns list of:
    {
        "track_id": int,
        "bbox": [x1, y1, x2, y2],
        "centroid": [cx, cy],
        "frame_idx": int,
        "timestamp": string,
        "camera_id": string,
        "confidence": float
    }
    """
    tracked = []
    for det in detections:
        track_id = det["track_id"]
        cx, cy = compute_centroid(det["bbox"])

        update_track_history(track_id, (cx, cy), det["frame_idx"], det["timestamp"])

        tracked.append({
            "track_id": track_id,
            "bbox": det["bbox"],
            "centroid": [cx, cy],
            "frame_idx": det["frame_idx"],
            "timestamp": det["timestamp"],
            "camera_id": camera_id,
            "confidence": det["confidence"],
        })

    return tracked


def get_track_history() -> Dict[int, List[Tuple[float, float]]]:
    """Return the track history dict."""
    return track_history.copy()


def get_track_metadata(track_id: int) -> dict:
    """Return metadata for a specific track."""
    return track_metadata.get(track_id, {})


def get_all_track_ids() -> List[int]:
    """Return all known track IDs."""
    return list(track_history.keys())


def reset_tracker():
    """Reset tracker state between clips."""
    global track_history, track_metadata
    track_history = {}
    track_metadata = {}
