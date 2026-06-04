"""
Cross-camera Re-ID using composite scoring:
  Layer 1: ByteTrack within-camera continuity (handled by tracker.py)
  Layer 2: Spatial-temporal matching with exit registry
  Layer 3: Colour histogram similarity (HSV)
  Layer 4: Exit position proximity

Composite score = colour (0.5) + temporal (0.3) + spatial (0.2)
"""

import logging
import math
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Exit registry: camera_id → list of exited tracks
exit_registry: Dict[str, List[dict]] = {}

# Active tracks: camera_id → {track_id → last_detection_info}
_active_tracks: Dict[str, Dict[int, dict]] = {}

# Visitor ID mapping: track_id → visitor_id
visitor_map: Dict[int, str] = {}

# Camera adjacency rules
CAMERA_ADJACENCY = {
    "CAM_ENTRY_01": ["CAM_FLOOR_01", "CAM_FLOOR_02"],
    "CAM_FLOOR_01": ["CAM_ENTRY_01", "CAM_BILLING_01", "CAM_FLOOR_02"],
    "CAM_FLOOR_02": ["CAM_ENTRY_01", "CAM_BILLING_01", "CAM_FLOOR_01"],
    "CAM_BILLING_01": ["CAM_FLOOR_01", "CAM_FLOOR_02"],
    # Entry is NOT adjacent to Billing
}

# Disappearance threshold (seconds) to consider a track as exited
DISAPPEAR_SECONDS = 5.0

# Temporal matching window (seconds)
TEMPORAL_WINDOW = 30

# Composite score threshold for cross-camera match
COMPOSITE_SCORE_THRESHOLD = 0.40

# Score weights
WEIGHT_COLOUR = 0.5
WEIGHT_TEMPORAL = 0.3
WEIGHT_SPATIAL = 0.2

# Maximum spatial distance for exit-to-entry position match (normalised 0-1)
MAX_SPATIAL_DISTANCE = 0.5


def generate_visitor_id() -> str:
    """Generate new visitor_id: VIS_ + first 6 chars of uuid4."""
    return "VIS_" + str(uuid.uuid4())[:6]


def extract_clothing_histogram(frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
    """
    Extract HSV histogram from clothing region of bounding box.
    Top 60% height, centre 60% width — avoids background bleed at edges.
    """
    try:
        x1, y1, x2, y2 = [int(b) for b in bbox]
        h = y2 - y1
        w = x2 - x1

        # Clothing region: top 60% height, centre 60% width
        crop_y1 = y1
        crop_y2 = y1 + int(h * 0.6)
        crop_x1 = x1 + int(w * 0.2)
        crop_x2 = x2 - int(w * 0.2)

        # Bounds check
        crop_y1 = max(0, crop_y1)
        crop_y2 = min(frame.shape[0], crop_y2)
        crop_x1 = max(0, crop_x1)
        crop_x2 = min(frame.shape[1], crop_x2)

        if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
            return None

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # HSV histogram: bins=[50, 60], ranges=[0,180, 0,256]
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [50, 60], [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
        return hist

    except Exception as e:
        logger.warning(f"Error extracting clothing histogram: {e}")
        return None


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compare two histograms using correlation."""
    if hist1 is None or hist2 is None:
        return 0.0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def _compute_spatial_score(
    exit_centroid: Tuple[float, float],
    entry_centroid: Tuple[float, float],
) -> float:
    """
    Compute spatial proximity score between exit and entry positions.
    Returns 0.0 to 1.0 (1.0 = perfect match, 0.0 = too far).
    """
    dist = math.sqrt(
        (exit_centroid[0] - entry_centroid[0]) ** 2 +
        (exit_centroid[1] - entry_centroid[1]) ** 2
    )
    if dist >= MAX_SPATIAL_DISTANCE:
        return 0.0
    return 1.0 - (dist / MAX_SPATIAL_DISTANCE)


def _compute_temporal_score(time_diff_seconds: float) -> float:
    """
    Compute temporal proximity score.
    Returns 1.0 for 0-second gap, decaying to 0.0 at TEMPORAL_WINDOW.
    """
    if time_diff_seconds < 0 or time_diff_seconds > TEMPORAL_WINDOW:
        return 0.0
    return 1.0 - (time_diff_seconds / TEMPORAL_WINDOW)


def register_active_track(
    camera_id: str,
    track_id: int,
    visitor_id: str,
    timestamp: datetime,
    centroid: Tuple[float, float],
    colour_histogram: Optional[np.ndarray] = None,
):
    """Register or update an active track on a camera."""
    if camera_id not in _active_tracks:
        _active_tracks[camera_id] = {}

    _active_tracks[camera_id][track_id] = {
        "visitor_id": visitor_id,
        "timestamp": timestamp,
        "centroid": centroid,
        "colour_histogram": colour_histogram,
    }


def register_exit(
    camera_id: str,
    track_id: int,
    visitor_id: str,
    exit_time: datetime,
    last_centroid: Tuple[float, float],
    colour_histogram: Optional[np.ndarray] = None,
):
    """Register a track exit in the exit registry."""
    if camera_id not in exit_registry:
        exit_registry[camera_id] = []

    exit_registry[camera_id].append({
        "track_id": track_id,
        "visitor_id": visitor_id,
        "exit_time": exit_time,
        "last_centroid": list(last_centroid),
        "colour_histogram": colour_histogram,
    })

    # Remove from active tracks
    if camera_id in _active_tracks and track_id in _active_tracks[camera_id]:
        del _active_tracks[camera_id][track_id]


def match_cross_camera(
    camera_id: str,
    new_track_centroid: Tuple[float, float],
    appearance_time: datetime,
    colour_histogram: Optional[np.ndarray] = None,
) -> Optional[str]:
    """
    Attempt cross-camera re-identification using composite scoring.

    Composite score = colour_similarity * 0.5 + temporal_score * 0.3 + spatial_score * 0.2

    Also checks active tracks on overlapping cameras to prevent
    double-counting the same physical visitor.

    Returns visitor_id if match found, None otherwise.
    """
    # Get adjacent cameras (include current camera for same-camera track merging / fragmentation healing)
    adjacent = CAMERA_ADJACENCY.get(camera_id, []) + [camera_id]

    best_match = None
    best_score = 0.0

    # --- Check active tracks on adjacent cameras (cross-camera dedup) ---
    for adj_cam in adjacent:
        if adj_cam not in _active_tracks:
            continue
        for tid, info in _active_tracks[adj_cam].items():
            time_diff = abs((appearance_time - info["timestamp"]).total_seconds())
            if time_diff > 5.0:  # Only dedup within 5 seconds
                continue

            # If checking the same camera, don't match concurrent detections in the same frame
            if adj_cam == camera_id and time_diff < 0.2:
                continue

            # Compute composite score
            colour_score = 0.0
            if colour_histogram is not None and info.get("colour_histogram") is not None:
                colour_score = max(0.0, compare_histograms(colour_histogram, info["colour_histogram"]))

            temporal_score = _compute_temporal_score(time_diff)
            spatial_score = _compute_spatial_score(
                tuple(info["centroid"]), new_track_centroid
            )

            composite = (
                colour_score * WEIGHT_COLOUR +
                temporal_score * WEIGHT_TEMPORAL +
                spatial_score * WEIGHT_SPATIAL
            )

            if composite > best_score and composite >= COMPOSITE_SCORE_THRESHOLD:
                best_score = composite
                best_match = info["visitor_id"]

    # --- Check exit registry on adjacent cameras ---
    for adj_cam in adjacent:
        if adj_cam not in exit_registry:
            continue

        for entry in exit_registry[adj_cam]:
            # Temporal check
            time_diff = (appearance_time - entry["exit_time"]).total_seconds()
            if time_diff < 0 or time_diff > TEMPORAL_WINDOW:
                continue

            # Compute composite score
            colour_score = 0.0
            if colour_histogram is not None and entry["colour_histogram"] is not None:
                colour_score = max(0.0, compare_histograms(
                    colour_histogram, entry["colour_histogram"]
                ))

            temporal_score = _compute_temporal_score(time_diff)
            spatial_score = _compute_spatial_score(
                tuple(entry["last_centroid"]), new_track_centroid
            )

            composite = (
                colour_score * WEIGHT_COLOUR +
                temporal_score * WEIGHT_TEMPORAL +
                spatial_score * WEIGHT_SPATIAL
            )

            if composite > best_score and composite >= COMPOSITE_SCORE_THRESHOLD:
                best_score = composite
                best_match = entry["visitor_id"]

    if best_match:
        logger.info(
            f"Cross-camera match: camera {camera_id}, "
            f"matched visitor {best_match} (composite score: {best_score:.3f})"
        )

    return best_match


def get_or_assign_visitor_id(
    track_id: int,
    camera_id: str = None,
    centroid: Tuple[float, float] = None,
    timestamp: datetime = None,
    colour_histogram: Optional[np.ndarray] = None,
) -> Tuple[str, bool]:
    """
    Get existing visitor_id for a track or assign a new one.
    Attempts cross-camera matching first.

    Returns: (visitor_id, is_new)
    """
    # Already mapped?
    if track_id in visitor_map:
        return visitor_map[track_id], False

    # Try cross-camera re-ID
    if camera_id and centroid and timestamp:
        matched_vid = match_cross_camera(
            camera_id, centroid, timestamp, colour_histogram
        )
        if matched_vid:
            visitor_map[track_id] = matched_vid
            return matched_vid, False

    # New visitor
    vid = generate_visitor_id()
    visitor_map[track_id] = vid
    return vid, True


def reset_reid():
    """Reset Re-ID state between processing sessions."""
    global exit_registry, visitor_map, _active_tracks
    exit_registry = {}
    visitor_map = {}
    _active_tracks = {}


def process_active_track_exits(camera_id: str, current_time: datetime):
    """
    Check active tracks on the camera. If any track has not been seen for
    more than DISAPPEAR_SECONDS, move it to the exit registry.
    """
    if camera_id not in _active_tracks:
        return
    
    expired_tids = []
    for tid, info in _active_tracks[camera_id].items():
        time_diff = (current_time - info["timestamp"]).total_seconds()
        if time_diff > DISAPPEAR_SECONDS:
            expired_tids.append((tid, info))
            
    for tid, info in expired_tids:
        register_exit(
            camera_id=camera_id,
            track_id=tid,
            visitor_id=info["visitor_id"],
            exit_time=info["timestamp"],
            last_centroid=info["centroid"],
            colour_histogram=info["colour_histogram"],
        )
