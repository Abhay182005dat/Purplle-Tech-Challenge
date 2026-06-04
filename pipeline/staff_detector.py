"""
Staff detection using multi-signal scoring system.
is_staff = True when score >= 4.

Four signals:
  1. Uniform appearance — HSV histogram match (weight: 3)
  2. Long presence — total duration > 30 min (weight: 2)
  3. Zone traversal — > 4 zones in 20 min (weight: 2)
  4. Session persistence — present before store open (weight: 3)
"""

import json
import logging
import os
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Staff colour profiles per store: store_id → mean HSV histogram
STAFF_COLOUR_PROFILES: Dict[str, np.ndarray] = {}

# Store open times
_store_open_times: Dict[str, time] = {}

# Track first-seen timestamps
_track_first_seen: Dict[int, datetime] = {}

# Track last-seen timestamps (for duration calculation)
_track_last_seen: Dict[int, datetime] = {}

# Track zone visit history: track_id → set of zone_ids
_track_zone_visits: Dict[int, Set[str]] = {}

# Track zone visit timestamps: track_id → list of (zone_id, timestamp)
_track_zone_timestamps: Dict[int, List[tuple]] = {}

# Staff detection threshold — require stronger signal to avoid false positives
STAFF_SCORE_THRESHOLD = 5

# Colour similarity threshold for uniform match — raised to reduce false matches
UNIFORM_SIMILARITY_THRESHOLD = 0.80

# Long presence threshold (seconds)
LONG_PRESENCE_THRESHOLD = 1800  # 30 minutes

# Minimum long-presence threshold (seconds) — avoids flagging browsing customers
MIN_LONG_PRESENCE_THRESHOLD = 600  # 10 minutes


def _load_store_open_times():
    """Load store open times from store_layout.json."""
    global _store_open_times
    if _store_open_times:
        return

    layout_path = os.environ.get("STORE_LAYOUT_PATH", "data/store_layout.json")
    try:
        with open(layout_path, "r") as f:
            data = json.load(f)
        for store in data.get("stores", []):
            h, m = map(int, store["open_time"].split(":"))
            _store_open_times[store["store_id"]] = time(h, m)
    except Exception as e:
        logger.error(f"Error loading store open times: {e}")


def calibrate_staff_colour(frames: list, store_id: str, detections: list = None):
    """
    Auto-calibrate staff uniform colour per store at startup.
    Analyse frames from BEFORE store_open_time — all moving people = staff.
    Extract HSV histograms from their bounding boxes.
    Store as STAFF_COLOUR_PROFILE per store_id.
    """
    _load_store_open_times()
    open_time = _store_open_times.get(store_id)

    if open_time is None:
        logger.warning(f"No open time found for {store_id}, skipping calibration")
        return

    histograms = []

    if detections:
        for det in detections:
            try:
                det_time = datetime.fromisoformat(det["timestamp"])
                if det_time.time() < open_time:
                    # This person is in store before opening = staff
                    bbox = det["bbox"]
                    frame_idx = det.get("frame_idx", 0)

                    if frame_idx < len(frames):
                        frame = frames[frame_idx]
                        x1, y1, x2, y2 = [int(b) for b in bbox]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)

                        if x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2]
                            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                            hist = cv2.calcHist(
                                [hsv], [0, 1], None,
                                [50, 60], [0, 180, 0, 256]
                            )
                            cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
                            histograms.append(hist)
            except Exception as e:
                logger.warning(f"Error processing pre-open detection: {e}")

    if histograms:
        # Compute mean histogram
        mean_hist = np.mean(histograms, axis=0)
        cv2.normalize(mean_hist, mean_hist, norm_type=cv2.NORM_L1)
        STAFF_COLOUR_PROFILES[store_id] = mean_hist
        logger.info(f"Calibrated staff colour profile for {store_id} from {len(histograms)} detections")
    else:
        logger.warning(f"No pre-open detections found for {store_id}, staff colour not calibrated")


def register_track_first_seen(track_id: int, timestamp: datetime):
    """Record the first time a track was seen."""
    if track_id not in _track_first_seen:
        _track_first_seen[track_id] = timestamp
    _track_last_seen[track_id] = timestamp


def register_track_seen(track_id: int, timestamp: datetime):
    """Update last-seen timestamp for duration calculation."""
    _track_last_seen[track_id] = timestamp


def register_zone_visit(track_id: int, zone_id: str, timestamp: datetime):
    """Record a zone visit for a track."""
    if track_id not in _track_zone_visits:
        _track_zone_visits[track_id] = set()
    _track_zone_visits[track_id].add(zone_id)

    if track_id not in _track_zone_timestamps:
        _track_zone_timestamps[track_id] = []
    _track_zone_timestamps[track_id].append((zone_id, timestamp))


def get_running_clip_duration() -> float:
    """Get the running duration of the clip processed so far."""
    if not _track_first_seen or not _track_last_seen:
        return 0.0
    first_times = [t for t in _track_first_seen.values() if t]
    last_times = [t for t in _track_last_seen.values() if t]
    if first_times and last_times:
        return (max(last_times) - min(first_times)).total_seconds()
    return 0.0


def get_track_duration(track_id: int) -> float:
    """Get total tracked duration in seconds."""
    first = _track_first_seen.get(track_id)
    last = _track_last_seen.get(track_id)
    if first and last:
        return (last - first).total_seconds()
    return 0.0


def is_staff(
    track_id: int,
    store_id: str,
    colour_histogram: Optional[np.ndarray] = None,
    timestamp: Optional[datetime] = None,
) -> tuple:
    """
    Multi-signal scoring to determine if a track is staff.
    Returns (is_staff: bool, score: int, signals: dict)

    SIGNAL 1 — HSV uniform colour match (weight: 3)
    SIGNAL 2 — Long presence (weight: 2)
    SIGNAL 3 — Zone traversal > 4 zones in 20 min (weight: 2)
    SIGNAL 4 — Present before store open (weight: 3)

    is_staff = score >= 4
    """
    _load_store_open_times()

    score = 0
    signals = {}

    # SIGNAL 1: HSV colour match to staff uniform (weight: 3)
    if colour_histogram is not None and store_id in STAFF_COLOUR_PROFILES:
        similarity = cv2.compareHist(
            colour_histogram,
            STAFF_COLOUR_PROFILES[store_id],
            cv2.HISTCMP_CORREL,
        )
        signals["uniform_similarity"] = round(similarity, 3)
        if similarity > UNIFORM_SIMILARITY_THRESHOLD:
            score += 3
            signals["uniform_match"] = True
        else:
            signals["uniform_match"] = False

    # SIGNAL 2: Long presence (weight: 2)
    duration = get_track_duration(track_id)
    running_clip_dur = get_running_clip_duration()
    # Dynamic threshold: 75% of running clip duration (min 10 minutes, max 30 minutes)
    # Higher threshold prevents browsing customers from being flagged as staff
    dynamic_threshold = min(LONG_PRESENCE_THRESHOLD, max(MIN_LONG_PRESENCE_THRESHOLD, running_clip_dur * 0.75))

    signals["presence_seconds"] = round(duration, 1)
    if duration > dynamic_threshold:
        score += 2
        signals["long_presence"] = True
    else:
        signals["long_presence"] = False

    # SIGNAL 3: Zone traversal > 4 zones in 20 min (weight: 2)
    zone_timestamps = _track_zone_timestamps.get(track_id, [])
    signals["rapid_zone_visits"] = False
    if len(zone_timestamps) > 0:
        # Check distinct zones within any 20-minute window
        sorted_visits = sorted(zone_timestamps, key=lambda x: x[1])
        for i, (_, t1) in enumerate(sorted_visits):
            window_end = t1 + timedelta(minutes=20)
            zones_in_window = set()
            for zone_id, t2 in sorted_visits[i:]:
                if t2 <= window_end:
                    zones_in_window.add(zone_id)
            if len(zones_in_window) > 4:
                score += 2
                signals["rapid_zone_visits"] = True
                signals["zones_in_20min"] = len(zones_in_window)
                break

    # SIGNAL 4: Present before store open (weight: 3)
    open_time = _store_open_times.get(store_id)
    first_seen = _track_first_seen.get(track_id)
    if open_time and first_seen:
        if first_seen.time() < open_time:
            score += 3
            signals["before_open"] = True
        else:
            signals["before_open"] = False

    signals["total_score"] = score
    return score >= STAFF_SCORE_THRESHOLD, score, signals


def reset_staff_detector():
    """Reset staff detector state."""
    global _track_first_seen, _track_last_seen, _track_zone_visits, _track_zone_timestamps
    _track_first_seen = {}
    _track_last_seen = {}
    _track_zone_visits = {}
    _track_zone_timestamps = {}
