"""
Zone mapper using polygon-based point-in-polygon detection.
Loads zone definitions from data/store_layout.json.
Loads zone categories from data/zone_categories.json (optional).
Uses shapely.geometry for Point and Polygon operations.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

# Zone definitions cache: store_id → list of zone dicts
_zones_cache: Dict[str, List[dict]] = {}

# Zone polygons cache: (store_id, zone_id) → Polygon object
_polygon_cache: Dict[Tuple[str, str], Polygon] = {}

# Zone category mapping: zone_id → category (e.g., "AQUALOGICA" → "SKINCARE")
_zone_category_map: Dict[str, str] = {}

# Overlap deduplication: visitor_id → list of recent zone events
_recent_zone_events: Dict[str, List[dict]] = {}


def load_zones(layout_path: str = None):
    """Load zone definitions from store_layout.json."""
    global _zones_cache, _polygon_cache

    if layout_path is None:
        layout_path = os.environ.get("STORE_LAYOUT_PATH", "data/store_layout.json")

    try:
        with open(layout_path, "r") as f:
            data = json.load(f)

        for store in data.get("stores", []):
            store_id = store["store_id"]
            _zones_cache[store_id] = store.get("zones", [])

            # Pre-build polygon objects
            for zone in _zones_cache[store_id]:
                key = (store_id, zone["zone_id"])
                _polygon_cache[key] = Polygon(zone["polygon"])

        logger.info(f"Loaded zones for {len(_zones_cache)} stores")

    except Exception as e:
        logger.error(f"Error loading zones: {e}")

    # Also load zone categories
    _load_zone_categories()


def _load_zone_categories(categories_path: str = None):
    """
    Load zone_categories.json for brand → category mapping.
    Falls back gracefully if file doesn't exist.
    """
    global _zone_category_map

    if categories_path is None:
        categories_path = os.environ.get("ZONE_CATEGORIES_PATH", "data/zone_categories.json")

    if not os.path.exists(categories_path):
        logger.info("No zone_categories.json found — using raw zone_id only")
        return

    try:
        with open(categories_path, "r") as f:
            data = json.load(f)

        for cat in data.get("categories", []):
            category = cat["category"]
            for zone_id in cat.get("zone_ids", []):
                _zone_category_map[zone_id] = category

        logger.info(f"Loaded zone categories: {len(_zone_category_map)} zone→category mappings")

    except Exception as e:
        logger.error(f"Error loading zone categories: {e}")


def get_zone(
    cx: float,
    cy: float,
    frame_w: int,
    frame_h: int,
    store_id: str,
    camera_id: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine which zone a centroid falls in.

    Polygon coordinates are NORMALISED (0.0 to 1.0 relative to frame dimensions).
    Returns (zone_id, zone_name) or (None, None) if not in any zone.

    zone_id is always the original brand-level ID (e.g., "AQUALOGICA")
    for backward compatibility with analytics endpoints.
    """
    if store_id not in _zones_cache:
        load_zones()

    if store_id not in _zones_cache:
        return None, None

    # Normalise centroid to 0-1 range
    point = Point(cx / frame_w, cy / frame_h)

    for zone in _zones_cache[store_id]:
        if zone["camera_id"] != camera_id:
            continue

        key = (store_id, zone["zone_id"])
        polygon = _polygon_cache.get(key)

        if polygon is None:
            polygon = Polygon(zone["polygon"])
            _polygon_cache[key] = polygon

        if polygon.contains(point):
            return zone["zone_id"], zone["zone_name"]

    return None, None


def get_zone_category(zone_id: str) -> Optional[str]:
    """
    Get the business category for a zone_id.
    Returns category string (e.g., "SKINCARE") or None if no mapping exists.
    """
    return _zone_category_map.get(zone_id)


def get_zone_info(store_id: str, zone_id: str) -> Optional[dict]:
    """Get full zone information by ID."""
    if store_id not in _zones_cache:
        load_zones()

    for zone in _zones_cache.get(store_id, []):
        if zone["zone_id"] == zone_id:
            return zone

    return None


def check_overlap_dedup(
    visitor_id: str,
    zone_id: str,
    camera_id: str,
    timestamp: datetime,
    detection_centroid: Tuple[float, float],
    store_id: str,
) -> bool:
    """
    OVERLAP DEDUPLICATION:
    If same visitor_id emits zone event from two cameras within 5 seconds,
    keep only the event where the zone polygon centroid is closer to the
    detection centroid. Discard the other.

    Returns True if this event should be KEPT, False if discarded.
    """
    if visitor_id not in _recent_zone_events:
        _recent_zone_events[visitor_id] = []

    # Clean up old entries (older than 10 seconds)
    cutoff = timestamp - timedelta(seconds=10)
    _recent_zone_events[visitor_id] = [
        e for e in _recent_zone_events[visitor_id]
        if e["timestamp"] > cutoff
    ]

    # Check for overlapping events
    for existing in _recent_zone_events[visitor_id]:
        if (
            existing["zone_id"] == zone_id
            and existing["camera_id"] != camera_id
            and abs((existing["timestamp"] - timestamp).total_seconds()) <= 5
        ):
            # Overlap detected — compare distances to zone polygon centroid
            zone_info = get_zone_info(store_id, zone_id)
            if zone_info:
                zone_polygon = Polygon(zone_info["polygon"])
                zone_centroid = zone_polygon.centroid

                existing_dist = Point(existing["centroid"]).distance(zone_centroid)
                new_dist = Point(detection_centroid).distance(zone_centroid)

                if new_dist < existing_dist:
                    # New event is closer — replace existing, keep new
                    existing["camera_id"] = camera_id
                    existing["timestamp"] = timestamp
                    existing["centroid"] = detection_centroid
                    return True
                else:
                    # Existing is closer — discard new
                    return False

    # No overlap — add to registry and keep
    _recent_zone_events[visitor_id].append({
        "zone_id": zone_id,
        "camera_id": camera_id,
        "timestamp": timestamp,
        "centroid": detection_centroid,
    })
    return True


def get_all_zones(store_id: str) -> List[dict]:
    """Get all zone definitions for a store."""
    if store_id not in _zones_cache:
        load_zones()
    return _zones_cache.get(store_id, [])


def reset_zone_mapper():
    """Reset zone mapper state."""
    global _recent_zone_events
    _recent_zone_events = {}
