"""
Zone Configuration & Annotation Tool.

1. Interactive OpenCV GUI to draw polygons on store layout images.
2. Normalizes coordinates and saves to data/store_layout.json.
3. VLM adapter to classify zone brands into business categories.

Usage:
    python -m pipeline.setup_zones --store_id STORE_BLR_002 --camera_id CAM_FLOOR_01 --image layout_floor.png
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LAYOUT_PATH = "data/store_layout.json"
DEFAULT_CATEGORIES_PATH = "data/zone_categories.json"

# --- 1. INTERACTIVE POLYGON ANNOTATOR ---

class ZoneAnnotator:
    def __init__(self, image_path: str, store_id: str, camera_id: str):
        self.image_path = image_path
        self.store_id = store_id
        self.camera_id = camera_id
        
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
            
        self.clone = self.img.copy()
        self.h, self.w = self.img.shape[:2]
        
        self.current_polygon = []
        self.zones = []
        self.window_name = f"Annotating {camera_id} - Press 'h' for help"
        
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_polygon:
                self.current_polygon.pop()

    def run(self) -> List[dict]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n--- ANNOTATION CONTROLS ---")
        print("Left Click  : Add polygon point")
        print("Right Click : Remove last point")
        print("'s' / Enter : Save current polygon & enter name")
        print("'c'         : Clear current polygon")
        print("'q' / Esc   : Finish and save to JSON\n")

        while True:
            display_img = self.clone.copy()
            
            # Draw existing zones
            for z in self.zones:
                pts = np.array([[int(px * self.w), int(py * self.h)] for px, py in z["polygon"]], np.int32)
                cv2.polylines(display_img, [pts], True, (0, 255, 0), 2)
                cv2.putText(display_img, z["zone_id"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw current polygon
            if self.current_polygon:
                pts = np.array(self.current_polygon, np.int32)
                cv2.polylines(display_img, [pts], False, (0, 0, 255), 2)
                for pt in self.current_polygon:
                    cv2.circle(display_img, pt, 4, (0, 0, 255), -1)

            cv2.imshow(self.window_name, display_img)
            key = cv2.waitKey(1) & 0xFF

            if key in [27, ord('q')]: # Esc or q
                break
            elif key == ord('c'):
                self.current_polygon = []
            elif key in [13, ord('s')]: # Enter or s
                if len(self.current_polygon) >= 3:
                    print("\n--- NEW ZONE ---")
                    zone_id = input("Enter Brand/Zone ID (e.g., AQUALOGICA, BILLING): ").strip().upper()
                    zone_name = input("Enter readable name (e.g., Aqualogica Display): ").strip()
                    
                    # Normalize coordinates (0.0 to 1.0)
                    normalized_poly = [[round(x/self.w, 4), round(y/self.h, 4)] for x, y in self.current_polygon]
                    
                    self.zones.append({
                        "zone_id": zone_id,
                        "zone_name": zone_name if zone_name else zone_id,
                        "camera_id": self.camera_id,
                        "polygon": normalized_poly
                    })
                    print(f"Saved zone: {zone_id}")
                    self.current_polygon = []
                else:
                    print("Need at least 3 points for a polygon!")

        cv2.destroyAllWindows()
        return self.zones

# --- 2. LAYOUT JSON MANAGER ---

def update_store_layout(store_id: str, new_zones: List[dict], layout_path: str):
    """Merges new zones into the existing store_layout.json"""
    data = {"stores": []}
    if os.path.exists(layout_path):
        with open(layout_path, "r") as f:
            data = json.load(f)
            
    store_idx = next((i for i, s in enumerate(data["stores"]) if s["store_id"] == store_id), None)
    
    if store_idx is None:
        data["stores"].append({
            "store_id": store_id,
            "open_time": "09:00",
            "close_time": "22:00",
            "cameras": list(set([z["camera_id"] for z in new_zones])),
            "zones": new_zones
        })
    else:
        # Append new zones, update camera list
        data["stores"][store_idx]["zones"].extend(new_zones)
        all_cams = set(data["stores"][store_idx].get("cameras", []))
        all_cams.update([z["camera_id"] for z in new_zones])
        data["stores"][store_idx]["cameras"] = list(all_cams)

    os.makedirs(os.path.dirname(layout_path) or ".", exist_ok=True)
    with open(layout_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved layout data to {layout_path}")

# --- 3. VLM CLASSIFICATION (Original Logic) ---

def load_zone_categories(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def build_zone_to_category_map(categories_data: dict) -> Dict[str, str]:
    mapping = {}
    for cat in categories_data.get("categories", []):
        category = cat["category"]
        for zone_id in cat.get("zone_ids", []):
            mapping[zone_id] = category
    return mapping

def classify_zones_with_vlm(layout_image_path: str, vlm_provider: str) -> dict:
    raise NotImplementedError(
        f"VLM provider '{vlm_provider}' not configured. "
        f"Implement API call here, or construct data/zone_categories.json manually."
    )

def ensure_zone_categories(store_id: str, layout_image_path: str = None, vlm_provider: str = None) -> Dict[str, str]:
    data = load_zone_categories(DEFAULT_CATEGORIES_PATH)
    if data:
        return build_zone_to_category_map(data)

    if vlm_provider and layout_image_path:
        logger.info(f"Attempting VLM classification with {vlm_provider}...")
        try:
            data = classify_zones_with_vlm(layout_image_path, vlm_provider)
            data["store_id"] = store_id
            with open(DEFAULT_CATEGORIES_PATH, "w") as f:
                json.dump(data, f, indent=2)
            return build_zone_to_category_map(data)
        except Exception as e:
            logger.error(f"VLM classification failed: {e}")

    return {}

def main():
    parser = argparse.ArgumentParser(description="Store Layout Annotation & Classification Tool")
    parser.add_argument("--store_id", required=True, help="Store identifier (e.g., STORE_BLR_002)")
    parser.add_argument("--camera_id", help="Camera identifier (e.g., CAM_FLOOR_01)")
    parser.add_argument("--image", help="Path to layout image (PNG/JPG)")
    parser.add_argument("--vlm", default=None, choices=["gemini", "openai", "claude"], help="VLM for category classification")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Run Interactive Annotator if image is provided
    if args.image and args.camera_id:
        logger.info(f"Starting annotator for {args.camera_id}...")
        annotator = ZoneAnnotator(args.image, args.store_id, args.camera_id)
        new_zones = annotator.run()
        
        if new_zones:
            update_store_layout(args.store_id, new_zones, DEFAULT_LAYOUT_PATH)
            logger.info(f"Added {len(new_zones)} zones to layout.")
        else:
            logger.info("No zones annotated.")
            
    elif args.image and not args.camera_id:
        logger.error("Must provide --camera_id when annotating an --image")

    # 2. Run VLM Categorization
    mapping = ensure_zone_categories(args.store_id, args.image, args.vlm)
    if mapping:
        print(f"\nZone category mapping ({len(mapping)} zones):")
        for zone_id, category in sorted(mapping.items()):
            print(f"  {zone_id:20s} -> {category}")

if __name__ == "__main__":
    main()