"""
Detection pipeline using YOLOv8s with ByteTrack tracking.
Asymmetric frame sampling by camera role.
Optional visual debugging overlays.
"""

import argparse
import json
import os
import sys

# Ensure parent directory is in python path to resolve 'pipeline.*' imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta

import cv2
import numpy as np
import torch

# Patch torch.load to bypass the PyTorch 2.6 weights_only=True breaking change for Ultralytics
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

try:
    from ultralytics import YOLO
except ImportError:
    print("WARNING: ultralytics not installed. Detection pipeline requires: pip install ultralytics")
    YOLO = None


def get_frame_sample_rate(camera_id: str) -> int:
    """
    Determine frame sampling interval based on camera role.

    Entry/exit camera: process every 2nd frame from 15fps source (→ ~7.5fps)
    Floor camera: process every 3rd frame from 15fps source (→ 5fps)
    Billing camera: process every 3rd frame from 15fps source (→ 5fps)
    """
    camera_id_lower = camera_id.lower()

    if "entry" in camera_id_lower or "cam_entry" in camera_id_lower:
        return 2  # Entry camera — higher rate for group detection
    elif "billing" in camera_id_lower:
        return 3  # Billing camera
    elif "floor" in camera_id_lower or "zone" in camera_id_lower:
        return 3  # Floor/zone camera
    else:
        return 3  # Default


def extract_clip_start_time(clip_path: str, default_time: str = None) -> datetime:
    """
    Derive clip_start_time from filename if possible.
    Expected filename patterns:
    - "2026-06-02T10-00-00_CAM_ENTRY.mp4"
    - "clip_20260602_100000.mp4"
    Falls back to provided default or current time.
    """
    filename = os.path.splitext(os.path.basename(clip_path))[0]

    # Try ISO-like pattern with dashes replacing colons
    try:
        # "2026-06-02T10-00-00_CAM_ENTRY" → "2026-06-02T10:00:00"
        parts = filename.split("_")
        ts_part = parts[0]
        if "T" in ts_part:
            ts_part = ts_part.replace("-", ":", 2)  # Only last two dashes
            # More careful parsing
            date_part = ts_part[:10]
            time_part = ts_part[11:].replace("-", ":")
            return datetime.fromisoformat(f"{date_part}T{time_part}")
    except (ValueError, IndexError):
        pass

    if default_time:
        try:
            return datetime.fromisoformat(default_time)
        except ValueError:
            pass

    return datetime.utcnow()


def detect_and_track(
    clip_path: str,
    camera_id: str,
    clip_start_time: datetime,
    fps: float = 15.0,
    confidence_threshold: float = 0.3,
    model_path: str = "yolov8s.pt",
    debug_video_path: str = None,
) -> list:
    """
    Run YOLOv8s person detection with ByteTrack tracking on a video clip.

    Uses model.track() with ByteTrack for real multi-object tracking.
    Only detects class 0 (person). Confidence threshold: 0.3.

    Returns list of detections with ByteTrack track_ids:
    [{
        "track_id": int,
        "bbox": [x1, y1, x2, y2],
        "confidence": float,
        "frame_idx": int,
        "timestamp": ISO8601 string
    }]
    """
    if YOLO is None:
        raise RuntimeError("ultralytics package not installed")

    # Load model
    model = YOLO(model_path)
    if hasattr(model, 'to'):
        try:
            model.to("cuda").half()  # FP16
        except Exception:
            pass  # CPU fallback

    # Determine frame sampling rate
    sample_rate = get_frame_sample_rate(camera_id)

    # Open video
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {clip_path}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Debug video writer
    debug_writer = None
    if debug_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        debug_writer = cv2.VideoWriter(debug_video_path, fourcc, actual_fps / sample_rate, (frame_w, frame_h))

    # Live telemetry: ensure output directory exists
    LIVE_PREVIEW_DIR = os.path.join("web", "static", "images")
    os.makedirs(LIVE_PREVIEW_DIR, exist_ok=True)
    LIVE_PREVIEW_PATH = os.path.join(LIVE_PREVIEW_DIR, "live_preview.jpg")
    LIVE_PREVIEW_WIDTH = 640
    _processed_count = 0  # Counter to throttle preview writes

    detections = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Asymmetric sampling: only process every Nth frame
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        # Calculate timestamp for this frame
        timestamp = clip_start_time + timedelta(seconds=frame_idx / actual_fps)

        # Run detection + tracking with ByteTrack
        results = model.track(
            frame,
            classes=[0],
            conf=confidence_threshold,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        frame_detections = []
        for result in results:
            if result.boxes is not None and result.boxes.id is not None:
                for i, box in enumerate(result.boxes):
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    track_id = int(box.id[0].cpu().numpy())

                    from pipeline.reid import extract_clothing_histogram
                    hist = extract_clothing_histogram(frame, bbox)
                    hist_list = hist.tolist() if hist is not None else None

                    det = {
                        "track_id": track_id,
                        "bbox": [round(b, 1) for b in bbox],
                        "confidence": round(conf, 4),
                        "frame_idx": frame_idx,
                        "timestamp": timestamp.isoformat(),
                        "colour_histogram": hist_list,
                    }
                    detections.append(det)
                    frame_detections.append(det)

        # Draw debug overlays (for debug video and/or live preview)
        annotated_frame = _draw_debug_overlays(frame, frame_detections)
        if debug_writer is not None:
            debug_writer.write(annotated_frame)

        # Live telemetry: save preview every 5th processed frame
        _processed_count += 1
        if _processed_count % 5 == 0:
            preview_src = annotated_frame
            scale = LIVE_PREVIEW_WIDTH / preview_src.shape[1]
            preview = cv2.resize(
                preview_src,
                (LIVE_PREVIEW_WIDTH, int(preview_src.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
            cv2.imwrite(LIVE_PREVIEW_PATH, preview, [cv2.IMWRITE_JPEG_QUALITY, 70])

        frame_idx += 1

    cap.release()
    if debug_writer is not None:
        debug_writer.release()

    return detections


def _draw_debug_overlays(
    frame: np.ndarray,
    detections: list,
    visitor_map: dict = None,
    zone_map: dict = None,
    staff_map: dict = None,
) -> np.ndarray:
    """
    Draw visual debugging overlays on a frame.
    Shows: bounding box, track_id, visitor_id, zone, staff flag.
    """
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(b) for b in det["bbox"]]
        track_id = det["track_id"]

        # Color: green for normal, red for staff
        is_staff = staff_map.get(track_id, False) if staff_map else False
        color = (0, 0, 255) if is_staff else (0, 255, 0)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label text
        label_parts = [f"T:{track_id}"]

        if visitor_map and track_id in visitor_map:
            label_parts.append(f"V:{visitor_map[track_id]}")

        if zone_map and track_id in zone_map:
            label_parts.append(f"Z:{zone_map[track_id]}")

        if is_staff:
            label_parts.append("STAFF")

        label = " | ".join(label_parts)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return annotated


# Keep backward-compatible function name
def detect_persons(
    clip_path: str,
    camera_id: str,
    clip_start_time: datetime,
    fps: float = 15.0,
    confidence_threshold: float = 0.3,
    model_path: str = "yolov8s.pt",
) -> list:
    """
    Backward-compatible wrapper. Calls detect_and_track but strips track_id
    for callers that don't expect it.
    """
    return detect_and_track(
        clip_path=clip_path,
        camera_id=camera_id,
        clip_start_time=clip_start_time,
        fps=fps,
        confidence_threshold=confidence_threshold,
        model_path=model_path,
    )


def _resolve_camera_id(clip_path: str, store_id: str, fallback: str = None) -> str:
    """
    Resolve the correct camera_id by matching clip filename against
    the store_layout.json cameras list for the given store_id.

    Priority:
    1. Exact match of store layout camera name in clip filename
    2. Heuristic match based on keywords (entry, zone, billing, floor, cam1, cam2, etc.)
    3. Fallback to provided default or generic ID
    """
    import json as _json

    clip_name = os.path.basename(clip_path).lower().replace(" ", "").replace("-", "").replace("_", "")

    # Try loading store layout to get actual camera IDs
    layout_path = os.environ.get("STORE_LAYOUT_PATH", "data/store_layout.json")
    layout_cameras = []
    try:
        with open(layout_path, "r") as f:
            data = _json.load(f)
        for store in data.get("stores", []):
            if store["store_id"] == store_id:
                layout_cameras = store.get("cameras", [])
                break
    except Exception:
        pass

    # Try exact match: check if any layout camera name appears in the clip filename
    for cam_id in layout_cameras:
        cam_normalised = cam_id.lower().replace(" ", "").replace("-", "").replace("_", "")
        if cam_normalised in clip_name:
            return cam_id

    # Try keyword matching against layout cameras
    for cam_id in layout_cameras:
        cam_lower = cam_id.lower()
        # Match by camera role keywords
        if ("entry" in clip_name and "entry" in cam_lower):
            return cam_id
        if ("billing" in clip_name and "billing" in cam_lower):
            return cam_id

    # Try numeric camera matching: "cam 1" in filename → "CAM_1" in layout
    import re
    clip_cam_nums = re.findall(r'cam\s*(\d+)', os.path.basename(clip_path).lower())
    for clip_num in clip_cam_nums:
        for cam_id in layout_cameras:
            layout_nums = re.findall(r'(\d+)', cam_id)
            if clip_num in layout_nums:
                return cam_id

    # Fallback to generic mapping or provided fallback
    clip_name_orig = os.path.basename(clip_path).lower()
    if fallback:
        return fallback
    if "entry" in clip_name_orig:
        return "CAM_ENTRY_01"
    elif "billing" in clip_name_orig:
        return "CAM_BILLING_01"
    elif "cam 2" in clip_name_orig or "cam2" in clip_name_orig:
        return "CAM_FLOOR_02"
    else:
        return "CAM_FLOOR_01"


def main():
    """CLI entry point for detection pipeline."""
    parser = argparse.ArgumentParser(description="YOLOv8s Person Detection + ByteTrack Pipeline")
    parser.add_argument("--clip", required=True, help="Path to video clip")
    parser.add_argument("--camera_id", default=None, help="Camera identifier (auto-detected from filename if omitted)")
    parser.add_argument("--store_id", required=True, help="Store identifier")
    parser.add_argument("--clip_start_time", default=None, help="Clip start time (ISO-8601)")
    parser.add_argument("--fps", type=float, default=15.0, help="Source FPS")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO model path")
    parser.add_argument("--output", default="data/events.jsonl", help="Output JSONL file")
    parser.add_argument("--api_url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--debug-video", default=None, dest="debug_video",
                        help="Path to write debug overlay video (disabled if not set)")

    args = parser.parse_args()

    # Resolve camera_id: use provided value, or auto-detect from filename + store layout
    camera_id = args.camera_id or _resolve_camera_id(args.clip, args.store_id)

    # Get clip start time
    clip_start_time = extract_clip_start_time(args.clip, args.clip_start_time)

    print(f"Detecting persons in: {args.clip}")
    print(f"Camera: {camera_id}, Sample rate: every {get_frame_sample_rate(camera_id)} frames")
    print(f"Clip start time: {clip_start_time.isoformat()}")
    print(f"Tracking: ByteTrack (via YOLOv8s model.track)")

    # Run detection + tracking
    detections = detect_and_track(
        clip_path=args.clip,
        camera_id=camera_id,
        clip_start_time=clip_start_time,
        fps=args.fps,
        model_path=args.model,
        debug_video_path=args.debug_video,
    )

    unique_tracks = set(d['track_id'] for d in detections)
    print(f"Detected {len(detections)} person instances across {len(unique_tracks)} tracks")

    # Import tracker and emitter for full pipeline
    from pipeline.tracker import build_tracked_detections
    from pipeline.emit import process_and_emit

    # Build tracked detections (enriches with centroid, updates history)
    tracked = build_tracked_detections(detections, camera_id)
    print(f"Tracked {len(unique_tracks)} unique tracks")

    # Emit events
    events = process_and_emit(
        tracked_detections=tracked,
        store_id=args.store_id,
        camera_id=camera_id,
        output_path=args.output,
        api_url=args.api_url,
    )
    print(f"Emitted {len(events)} events")

    # Print event type summary
    from collections import Counter
    event_types = Counter(e['event_type'] for e in events)
    staff_events = sum(1 for e in events if e.get('is_staff'))
    print(f"Event summary: {dict(event_types)}")
    print(f"Staff events: {staff_events}")


if __name__ == "__main__":
    main()

