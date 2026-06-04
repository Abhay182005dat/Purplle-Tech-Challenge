"""
Background pipeline runner for the dashboard.
Wraps detect → track → emit, tracks progress, stores results in-memory.
Triggered via POST /pipeline/run from the web UI.
"""

import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineRun:
    run_id: str
    status: RunStatus = RunStatus.QUEUED
    clip_path: str = ""
    store_id: str = ""
    camera_id: str = ""
    clip_start_time: str = ""
    fps: float = 15.0
    confidence_threshold: float = 0.3
    # Progress
    total_frames: int = 0
    processed_frames: int = 0
    progress_pct: float = 0.0
    # Results
    detection_count: int = 0
    track_count: int = 0
    event_count: int = 0
    staff_count: int = 0
    events: List[dict] = field(default_factory=list)
    event_type_summary: Dict[str, int] = field(default_factory=dict)
    # Timing
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_seconds: float = 0.0
    # Errors
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "clip_path": os.path.basename(self.clip_path),
            "store_id": self.store_id,
            "camera_id": self.camera_id,
            "progress_pct": round(self.progress_pct, 1),
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "detection_count": self.detection_count,
            "track_count": self.track_count,
            "event_count": self.event_count,
            "staff_count": self.staff_count,
            "event_type_summary": self.event_type_summary,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "error": self.error,
        }


# In-memory store of pipeline runs
_runs: Dict[str, PipelineRun] = {}
_lock = threading.Lock()


def get_run(run_id: str) -> Optional[PipelineRun]:
    with _lock:
        return _runs.get(run_id)


def list_runs() -> List[dict]:
    with _lock:
        return [r.to_dict() for r in sorted(_runs.values(), key=lambda r: r.started_at or "", reverse=True)]


def start_pipeline_run(
    clip_path: str,
    store_id: str,
    camera_id: str = "",
    clip_start_time: str = "",
    fps: float = 15.0,
    confidence_threshold: float = 0.3,
    api_url: str = "http://localhost:8000",
) -> str:
    """Start a pipeline run in a background thread. Returns run_id."""
    run_id = str(uuid.uuid4())[:8]

    run = PipelineRun(
        run_id=run_id,
        clip_path=clip_path,
        store_id=store_id,
        camera_id=camera_id,
        clip_start_time=clip_start_time,
        fps=fps,
        confidence_threshold=confidence_threshold,
    )

    with _lock:
        _runs[run_id] = run

    thread = threading.Thread(
        target=_execute_pipeline,
        args=(run, api_url),
        daemon=True,
    )
    thread.start()

    return run_id


def _execute_pipeline(run: PipelineRun, api_url: str):
    """Execute the full detection pipeline in a background thread."""
    import cv2

    run.status = RunStatus.RUNNING
    run.started_at = datetime.utcnow().isoformat()
    start_time = time.time()

    try:
        # Count total frames for progress tracking
        cap = cv2.VideoCapture(run.clip_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {run.clip_path}")
        run.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        logger.info(f"[{run.run_id}] Starting pipeline: {os.path.basename(run.clip_path)} ({run.total_frames} frames)")

        # Ensure pipeline module is importable
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from pipeline.detect import detect_and_track, extract_clip_start_time, _resolve_camera_id
        from pipeline.tracker import build_tracked_detections
        from pipeline.emit import process_and_emit, reset_emitter

        # Reset emitter state to avoid cross-contamination between runs
        reset_emitter()

        # Resolve camera_id if not provided
        if not run.camera_id:
            run.camera_id = _resolve_camera_id(run.clip_path, run.store_id)

        # Parse clip start time
        clip_start = extract_clip_start_time(run.clip_path, run.clip_start_time or None)

        # Step 1: Detection + Tracking
        logger.info(f"[{run.run_id}] Step 1/3: Detection + Tracking")

        detections = detect_and_track(
            clip_path=run.clip_path,
            camera_id=run.camera_id,
            clip_start_time=clip_start,
            fps=run.fps,
            confidence_threshold=run.confidence_threshold,
            model_path="yolov8s.pt",
        )

        run.detection_count = len(detections)
        unique_tracks = set(d["track_id"] for d in detections)
        run.track_count = len(unique_tracks)
        run.processed_frames = run.total_frames
        run.progress_pct = 66.0  # Detection done = 2/3

        logger.info(f"[{run.run_id}] Detected {len(detections)} instances, {len(unique_tracks)} tracks")

        # Step 2: Build tracked detections
        logger.info(f"[{run.run_id}] Step 2/3: Building tracked detections")

        tracked = build_tracked_detections(detections, run.camera_id)
        run.progress_pct = 80.0

        # Step 3: Emit events
        logger.info(f"[{run.run_id}] Step 3/3: Emitting events")

        output_path = f"data/events_{run.run_id}.jsonl"

        events = process_and_emit(
            tracked_detections=tracked,
            store_id=run.store_id,
            camera_id=run.camera_id,
            output_path=output_path,
            api_url=api_url,
        )

        run.events = events
        run.event_count = len(events)
        run.staff_count = sum(1 for e in events if e.get("is_staff"))

        # Event type summary
        from collections import Counter
        run.event_type_summary = dict(Counter(e["event_type"] for e in events))

        run.progress_pct = 100.0
        run.status = RunStatus.COMPLETED
        run.completed_at = datetime.utcnow().isoformat()
        run.elapsed_seconds = time.time() - start_time

        logger.info(
            f"[{run.run_id}] Pipeline complete: "
            f"{run.event_count} events, {run.staff_count} staff events, "
            f"{run.elapsed_seconds:.1f}s"
        )

    except Exception as e:
        run.status = RunStatus.FAILED
        run.error = str(e)
        run.completed_at = datetime.utcnow().isoformat()
        run.elapsed_seconds = time.time() - start_time
        logger.error(f"[{run.run_id}] Pipeline failed: {e}", exc_info=True)
