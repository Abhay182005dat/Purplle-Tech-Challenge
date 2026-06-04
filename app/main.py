"""
FastAPI main application.
Routes, structured logging middleware, SSE endpoint, graceful degradation.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.db import get_db, init_db, load_pos_transactions, get_table_counts, SessionLocal
from app.models import IngestRequest, IngestResponse
from app.ingestion import ingest_events
from app.metrics import get_metrics as compute_metrics
from app.funnel import get_funnel as compute_funnel
from app.heatmap import get_heatmap as compute_heatmap
from app.anomalies import get_anomalies as compute_anomalies
from app.health import get_health as compute_health

# Configure structured logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler — runs on startup and shutdown.
    1. Create all database tables
    2. Load pos_transactions.csv
    3. Log table row counts
    """
    logger.info("Starting Store Intelligence API...")

    # 1. Create tables
    init_db()

    # 2. Load POS transactions
    csv_path = os.environ.get("POS_CSV_PATH", "data/pos_transactions.csv")
    loaded = load_pos_transactions(csv_path)
    logger.info(f"POS transactions loaded: {loaded}")

    # 3. Log table counts
    counts = get_table_counts()
    logger.info(f"API ready — table counts: {json.dumps(counts)}")

    yield  # Application runs here

    logger.info("Shutting down Store Intelligence API...")


app = FastAPI(
    title="Store Intelligence API",
    description="Retail analytics API — converts CCTV footage events into actionable business metrics",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Structured Logging Middleware ─────────────────────────────────────────────

@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    """Log structured JSON for every request with trace_id, latency, etc."""
    trace_id = str(uuid.uuid4())
    start_time = time.time()

    # Extract store_id from path params if present
    store_id = None
    path_parts = request.url.path.strip("/").split("/")
    if len(path_parts) >= 2 and path_parts[0] == "stores":
        store_id = path_parts[1]

    try:
        response = await call_next(request)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        log_entry = {
            "trace_id": trace_id,
            "store_id": store_id,
            "endpoint": request.url.path,
            "method": request.method,
            "latency_ms": latency_ms,
            "status_code": response.status_code,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(json.dumps(log_entry))
        return response

    except Exception as e:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        log_entry = {
            "trace_id": trace_id,
            "store_id": store_id,
            "endpoint": request.url.path,
            "method": request.method,
            "latency_ms": latency_ms,
            "status_code": 500,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.error(json.dumps(log_entry))
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "status": 500,
            },
        )




# ─── Routes ───────────────────────────────────────────────────────────────────

@app.post("/events/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest, db: Session = Depends(get_db)):
    """
    Ingest a batch of events (max 500).
    Partial success returns HTTP 200.
    Idempotent: same payload twice = duplicates, no new rows.
    """
    event_count = len(request.events)
    try:
        response = ingest_events(request, db)
        # Log event_count for structured logging
        logger.info(json.dumps({
            "action": "ingest",
            "event_count": event_count,
            "accepted": response.accepted,
            "rejected": response.rejected,
            "duplicate": response.duplicate,
        }))
        return response
    except ValidationError as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": "validation_error",
                "message": str(e),
                "status": 400,
            },
        )
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "database_unavailable",
                "message": "Database operation failed",
                "status": 503,
            },
        )


@app.get("/stores/{store_id}/metrics")
async def metrics_endpoint(store_id: str, db: Session = Depends(get_db)):
    """Real-time store metrics. Never cached."""
    try:
        return compute_metrics(store_id, db)
    except Exception as e:
        logger.error(f"Metrics error for {store_id}: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "database_unavailable",
                "message": "Database operation failed",
                "status": 503,
            },
        )


@app.get("/stores/{store_id}/funnel")
async def funnel_endpoint(store_id: str, db: Session = Depends(get_db)):
    """Session-based conversion funnel."""
    try:
        return compute_funnel(store_id, db)
    except Exception as e:
        logger.error(f"Funnel error for {store_id}: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "database_unavailable",
                "message": "Database operation failed",
                "status": 503,
            },
        )


@app.get("/stores/{store_id}/heatmap")
async def heatmap_endpoint(store_id: str, db: Session = Depends(get_db)):
    """Zone activity heatmap."""
    try:
        return compute_heatmap(store_id, db)
    except Exception as e:
        logger.error(f"Heatmap error for {store_id}: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "database_unavailable",
                "message": "Database operation failed",
                "status": 503,
            },
        )


@app.get("/stores/{store_id}/anomalies")
async def anomalies_endpoint(store_id: str, db: Session = Depends(get_db)):
    """Active anomaly detection."""
    try:
        return compute_anomalies(store_id, db)
    except Exception as e:
        logger.error(f"Anomalies error for {store_id}: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "database_unavailable",
                "message": "Database operation failed",
                "status": 503,
            },
        )


@app.get("/health")
async def health_endpoint(db: Session = Depends(get_db)):
    """
    System health check. Works on empty state.
    Returns HTTP 503 on database failure.
    """
    try:
        result = compute_health(db)
        if result.get("database") == "disconnected":
            return JSONResponse(
                status_code=503,
                content={
                    "error": "database_unavailable",
                    "message": "Database connection failed",
                    "status": 503,
                },
            )
        return result
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "database_unavailable",
                "message": "Database connection failed",
                "status": 503,
            },
        )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML."""
    dashboard_path = Path(__file__).parent / "dashboard" / "index.html"
    if not dashboard_path.exists():
        return HTMLResponse(
            content="<h1>Dashboard not found</h1>",
            status_code=404,
        )
    return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))


# ─── Pipeline Endpoints ────────────────────────────────────────────────────────

@app.post("/pipeline/run")
async def pipeline_run(
    video: UploadFile = File(...),
    store_id: str = Form("ST1008"),
    camera_id: str = Form(""),
    clip_start_time: str = Form(""),
    fps: float = Form(15.0),
    confidence_threshold: float = Form(0.3),
):
    """Upload a video and start the detection pipeline."""
    from app.pipeline_runner import start_pipeline_run

    # Save uploaded file
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = video.filename.replace(" ", "_") if video.filename else "upload.mp4"
    file_path = upload_dir / safe_name

    with open(file_path, "wb") as f:
        content = await video.read()
        f.write(content)

    logger.info(f"Uploaded video: {file_path} ({len(content)} bytes)")

    # Start pipeline in background
    run_id = start_pipeline_run(
        clip_path=str(file_path),
        store_id=store_id,
        camera_id=camera_id,
        clip_start_time=clip_start_time,
        fps=fps,
        confidence_threshold=confidence_threshold,
        api_url="http://localhost:8000",
    )

    return {"run_id": run_id, "status": "queued", "filename": safe_name}


@app.get("/pipeline/status/{run_id}")
async def pipeline_status(run_id: str):
    """Get the status of a pipeline run."""
    from app.pipeline_runner import get_run

    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run.to_dict()


@app.get("/pipeline/results/{run_id}")
async def pipeline_results(run_id: str):
    """Get the events produced by a pipeline run."""
    from app.pipeline_runner import get_run

    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if run.status.value not in ("completed", "failed"):
        return {"status": run.status.value, "message": "Pipeline still running"}

    return {
        "run_id": run.run_id,
        "status": run.status.value,
        "event_count": run.event_count,
        "staff_count": run.staff_count,
        "event_type_summary": run.event_type_summary,
        "events": run.events[:50],  # First 50 for preview
        "total_events": len(run.events),
    }


@app.get("/pipeline/runs")
async def pipeline_runs():
    """List all pipeline runs."""
    from app.pipeline_runner import list_runs
    return {"runs": list_runs()}


@app.get("/stores/list")
async def list_stores():
    """Return stores from store_layout.json for the UI dropdown."""
    layout_path = os.environ.get("STORE_LAYOUT_PATH", "data/store_layout.json")
    try:
        with open(layout_path, "r") as f:
            data = json.load(f)
        stores = []
        for store in data.get("stores", []):
            stores.append({
                "store_id": store["store_id"],
                "cameras": store.get("cameras", []),
            })
        return {"stores": stores}
    except Exception as e:
        return {"stores": [], "error": str(e)}


@app.get("/static/images/live_preview.jpg")
async def live_preview():
    """Serve the live preview image from pipeline telemetry."""
    preview_path = Path("web/static/images/live_preview.jpg")
    if not preview_path.exists():
        # Fallback to a tiny 1x1 transparent GIF or 404
        raise HTTPException(
            status_code=404,
            detail="Live preview image not generated yet. Start the video pipeline to view."
        )
    return FileResponse(preview_path)



@app.get("/stores/{store_id}/stream")
async def sse_endpoint(store_id: str):
    """
    Server-Sent Events endpoint for live dashboard updates.
    Handles client disconnect gracefully.
    """
    async def event_stream():
        while True:
            try:
                db = SessionLocal()
                try:
                    metrics = compute_metrics(store_id, db)
                    anomalies = compute_anomalies(store_id, db)
                    data = {
                        "metrics": metrics,
                        "anomalies": anomalies,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                finally:
                    db.close()

                await asyncio.sleep(5)
            except asyncio.CancelledError:
                # Client disconnected — exit gracefully
                logger.info(f"SSE client disconnected for store {store_id}")
                break
            except Exception as e:
                logger.error(f"SSE error for {store_id}: {e}")
                error_data = {
                    "error": "stream_error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                await asyncio.sleep(5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
