# Store Intelligence System

Real-time retail analytics platform that converts CCTV footage into actionable business metrics. Built for the Purplle Apex Retail Challenge.

**North Star Metric:** Offline Store Conversion Rate = Unique purchasing visitors ÷ Total unique visitors

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  CCTV Clips (.mp4)                                       │
│    ↓                                                     │
│  Pipeline Layer                                          │
│    detect.py → tracker.py → reid.py → zone_mapper.py    │
│    staff_detector.py → emit.py                           │
│    ↓                                                     │
│  events.jsonl + POST /events/ingest                      │
│    ↓                                                     │
│  API Layer (FastAPI)                                     │
│    /metrics → /funnel → /heatmap → /anomalies → /health │
│    ↓                                                     │
│  Dashboard (SSE → Live Updates)                          │
└──────────────────────────────────────────────────────────┘
```

## Setup (5 commands)

```bash
git clone <repo>
cd store-intelligence
cp data/pos_transactions.csv data/
docker compose up --build
# API available at http://localhost:8000
# Dashboard at http://localhost:8000/dashboard
```

## Running the Detection Pipeline

```bash
# Process clips and feed to API
./pipeline/run.sh /path/to/clips ST1008

# Events written to: data/events.jsonl
# Events ingested to: http://localhost:8000/events/ingest
```

## API Endpoints

### POST /events/ingest
Ingest a batch of up to 500 events. Idempotent — duplicate event_ids are ignored.
```bash
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d '{"events": [{"event_id": "uuid", "store_id": "ST1008", "camera_id": "CAM_ENTRY_01", "visitor_id": "VIS_abc123", "event_type": "ENTRY", "timestamp": "2026-06-02T10:00:00", "confidence": 0.85}]}'
```

### GET /stores/{store_id}/metrics
Real-time store metrics: unique visitors, conversion rate, dwell times, queue depth.
```bash
curl http://localhost:8000/stores/ST1008/metrics
```

### GET /stores/{store_id}/funnel
Session-based conversion funnel: Entry → Zone Visit → Billing Queue → Purchase.
```bash
curl http://localhost:8000/stores/ST1008/funnel
```

### GET /stores/{store_id}/heatmap
Zone activity heatmap with normalised visit frequency and dwell times.
```bash
curl http://localhost:8000/stores/ST1008/heatmap
```

### GET /stores/{store_id}/anomalies
Active anomaly detection: queue spikes, conversion drops, dead zones, stale feeds.
```bash
curl http://localhost:8000/stores/ST1008/anomalies
```

### GET /health
System health check with per-camera feed status.
```bash
curl http://localhost:8000/health
```

## Running Tests

```bash
docker compose exec api pytest tests/ -v --cov=app
```

Or locally:
```bash
pip install -r requirements.txt
pytest tests/ -v --cov=app
```

## Dashboard

**URL:** http://localhost:8000/dashboard

Live metrics update every 5 seconds via Server-Sent Events (SSE).

Features:
- **Metric Cards** — Unique visitors, conversion rate, queue depth, abandonment rate
- **Zone Heatmap** — Colour-interpolated zone activity with dwell times
- **Conversion Funnel** — Visual funnel with drop-off percentages
- **Anomaly Alerts** — Real-time anomaly cards with severity and suggested actions
- **Store Selector** — Switch between stores with live reconnection

## Tech Stack

| Component | Technology |
|---|---|
| Detection | YOLOv8s at FP16, asymmetric frame sampling |
| Tracking | ByteTrack with 80px identity switch mitigation |
| Re-ID | 3-layer: ByteTrack → Spatial-temporal → HSV histogram |
| Staff Detection | Multi-signal scoring (uniform colour, pre-open, zones, billing) |
| Zone Mapping | Shapely polygon point-in-polygon with overlap dedup |
| API | FastAPI with structured logging middleware |
| Storage | SQLite + SQLAlchemy ORM (PostgreSQL-swappable) |
| Dashboard | Vanilla HTML/CSS/JS with SSE |
| Deployment | Docker Compose, single command |

## Design Decisions

See [DESIGN.md](docs/DESIGN.md) for full architecture documentation and [CHOICES.md](docs/CHOICES.md) for the three key engineering decisions with alternatives considered.
