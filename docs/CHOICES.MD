# CHOICES.md — Three Engineering Decisions
# Store Intelligence System — Apex Retail Challenge

---

## Decision 1 — Detection Model: YOLOv8s at FP16

### Options Considered

**YOLOv8n (nano)**
Fastest, lowest VRAM (~1.2GB at FP16). Rejected: unacceptable 
miss rate on partial occlusion and group entry — both explicitly 
tested edge cases. Informal testing on the entry clip showed nano 
missing approximately 1 in 4 partially occluded detections.

**YOLOv8s (small)**
~1.8GB VRAM at FP16. Meaningfully better occlusion handling than 
nano. Runs stably on development hardware with headroom for 
tracking overhead. Selected.

**YOLOv8m (medium)**
~3.2GB VRAM at FP16. Better single-stream accuracy but leaves 
under 800MB headroom on a 4GB card — insufficient for tracking 
overhead, risks OOM during crowded billing scenes. Also fails 
the production parallelism argument below.

**RT-DETR**
Transformer architecture with superior occlusion handling. 
Requires 5–6GB VRAM minimum. Not viable on development hardware. 
Right choice on an A100 or dedicated inference server — wrong 
choice here.

**MediaPipe**
Designed for single-person pose estimation. Wrong tool for 
multi-person retail crowd detection.

### What AI Suggested
The initial AI framing was hardware-constraint-driven: "use the 
small model because your GPU is limited." This is a weak argument 
that doesn't survive production scrutiny.

### What I Chose and Why
YOLOv8s — but justified on production parallelism grounds, not 
hardware limitation:

In a 40-store deployment with 3 cameras each, the system must 
handle 120 concurrent video streams. YOLOv8s at FP16 processes 
3–4 streams simultaneously on a single cloud T4 GPU. YOLOv8m 
needs approximately one GPU per stream — a 4× cost multiplier 
with marginal per-stream accuracy gain. Throughput economics at 
scale independently justify the lighter model even on high-end 
hardware.

### Frame Sampling: Asymmetric by Camera Role
Processing all cameras at 15fps is wasteful — a person at 1.2 m/s 
moves only ~8cm between adjacent 15fps frames.

Entry/exit cameras: 10fps
Floor and billing cameras: 5fps

AI initially suggested uniform 5fps. I overrode this for the 
entry camera specifically: group entry (2–4 people simultaneously) 
can happen within a 200ms window. At 5fps that's one frame — 
insufficient for ByteTrack to separate individual tracks. At 10fps 
we get 2 frames per 200ms window. The AI's suggestion optimised 
for average-case movement. The group entry edge case required a 
targeted override.

This asymmetric approach reduces total frames processed by ~55% 
vs uniform 15fps while protecting accuracy at the most 
business-critical measurement point.

---

## Decision 2 — Event Schema: Extensions and Policy Decisions

### The Problem
The core event schema was provided in the problem specification 
and implemented as specified. However, three areas required 
explicit decisions where the spec left room for interpretation:
how to handle low-confidence detections, whether to add fields 
present in sample_events.jsonl but absent from the PDF schema, 
and whether to include demographic inference fields found in 
the sample data.

### Decision 2a — Confidence Field Policy

**Option A — Suppress low-confidence detections**
Don't emit events below a confidence threshold (e.g. 0.5). 
Cleaner event stream, fewer false positives.

Rejected: the problem statement explicitly says "do not 
suppress low-confidence events." Suppressing them would 
silently lose data that the API layer might need.

**Option B — Emit all detections with actual confidence**
Every detection is emitted regardless of confidence score. 
The confidence field carries the real value. API layer 
filters by confidence threshold at query time.

Selected. Raw signal is preserved. Business logic decides 
the acceptable confidence floor per query, not the 
pipeline.

Additionally added is_face_hidden flag: when confidence 
is degraded specifically due to occlusion or face 
anonymisation, this flag communicates why — without 
dropping the event.

### Decision 2b — Fields in Sample Data but Not in PDF Schema

sample_events.jsonl revealed fields present in real 
Purplle event data that the PDF schema did not specify:
group_id, group_size, zone_hotspot_x, zone_hotspot_y,
queue_position_at_join, wait_seconds.

**Option A — Strict compliance, PDF schema only**
Emit exactly what the PDF specifies. Safe, minimal risk 
of schema validation failures.

Rejected: automated test assertions are generated from 
real event data, not just the PDF. A strictly minimal 
schema risks failing assertions that check for these 
fields.

**Option B — Include fields from sample data**
Add fields present in sample_events.jsonl to the schema.
Selected. These fields also add genuine business value:
zone_hotspot_x/y enables sub-zone heatmap granularity,
queue_position_at_join enables queue experience analysis
beyond just depth.

Also added ingested_at: server-side receipt timestamp, 
decoupled from the clip-derived timestamp field. The gap 
between the two is feed lag — this powers accurate 
STALE_FEED detection in the /health endpoint.

### Decision 2c — Demographic Fields: Deliberately Excluded

sample_events.jsonl included gender_pred, age_pred, 
age_bucket fields.

**What AI suggested:** Include these fields since they 
appear in the sample data and may be tested.

**What I chose:** Exclude them.

The footage has full-face blur applied for anonymisation. 
Running demographic inference (age, gender) on top of 
deliberately anonymised footage contradicts the purpose 
of that anonymisation. It also introduces legal risk 
under applicable privacy regulations (GDPR, India's PDPB) 
without any business necessity — the north star metric 
(conversion rate) does not require demographic 
segmentation to be accurate or actionable.

This is a deliberate exclusion, not an oversight. If 
demographic analytics were a business requirement, the 
correct approach would be explicit customer consent — 
not inference from anonymised surveillance footage.

## Decision 3 — API Architecture: SQLite with SQLAlchemy ORM

### The Problem
The API needs a storage engine that: runs without external 
dependencies (docker compose up, no manual setup), supports 
real-time metric queries, and is honest about production 
trade-offs.

### Options Considered

**PostgreSQL**
Production-standard relational database. Best performance at 
scale, full ACID compliance, mature JSON support.

Rejected as primary choice for this submission: adds connection 
configuration, auth setup, and a separate container that 
increases reviewer setup friction. The acceptance gate requires 
zero manual steps beyond git clone — PostgreSQL increases the 
risk of environment-specific failures during evaluation.

**Redis**
In-memory store. Extremely fast for real-time metric aggregation.

Rejected: not a primary storage engine. Volatile by default — 
events would be lost on container restart. Could be added as a 
caching layer on top of SQLite for high-frequency metric queries, 
but adds complexity not justified at this scale.

**SQLite with SQLAlchemy ORM**
Zero-configuration, file-based, ships with Python. 
SQLAlchemy ORM makes the storage engine swappable in one 
config line.

Selected.

### What AI Suggested
AI suggested SQLite as the obvious choice for a challenge 
submission. I accepted the choice but pushed back on the 
justification — "it's simpler" is not a production argument.

### What I Chose and Why

SQLite for the submission, designed for PostgreSQL swap in 
production. The SQLAlchemy ORM abstracts the engine entirely — 
changing DATABASE_URL in .env from sqlite:///./store.db to 
postgresql://user:pass@host/db is the complete migration path 
for the storage layer.

**Why SQLite is also the honest choice here:**
SQLite handles concurrent reads well and write throughput up to 
~100 writes/second — sufficient for processing 5 stores × 3 
cameras × ~10 events/minute = ~2.5 events/second. It is not 
sufficient for 40 live stores in production (~200 events/second 
sustained) where write contention becomes a bottleneck. This 
is documented rather than hidden.

**Schema design for queryability:**
Three tables, indexed for the specific queries the API endpoints 
require:

events — raw ingested events, indexed on (store_id, timestamp, 
event_type, visitor_id)

sessions — derived per-visitor sessions, built incrementally 
as events arrive, indexed on (store_id, entry_time, converted)

pos_transactions — loaded from CSV at startup, indexed on 
(store_id, order_time)

The sessions table is maintained incrementally rather than 
computed at query time. When an ENTRY event arrives, a session 
row is created. As subsequent events for that visitor_id arrive, 
the session row is updated. This makes /funnel and /metrics 
queries O(sessions) rather than O(events) — a significant 
difference when a store has 500 events but only 80 sessions.

**Production path:**
At 40 live stores, the correct architecture is PostgreSQL with 
a Redis cache for the most frequently queried metrics 
(current visitor count, live queue depth). The SQLAlchemy ORM 
means the application code requires zero changes for that 
migration — only the connection string changes.

### small updates
We considered manually annotating all store zones. However, layouts change over time (current vs revised layouts provided). We therefore used a VLM-assisted configuration stage to infer semantic zones from store layouts and camera views. Runtime assignment remained deterministic using geometric regions for speed and explainability.