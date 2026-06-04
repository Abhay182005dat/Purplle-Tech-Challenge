"""
Pydantic models for the Store Intelligence API.
Defines event schema, ingestion request/response, and validation.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, field_validator

# Valid event types — strict catalogue
VALID_EVENT_TYPES = {
    "ENTRY",
    "EXIT",
    "ZONE_ENTER",
    "ZONE_EXIT",
    "ZONE_DWELL",
    "BILLING_QUEUE_JOIN",
    "BILLING_QUEUE_ABANDON",
    "REENTRY",
}


class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    queue_position_at_join: Optional[int] = None
    wait_seconds: Optional[int] = None
    zone_hotspot_x: Optional[float] = None
    zone_hotspot_y: Optional[float] = None
    sku_zone: Optional[str] = None
    session_seq: int = 1


class Event(BaseModel):
    event_id: str
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: str
    timestamp: datetime
    ingested_at: Optional[datetime] = None
    zone_id: Optional[str] = None
    dwell_ms: int = 0
    is_staff: bool = False
    confidence: float
    is_face_hidden: bool = False
    group_id: Optional[str] = None
    group_size: Optional[int] = None
    metadata: EventMetadata = EventMetadata()

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        if v not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid event_type '{v}'. Must be one of: {', '.join(sorted(VALID_EVENT_TYPES))}"
            )
        return v


class IngestRequest(BaseModel):
    events: List[Event]

    @field_validator("events")
    @classmethod
    def validate_batch_size(cls, v: List[Event]) -> List[Event]:
        if len(v) > 500:
            raise ValueError(
                f"Batch size {len(v)} exceeds maximum of 500 events per batch"
            )
        return v


class IngestResponse(BaseModel):
    accepted: int
    rejected: int
    duplicate: int
    errors: List[dict]
