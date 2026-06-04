"""
Database layer using SQLAlchemy with SQLite.
Defines events, sessions, and pos_transactions tables with proper indexing.
"""

import os
import logging
from datetime import datetime

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    JSON,
    Index,
    text,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.mutable import MutableList

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./store_intelligence.db")

# Handle SQLite-specific connect args
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class EventRow(Base):
    """Raw ingested events table."""
    __tablename__ = "events"

    event_id = Column(String, primary_key=True)
    store_id = Column(String, index=True, nullable=False)
    camera_id = Column(String, nullable=False)
    visitor_id = Column(String, index=True, nullable=False)
    event_type = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    ingested_at = Column(DateTime, nullable=False)
    zone_id = Column(String, nullable=True)
    dwell_ms = Column(Integer, default=0)
    is_staff = Column(Boolean, index=True, default=False)
    confidence = Column(Float, nullable=False)
    is_face_hidden = Column(Boolean, default=False)
    group_id = Column(String, nullable=True)
    group_size = Column(Integer, nullable=True)
    metadata_json = Column("metadata", JSON, nullable=True)

    # Composite index for common query patterns
    __table_args__ = (
        Index("ix_events_store_ts_type", "store_id", "timestamp", "event_type"),
    )


class SessionRow(Base):
    """Derived per-visitor sessions, maintained incrementally."""
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True)
    store_id = Column(String, index=True, nullable=False)
    visitor_id = Column(String, index=True, nullable=False)
    entry_time = Column(DateTime, index=True, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    is_reentry = Column(Boolean, default=False)
    zones_visited = Column(MutableList.as_mutable(JSON), default=list)
    converted = Column(Boolean, default=False)
    is_staff = Column(Boolean, default=False)

    # Composite index for funnel and metrics queries
    __table_args__ = (
        Index("ix_sessions_store_entry_converted", "store_id", "entry_time", "converted"),
    )


class PosTransactionRow(Base):
    """POS transactions loaded from CSV at startup."""
    __tablename__ = "pos_transactions"

    transaction_id = Column(String, primary_key=True)
    store_id = Column(String, index=True, nullable=False)
    order_time = Column(DateTime, index=True, nullable=False)
    total_amount = Column(Float, nullable=False)
    order_id = Column(String, nullable=False)

    __table_args__ = (
        Index("ix_pos_store_ordertime", "store_id", "order_time"),
    )


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def get_db():
    """Dependency for FastAPI — yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
    """Direct session creation for non-dependency contexts."""
    return SessionLocal()


def load_pos_transactions(csv_path: str = "data/pos_transactions.csv"):
    """
    Load POS transactions from CSV into database.
    Handles the actual Purplle CSV schema: order_id, order_date, order_time, store_id, product_id, brand_name, total_amount
    Groups by order_id + order_time — multiple line items with same order_id and order_time = one transaction.
    Deduplicates on transaction_id (row number).
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        logger.warning(f"POS transactions file not found: {csv_path}")
        return 0

    try:
        df = pd.read_csv(csv_path)
        db = SessionLocal()
        try:
            loaded = 0

            # Detect CSV schema
            has_order_date = "order_date" in df.columns

            for _, row in df.iterrows():
                # Build transaction_id from the order_id column (row-level id)
                txn_id = str(row["order_id"])

                existing = db.query(PosTransactionRow).filter_by(
                    transaction_id=txn_id
                ).first()
                if existing:
                    continue

                # Parse order_time based on schema
                if has_order_date:
                    # Actual Purplle schema: order_date="10-04-2026", order_time="12:15:05"
                    date_str = str(row["order_date"])
                    time_str = str(row["order_time"])
                    try:
                        order_dt = datetime.strptime(f"{date_str} {time_str}", "%d-%m-%Y %H:%M:%S")
                    except ValueError:
                        try:
                            order_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            order_dt = datetime.fromisoformat(f"{date_str}T{time_str}")
                else:
                    # Fallback schema with single order_time column
                    order_dt = datetime.fromisoformat(str(row["order_time"]))

                # Use store_id from CSV (could be "ST1008" or "STORE_BLR_002")
                store_id = str(row["store_id"])

                # Total amount
                total = float(row["total_amount"])

                # order_id for grouping — use same column or a separate one
                # In the actual CSV, order_id is actually row-level, group by order_time
                oid = str(row.get("order_id", txn_id))

                txn = PosTransactionRow(
                    transaction_id=txn_id,
                    store_id=store_id,
                    order_time=order_dt,
                    total_amount=total,
                    order_id=oid,
                )
                db.add(txn)
                loaded += 1

            db.commit()
            logger.info(f"Loaded {loaded} POS transactions from {csv_path}")
            return loaded
        except Exception as e:
            db.rollback()
            logger.error(f"Error loading POS transactions: {e}")
            return 0
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error reading POS CSV: {e}")
        return 0


def get_table_counts():
    """Return row counts for all tables — used at startup logging."""
    db = SessionLocal()
    try:
        events_count = db.query(EventRow).count()
        sessions_count = db.query(SessionRow).count()
        pos_count = db.query(PosTransactionRow).count()
        return {
            "events": events_count,
            "sessions": sessions_count,
            "pos_transactions": pos_count,
        }
    except Exception as e:
        logger.error(f"Error getting table counts: {e}")
        return {"events": 0, "sessions": 0, "pos_transactions": 0}
    finally:
        db.close()
