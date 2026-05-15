"""SQLAlchemy 2.0 ORM models for turns and memories."""
import uuid
from datetime import datetime, timezone
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey, String, Text, Enum,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import enum


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryType(str, enum.Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    OPINION = "opinion"
    EVENT = "event"


class Turn(Base):
    __tablename__ = "turns"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    messages: Mapped[list] = mapped_column(JSON, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    turn_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    memories: Mapped[list["Memory"]] = relationship(
        back_populates="source_turn",
        cascade="all, delete-orphan",
    )


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)

    source_turn_id: Mapped[str] = mapped_column(
        ForeignKey("turns.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_turn: Mapped["Turn"] = relationship(back_populates="memories")

    type: Mapped[MemoryType] = mapped_column(Enum(MemoryType), nullable=False)
    key: Mapped[str] = mapped_column(String, nullable=False, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.8)

    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(1536),  # text-embedding-3-small dimension
        nullable=True,
    )
    content_tsv: Mapped[Optional[str]] = mapped_column(TSVECTOR, nullable=True)

    # Supersession chain — old fact stays in DB but inactive.
    supersedes: Mapped[Optional[str]] = mapped_column(
        ForeignKey("memories.id", ondelete="SET NULL"),
        nullable=True,
    )
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now,
    )