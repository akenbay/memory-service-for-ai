"""Memory service HTTP API."""
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, delete

from src.db import init_db, session_scope
from src.models import Turn, Memory, MemoryType


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Memory Service", lifespan=lifespan)


# ---------- Request / response schemas ----------

class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class TurnRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    messages: list[Message]
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnResponse(BaseModel):
    id: str


class RecallRequest(BaseModel):
    query: str
    session_id: str
    user_id: Optional[str] = None
    max_tokens: int = 1024


class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallResponse(BaseModel):
    context: str
    citations: list[Citation]


class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    limit: int = 10


class SearchResult(BaseModel):
    content: str
    score: float
    session_id: str
    timestamp: datetime
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    results: list[SearchResult]


class MemoryRecord(BaseModel):
    id: str
    type: str
    key: str
    value: str
    confidence: float
    source_session: str
    source_turn: str
    created_at: datetime
    updated_at: datetime
    supersedes: Optional[str]
    active: bool


class MemoriesResponse(BaseModel):
    memories: list[MemoryRecord]


# ---------- Endpoints ----------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/turns", response_model=TurnResponse, status_code=status.HTTP_201_CREATED)
async def write_turn(req: TurnRequest):
    async with session_scope() as session:
        turn = Turn(
            session_id=req.session_id,
            user_id=req.user_id,
            messages=[m.model_dump() for m in req.messages],
            timestamp=req.timestamp,
            turn_metadata=req.metadata,
        )
        session.add(turn)
        await session.flush()  # populate turn.id
        turn_id = turn.id

        # Extraction comes in Phase 2 — for now, persist only.

    return TurnResponse(id=turn_id)


@app.post("/recall", response_model=RecallResponse)
async def recall(req: RecallRequest):
    # Real recall comes in Phase 4. For now, empty.
    return RecallResponse(context="", citations=[])


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    # Real search comes in Phase 4.
    return SearchResponse(results=[])


@app.get("/users/{user_id}/memories", response_model=MemoriesResponse)
async def get_memories(user_id: str):
    async with session_scope() as session:
        result = await session.execute(
            select(Memory).where(Memory.user_id == user_id).order_by(Memory.created_at.desc())
        )
        memories = result.scalars().all()
        return MemoriesResponse(
            memories=[
                MemoryRecord(
                    id=m.id,
                    type=m.type.value,
                    key=m.key,
                    value=m.value,
                    confidence=m.confidence,
                    source_session=m.session_id,
                    source_turn=m.source_turn_id,
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                    supersedes=m.supersedes,
                    active=m.active,
                )
                for m in memories
            ]
        )


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    async with session_scope() as session:
        # Cascade through turns will clean up memories.
        await session.execute(delete(Turn).where(Turn.session_id == session_id))
    return None


@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str):
    async with session_scope() as session:
        await session.execute(delete(Turn).where(Turn.user_id == user_id))
        # Also delete user-scoped memories that may have outlived their turns
        await session.execute(delete(Memory).where(Memory.user_id == user_id))
    return None