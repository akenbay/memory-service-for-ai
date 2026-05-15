"""Memory service HTTP API."""
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, delete

from src.db import init_db, session_scope
from src.models import Turn, Memory, MemoryType

from src.extraction import extract_memories
from src.embeddings import embed_batch
from sqlalchemy import text as sql_text

from src.recall import hybrid_recall, get_stable_facts
from src.context_assembly import assemble_context, Citation as AsmCitation

from src.recall import hybrid_recall, get_stable_facts, get_recent_session_turns

from sqlalchemy import update
from src.contradiction import classify, Relationship

from datetime import datetime, timezone

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

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
        # 1. Persist the raw turn.
        turn = Turn(
            session_id=req.session_id,
            user_id=req.user_id,
            messages=[m.model_dump() for m in req.messages],
            timestamp=req.timestamp,
            turn_metadata=req.metadata,
        )
        session.add(turn)
        await session.flush()
        turn_id = turn.id

        # 2. Extract structured memories.
        extracted = await extract_memories(turn.messages)

        if extracted:
            # 3. For each extracted memory, decide supersession.
            #    We do this BEFORE writing so we can set supersedes
            #    pointers in the same transaction.
            to_insert: list[tuple[Memory, str | None]] = []  # (new_memory, supersedes_id)
            ids_to_deactivate: list[str] = []

            for em in extracted:
                # Look up existing active memories with the same (user_id, key).
                existing_rows = (await session.execute(
                    sql_text("""
                        SELECT id, value FROM memories
                        WHERE user_id = :uid
                          AND key = :key
                          AND active = true
                        ORDER BY created_at DESC
                    """),
                    {"uid": req.user_id, "key": em.key},
                )).mappings().all()

                supersedes_id: str | None = None
                skip_insert = False

                for row in existing_rows:
                    rel = await classify(em.key, row["value"], em.value)
                    if rel == Relationship.IDENTICAL:
                        # Don't insert a duplicate; just bump the existing.
                        await session.execute(
                            update(Memory)
                            .where(Memory.id == row["id"])
                            .values(updated_at=_now_utc())
                        )
                        skip_insert = True
                        break
                    elif rel == Relationship.CONTRADICTORY:
                        # The new memory supersedes this one.
                        # Mark old for deactivation; point the new one at it.
                        supersedes_id = row["id"]
                        ids_to_deactivate.append(row["id"])
                        # Note: we only chain to the most-recent contradicting
                        # memory. Older ones are already inactive (they were
                        # superseded earlier).
                        break
                    # ADDITIVE: keep looking through older rows or just insert.

                if skip_insert:
                    continue

                new_mem = Memory(
                    user_id=req.user_id,
                    session_id=req.session_id,
                    source_turn_id=turn_id,
                    type=em.type,
                    key=em.key,
                    value=em.value,
                    evidence=em.evidence,
                    confidence=em.confidence,
                    supersedes=supersedes_id,
                )
                to_insert.append((new_mem, supersedes_id))

            # 4. Deactivate superseded memories.
            if ids_to_deactivate:
                await session.execute(
                    update(Memory)
                    .where(Memory.id.in_(ids_to_deactivate))
                    .values(active=False, updated_at=_now_utc())
                )

            # 5. Batch-embed and insert the new memories.
            if to_insert:
                values = [mem.value for mem, _ in to_insert]
                embeddings = await embed_batch(values)
                for (mem, _), vec in zip(to_insert, embeddings):
                    mem.embedding = vec
                    session.add(mem)
                await session.flush()

                # 6. Populate the tsvector column for the new rows.
                await session.execute(
                    sql_text(
                        "UPDATE memories SET content_tsv = "
                        "to_tsvector('english', coalesce(key, '') || ' ' || value) "
                        "WHERE source_turn_id = :tid AND content_tsv IS NULL"
                    ),
                    {"tid": turn_id},
                )

    return TurnResponse(id=turn_id)

@app.post("/recall", response_model=RecallResponse)
async def recall(req: RecallRequest):
    async with session_scope() as session:
        stable = await get_stable_facts(session, req.user_id, limit=20)
        recalled = await hybrid_recall(
            session, req.query, req.user_id, limit=10,
        )
        recent_turns = await get_recent_session_turns(
            session, req.session_id, limit=5,
        )

        assembled = assemble_context(
            stable, recalled, recent_turns, req.max_tokens,
        )

    return RecallResponse(
        context=assembled.context,
        citations=[
            Citation(turn_id=c.turn_id, score=c.score, snippet=c.snippet)
            for c in assembled.citations
        ],
    )


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    async with session_scope() as session:
        if not req.user_id:
            return SearchResponse(results=[])

        results = await hybrid_recall(
            session, req.query, req.user_id, limit=req.limit,
        )

    return SearchResponse(
        results=[
            SearchResult(
                content=m.value,
                score=m.score,
                session_id=m.session_id,
                timestamp=m.created_at,
                metadata={
                    "key": m.key,
                    "type": m.type,
                    "confidence": m.confidence,
                    "source_turn_id": m.source_turn_id,
                },
            )
            for m in results
        ]
    )


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