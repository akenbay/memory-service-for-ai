"""Hybrid recall: vector + BM25 fused with Reciprocal Rank Fusion."""
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from src.embeddings import embed_text


# RRF constant from the original Cormack et al. paper. Larger k flattens
# the contribution of top results; 60 is the de facto default.
RRF_K = 60

# How many candidates each ranker returns before fusion.
VECTOR_CANDIDATES = 20
BM25_CANDIDATES = 20


@dataclass
class RetrievedMemory:
    """One memory returned by hybrid retrieval, with fusion score."""
    id: str
    user_id: Optional[str]
    session_id: str
    source_turn_id: str
    type: str
    key: str
    value: str
    evidence: Optional[str]
    confidence: float
    created_at: str
    score: float  # fused RRF score


async def hybrid_recall(
    session: AsyncSession,
    query: str,
    user_id: Optional[str],
    limit: int = 10,
) -> list[RetrievedMemory]:
    """
    Run vector + BM25 retrieval, fuse with RRF, return top `limit` results.

    Filters to active memories for the given user_id. If user_id is None,
    returns nothing — we never blend memories across users.
    """
    if not user_id or not query.strip():
        return []

    # 1. Embed the query for the vector arm. If embedding fails, fall back
    #    to BM25-only — degraded but functional.
    query_embedding = await embed_text(query)

    # 2. Single SQL query that does both rankings and fuses them with RRF.
    #    The CTE pattern is: rank each candidate set separately, then
    #    full-outer-join on memory id and sum reciprocal ranks.
    if query_embedding is not None:
        sql = sql_text("""
            WITH vector_candidates AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:qvec AS vector)) AS rank
                FROM memories
                WHERE user_id = :uid
                  AND active = true
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:qvec AS vector)
                LIMIT :vec_limit
            ),
            bm25_candidates AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank(content_tsv, plainto_tsquery('english', :q)) DESC
                    ) AS rank
                FROM memories
                WHERE user_id = :uid
                  AND active = true
                  AND content_tsv @@ plainto_tsquery('english', :q)
                ORDER BY ts_rank(content_tsv, plainto_tsquery('english', :q)) DESC
                LIMIT :bm25_limit
            ),
            fused AS (
                SELECT
                    COALESCE(v.id, b.id) AS id,
                    COALESCE(1.0 / (:rrf_k + v.rank), 0) +
                    COALESCE(1.0 / (:rrf_k + b.rank), 0) AS score
                FROM vector_candidates v
                FULL OUTER JOIN bm25_candidates b USING (id)
            )
            SELECT
                m.id, m.user_id, m.session_id, m.source_turn_id,
                m.type, m.key, m.value, m.evidence, m.confidence,
                m.created_at, f.score
            FROM fused f
            JOIN memories m ON m.id = f.id
            ORDER BY f.score DESC
            LIMIT :limit
        """)
        params = {
            "uid": user_id,
            "q": query,
            "qvec": str(query_embedding),
            "vec_limit": VECTOR_CANDIDATES,
            "bm25_limit": BM25_CANDIDATES,
            "rrf_k": RRF_K,
            "limit": limit,
        }
    else:
        # BM25-only fallback.
        sql = sql_text("""
            SELECT
                m.id, m.user_id, m.session_id, m.source_turn_id,
                m.type, m.key, m.value, m.evidence, m.confidence,
                m.created_at,
                ts_rank(m.content_tsv, plainto_tsquery('english', :q)) AS score
            FROM memories m
            WHERE m.user_id = :uid
              AND m.active = true
              AND m.content_tsv @@ plainto_tsquery('english', :q)
            ORDER BY score DESC
            LIMIT :limit
        """)
        params = {"uid": user_id, "q": query, "limit": limit}

    result = await session.execute(sql, params)
    rows = result.mappings().all()

    return [
        RetrievedMemory(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            source_turn_id=row["source_turn_id"],
            type=str(row["type"]),
            key=row["key"],
            value=row["value"],
            evidence=row["evidence"],
            confidence=row["confidence"],
            created_at=row["created_at"].isoformat(),
            score=float(row["score"]),
        )
        for row in rows
    ]


async def get_stable_facts(
    session: AsyncSession,
    user_id: Optional[str],
    limit: int = 20,
) -> list[RetrievedMemory]:
    """
    Return active facts and preferences for a user, ordered by confidence.
    These go in the 'Known facts about this user' section of /recall —
    they're shown regardless of query relevance because they're persistent.
    """
    if not user_id:
        return []

    sql = sql_text("""
        SELECT
            id, user_id, session_id, source_turn_id,
            type, key, value, evidence, confidence,
            created_at
        FROM memories
        WHERE user_id = :uid
          AND active = true
          AND type::text IN ('fact', 'preference')
        ORDER BY confidence DESC, created_at DESC
        LIMIT :limit
    """)
    result = await session.execute(sql, {"uid": user_id, "limit": limit})
    rows = result.mappings().all()

    return [
        RetrievedMemory(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            source_turn_id=row["source_turn_id"],
            type=str(row["type"]),
            key=row["key"],
            value=row["value"],
            evidence=row["evidence"],
            confidence=row["confidence"],
            created_at=row["created_at"].isoformat(),
            score=row["confidence"],  # use confidence as the score
        )
        for row in rows
    ]

async def get_recent_session_turns(
    session: AsyncSession,
    session_id: str,
    limit: int = 5,
) -> list[dict]:
    """
    Return the most recent turns from this session as lightweight dicts.
    Used to provide same-session continuity in /recall's third section.
    """
    if not session_id:
        return []

    sql = sql_text("""
        SELECT id, messages, timestamp
        FROM turns
        WHERE session_id = :sid
        ORDER BY timestamp DESC
        LIMIT :limit
    """)
    result = await session.execute(sql, {"sid": session_id, "limit": limit})
    rows = result.mappings().all()

    return [
        {
            "id": row["id"],
            "messages": row["messages"],
            "timestamp": row["timestamp"].isoformat(),
        }
        for row in rows
    ]