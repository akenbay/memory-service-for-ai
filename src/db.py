"""Async engine, session factory, and one-shot init."""
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy import text

from src.config import settings
from src.models import Base


engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Create pgvector extension, tables, and FTS index. Idempotent."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
        # GIN index on the tsvector column for fast BM25-like ranking.
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_memories_content_tsv
            ON memories USING GIN (content_tsv);
        """))
        # HNSW index on embeddings for vector similarity.
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_memories_embedding
            ON memories USING hnsw (embedding vector_cosine_ops);
        """))
        # Filtering indexes for the hot paths.
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_memories_user_active
            ON memories (user_id, active) WHERE active = true;
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_turns_user_session
            ON turns (user_id, session_id);
        """))


@asynccontextmanager
async def session_scope():
    """Async context manager for a DB session with auto-commit/rollback."""
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise