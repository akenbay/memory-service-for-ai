"""OpenAI embeddings client."""
from openai import AsyncOpenAI

from src.config import settings


_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def embed_text(text: str) -> list[float] | None:
    """Embed a single string. Returns None on failure — caller decides what to do."""
    if not settings.openai_api_key:
        return None
    if not text or not text.strip():
        return None
    try:
        client = get_client()
        resp = await client.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        return resp.data[0].embedding
    except Exception:
        # Degraded mode: extraction still works, vector recall just won't find this memory.
        return None


async def embed_batch(texts: list[str]) -> list[list[float] | None]:
    """Embed many strings in one API call. Returns same-length list, None on per-item failure."""
    if not settings.openai_api_key or not texts:
        return [None] * len(texts)
    valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
    if not valid_indices:
        return [None] * len(texts)
    try:
        client = get_client()
        resp = await client.embeddings.create(
            model=settings.embedding_model,
            input=[texts[i] for i in valid_indices],
        )
        result: list[list[float] | None] = [None] * len(texts)
        for vi, data in zip(valid_indices, resp.data):
            result[vi] = data.embedding
        return result
    except Exception:
        return [None] * len(texts)