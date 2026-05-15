# Changelog

## v0 — Skeleton

**What:** FastAPI + Postgres/pgvector under docker-compose. All seven endpoints return correctly-shaped stubs. Spec §7 smoke test passes.

**Stack choices:**

- **Postgres + pgvector (single store):** fact evolution is a transactional supersession problem. One store keeps supersession atomic and gives me `tsvector` for the BM25 half of hybrid retrieval in the same query.
- **Python + FastAPI:** mature ecosystem for structured LLM output and pgvector.
- **OpenAI `gpt-4o-mini` + `text-embedding-3-small`:** cheap, fast, one provider.

**Deferred:** all logic. v0 is contract correctness and a working Docker stack.

**Next:** persistence layer (`turns`, `memories` tables); wire `POST /turns` to actually insert.
