# Changelog

## v0 — Skeleton

**What:** FastAPI + Postgres/pgvector under docker-compose. All seven endpoints return correctly-shaped stubs. Spec §7 smoke test passes.

**Stack choices:**

- **Postgres + pgvector (single store):** fact evolution is a transactional supersession problem. One store keeps supersession atomic and gives me `tsvector` for the BM25 half of hybrid retrieval in the same query.
- **Python + FastAPI:** mature ecosystem for structured LLM output and pgvector.
- **OpenAI `gpt-4o-mini` + `text-embedding-3-small`:** cheap, fast, one provider.

**Deferred:** all logic. v0 is contract correctness and a working Docker stack.

**Next:** persistence layer (`turns`, `memories` tables); wire `POST /turns` to actually insert.

## v1 — Persistence layer

**What:** Two tables (`turns`, `memories`) with SQLAlchemy 2.0 async ORM. Postgres extensions (`vector`) and indexes (HNSW on embeddings, GIN on `content_tsv`, partial index on active memories) created on startup. `POST /turns` persists; `GET /users/{id}/memories` reads from DB; DELETEs cascade. Verified turns survive `docker compose restart`.

**Design notes:**

- **`memories.key` is a controlled-vocabulary column** to make supersession tractable: two memories with the same `(user_id, key)` are candidates for contradiction-checking later.
- **`supersedes` self-FK + `active` boolean** preserves history for `/users/{id}/memories` while keeping `/recall` filtered to current truth.
- **`evidence` column** stores the source span — enables citation provenance in `/recall`.
- HNSW chosen over ivfflat for pgvector; better recall on small datasets and the index is small enough not to matter.

**Deferred:** extraction. `POST /turns` persists the raw turn but doesn't derive memories yet.

**Next:** LLM-based structured extraction with controlled vocabulary.
