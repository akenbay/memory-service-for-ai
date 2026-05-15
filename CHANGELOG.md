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

## v2 — Extraction pipeline

**What:** `POST /turns` now runs synchronous LLM-based extraction (gpt-4o-mini via `instructor`) and persists typed structured memories with embeddings (text-embedding-3-small, 1536d) and a populated `content_tsv` column for FTS — all in one transaction with the turn itself.

**Design:**

- **Controlled vocabulary of ~25 keys** (employment, current_location, pet, dietary_restriction, opinion_tool, etc.). The LLM is told to prefer them and coin new snake_case keys only when nothing fits. This is what makes supersession tractable in v6: two memories about the same aspect of the user share a key.
- **`instructor` + Pydantic schema** instead of raw JSON parsing. The LLM is forced to produce a valid `ExtractionResult` or `instructor` retries automatically.
- **`evidence` field** stores the verbatim source span — enables provenance and citations in `/recall`.
- **Four memory types** — fact (immutable truth), preference (stable but updatable), opinion (mutable, evolves), event (time-bound).
- **Temperature 0.1** for consistency without rigidity.
- **Batched embeddings** — one API call per turn, not N.
- **Graceful degradation:** missing API key, LLM errors, or timeouts → `/turns` still returns 201, just without extracted memories. The raw turn always persists.

**What I deliberately don't extract:**

- Assistant claims about the assistant.
- Conversational filler.
- Facts from assistant messages unless they restate user-stated facts.

**What I miss (acknowledged):**

- Temporal qualifiers ("I'll be in Berlin next week" is extracted as a current event but the _future_ qualifier isn't structured).
- Relationships between facts beyond shared keys (e.g., "my partner Alex is vegetarian" — currently extracts partner and dietary_restriction separately; the link between them is lost).
- Sentiment trajectories in opinions (treated as plain supersession candidates in v6 — see CHANGELOG v6 for the tradeoff).

**Verified:** Ingested a multi-fact turn ("moved from NYC to Berlin for a PM role at Notion, dog Biscuit, vegetarian") and `/users/alice/memories` returned 5 typed records with appropriate keys and high confidence.

**Next:** Build the recall-quality fixture so I can measure every change from here on.
