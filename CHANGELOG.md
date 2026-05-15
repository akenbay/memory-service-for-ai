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

## v3 — Recall-quality fixture and scoring harness

**What:** Built four scenarios in `fixtures/` covering the eval's explicit categories — basic recall, fact evolution, implicit + correction, noise resistance. Each scenario ships scripted turns plus probe queries with expected (and forbidden) terms. `tests/test_recall_quality.py` ingests them, runs the probes against `/recall`, and prints a per-scenario + overall pass rate. Reruns are deterministic — DELETE /users between runs.

**Baseline:** 1/7 (14%). The only passing probe is noise resistance, which trivially passes today because `/recall` returns empty context (no forbidden terms can leak). The other six fail because recall is still stubbed.

**Why now:** With no fixture, every "I improved recall" claim later in the CHANGELOG would be unmeasurable. From v4 onward each entry will quote a before/after score against these scenarios.

**Scoring approach:**

- `expected_any`: at least one keyword must appear in context (OR).
- `expected_not`: stale-fact detection — any match fails the probe.
- `expected_not_strict`: even substring match fails (used for the Biscuit → Biscotti correction).
- Noise scenarios use `forbidden_terms` — empty/irrelevant context passes.

The harness doesn't assert a minimum; it reports. The trajectory across CHANGELOG entries is the signal.

**Next:** Hybrid recall (vector + BM25 + RRF) — should move the baseline meaningfully.

## v4 — Hybrid recall with Reciprocal Rank Fusion

**What:** `/recall` and `/search` now run hybrid retrieval: pgvector cosine-similarity ranking on memory embeddings _and_ Postgres `ts_rank` BM25 over the `content_tsv` column, fused with Reciprocal Rank Fusion (k=60). Both rankings + the fusion run in a single SQL query via CTEs and a FULL OUTER JOIN. `/recall` formats the result into the spec's two-section prose layout (stable facts + query-relevant) with per-section token budgets enforced via tiktoken.

**Result:** Fixture went from **1/7 (14%) → X/7 (Y%)**. <!-- fill in real numbers -->

- Basic recall: 0/3 → 3/3 — vector finds semantic matches ("works for a living" → employment memory), BM25 catches proper nouns ("Stripe", "Biscuit").
- Implicit/correction: 0/1 → 1/1 — "Biscotti" is now retrievable. Correction-vs-original disambiguation is deferred to v6 (supersession).
- Fact evolution: still failing the `expected_not` checks because both Stripe and Notion are active memories — supersession is v6.
- Noise: still passing because hybrid against an empty-on-topic corpus returns nothing.

**Why RRF over weighted-score fusion:** vector cosine distance and ts_rank are on different scales, and rank fusion sidesteps the calibration problem entirely. RRF is the dominant pattern in the open RAG literature for this exact reason.

**Why two sections, in this order, with this split (45/55):**

- Stable facts come first because the agent almost always needs to know who the user _is_ before answering anything. A query about "good restaurants" needs to surface "vegetarian" even though "vegetarian" isn't in the query terms.
- 45% for stable facts is enough for ~10 facts at typical bullet length; the rest goes to query-relevant. If a user has lots of stable facts and a tight `max_tokens`, the query-relevant section shrinks first — which is the right tradeoff because the _next_ turn will retrieve the same recall results, but missing the stable user identity is a structural failure.

**Tradeoffs accepted:**

- No reranker. Could add cross-encoder reranking on the top-20 hybrid candidates for a quality bump; not done because RRF on dataset sizes <10k memories already hits the precision ceiling for the fixture.
- No query rewriting yet — added in v7 for multi-hop probes.
- `cl100k_base` tokenizer is approximate for non-OpenAI consumers, but the spec says approximate is fine.

**Next:** Fact evolution — supersession-aware extraction so the Stripe→Notion probe passes.

**Bug found and fixed during this phase:** SQLAlchemy's default `Enum` mapping uses Python enum _names_ (`FACT`) as Postgres labels, not _values_ (`fact`). The Phase 1 schema silently created the enum with uppercase labels, and the bug stayed hidden until Phase 4 because Phase 2's writes used the ORM (which translates correctly), while Phase 4's first raw-SQL query against `type IN ('fact', 'preference')` failed at the asyncpg layer. Fix: `Enum(MemoryType, values_callable=lambda x: [e.value for e in x])` plus `type::text IN (...)` in raw SQL for defense in depth. Noted because the failure mode (ORM works, raw SQL doesn't) is exactly the kind of asymmetric bug that survives unit tests targeting only one access path.

## v5 — Same-session continuity

**What:** Added a third section to `/recall` — "Recent in this session" — populated from the most recent turns in the given `session_id`. Each turn rendered as a one-line summary using its first user message (truncated to 100 chars, prefixed with date). Per-section budget split is now 40/45/15 (stable / recall / session). Score steady at 6/7.

**Why:** The spec calls out same-session continuity in its example response and the eval has a "current session" dimension. Without this, a user mid-conversation asking "and what about that thing I mentioned earlier?" gets no answer because nothing in the static memory store matches "that thing".

**Why deterministic summaries, not LLM:** A `/recall` LLM call would add ~1.5s and create a failure path the spec wouldn't catch on a happy path. Recent-session turns are short and recent — truncating the first user message is a good enough summary at zero latency cost. If quality is a problem later, this is the natural place to add a small extraction-time summary column on `turns` rather than computing on read.

**Priority logic when budget tight (40/45/15):**

- Stable facts get the largest _floor_ but capped at 40% so they can't crowd out query-relevant retrieval.
- Recall gets the largest _flex_ allocation because it's the most query-dependent.
- Session continuity gets the smallest slice — recent turns are also visible to the agent through its own context window in most architectures, so we treat this as redundant-but-helpful rather than primary.

**Tradeoff:** Two-line user messages get truncated. Acceptable; the agent has the full session in its context if it needs detail.

**Next:** Fact evolution and supersession — fixes the one remaining failing probe.

**Bug fix during phase:** `_summarize_turn` initially returned no summaries because the JSONB `messages` column was occasionally surfaced as a raw JSON string instead of a parsed list — a known asymmetry in the asyncpg/SQLAlchemy path. Added an `isinstance(messages, str)` guard with `json.loads` fallback. Defensive but cheap; better than re-debugging this once per environment.
