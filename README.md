# Memory Service

A persistent memory service for AI agents. Ingests conversation turns,
extracts structured facts about the user, and serves a budgeted recall
context on demand. One Postgres instance backs both vector and full-text
retrieval, and supersession is transactional.

The self-eval fixture in `fixtures/` currently scores **7/7 (100%)**.

---

## 1. Quickstart

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

docker compose up -d
# wait ~10s for the API to come up healthy

curl http://localhost:8080/health
# {"status": "ok"}
```

The API is bound to `:8080`. Postgres is internal to the compose network
and is not exposed.

Hot reload during development: `src/`, `tests/`, and `fixtures/` are
bind-mounted into the API container, so editing them only requires
`docker compose restart api` — no rebuild.

---

## 2. Architecture

```
                          +-------------------------------+
                          |   FastAPI (src/main.py)       |
   POST /turns  --------> |                               |
                          |   write_turn:                 |
                          |     1. persist turn           |
                          |     2. extract memories (LLM) |
                          |     3. classify vs existing   |
                          |     4. embed + insert         |
                          |     5. populate content_tsv   |
                          |        (one transaction)      |
                          +---------------+---------------+
                                          |
                          +---------------v---------------+
                          |  Postgres + pgvector          |
                          |                               |
                          |  turns:     raw transcripts   |
                          |  memories:  typed facts       |
                          |             - HNSW(embedding) |
                          |             - GIN(content_tsv)|
                          |             - supersedes FK   |
                          |             - active flag     |
                          +---------------+---------------+
                                          |
                          +---------------v---------------+
   POST /recall --------> |  recall:                      |
                          |    stable_facts SELECT        |
                          |    + hybrid (vector + BM25,   |
                          |       RRF in one SQL CTE)     |
                          |    + recent session turns     |
                          |  assemble_context:            |
                          |    3 sections, token budget   |
                          +-------------------------------+
```

`POST /turns` is the heavy path: it issues an LLM call for extraction, an
LLM call per existing same-key memory for contradiction classification
(skipped on the single-valued-key fast path), and a batched embedding
call. Everything runs inside one SQLAlchemy transaction so partial state
is impossible.

`POST /recall` is read-only and lightweight: one query to fetch stable
facts, one query to do the hybrid retrieval, one query to fetch recent
session turns, and a deterministic in-process assembly step that
respects `max_tokens`.

---

## 3. Backing store choice

**Postgres + pgvector, single store.** The decision is driven by one
observation: fact evolution is a *transactional supersession* problem.
When a new memory contradicts an existing one, we need to atomically
insert the new row, deactivate the old, and link them — all visible to
the next reader, never in a half-written state.

A single relational store buys that for free. It also lets us run vector
similarity (`embedding <=> qvec`) and BM25 (`ts_rank(content_tsv, q)`)
against the same table inside one SQL query, fused with Reciprocal Rank
Fusion in a CTE. No cross-store join, no second consistency boundary.

Alternatives I considered and rejected:

- **Qdrant + Postgres.** Two stores means reconciliation: when supersession
  runs, we have to keep the vector index in sync with the relational
  truth. The savings on dedicated-vector-DB query latency don't justify
  the operational cost at our scale (<10k memories per user).
- **SQLite + sqlite-vec.** Vector support is still immature; concurrent
  writes are weak; and we don't get a real BM25 implementation. Fine for
  a prototype, not for a "persistence is the headline feature" service.
- **MongoDB.** Schema flexibility is the wrong trait here. Memories have
  a tight, controlled schema (5–7 columns plus a controlled-vocabulary
  `key`) and the whole point of the controlled vocabulary is to make
  supersession tractable. We need *more* schema, not less.
- **Redis.** A persistence-critical service shouldn't be memory-first.
  RDB snapshots are a workaround, not an answer.

The named volume `memory_db_data` in `docker-compose.yml` is what makes
"survives `docker compose restart`" not just a feature but a property of
the deployment.

---

## 4. Extraction pipeline

`POST /turns` runs synchronous LLM extraction (`gpt-4o-mini` via
`instructor`) and persists typed structured memories in the same
transaction as the raw turn.

**Controlled vocabulary.** ~40 canonical snake_case keys
(`current_location`, `employment`, `pet`, `dietary_restriction`,
`opinion_tool`, …) defined in [src/extraction.py:13-39](src/extraction.py#L13-L39).
The LLM is told to prefer them and to coin a new key only when nothing
fits. Two memories about the same aspect of the user must share a key —
that's what makes supersession tractable later.

**Structured output.** `instructor` + a Pydantic `ExtractionResult`
schema forces the LLM to return valid records or retry. No raw-JSON
parsing in the hot path.

**Memory types** (see [src/models.py:27-31](src/models.py#L27-L31)):

- `fact` — immutable truth (name, hometown).
- `preference` — stable but updatable (dietary restriction).
- `opinion` — mutable, evolves (opinion on a tool).
- `event` — time-bound (upcoming trip, recent activity).

**What is extracted:**

- Personal facts (location, employment, family, pets).
- Preferences and dietary/communication preferences.
- Opinions about tools or topics (these may evolve).
- Skills, hobbies, ongoing projects.
- Events: upcoming travel, recent activities.
- **Implicit facts:** "walking Biscuit this morning" → user has a pet
  named Biscuit.
- **Corrections:** "actually I meant X, not Y" → extract X with high
  confidence; old value is superseded downstream.

**What is deliberately NOT extracted:**

- Assistant claims about the assistant.
- Conversational filler ("thanks", "got it").
- Facts from assistant messages unless they restate user-stated facts.
- Temporal qualifiers in structured form ("next week" is captured in
  evidence text but not modelled as a typed time window — see Tradeoffs).

**Evidence column.** Every memory stores the verbatim source span. This
is what powers the citations array in `POST /recall` — no paraphrase
drift between what the user said and what the agent is told they said.

**Graceful degradation.** Missing `OPENAI_API_KEY`, LLM timeout, schema
violation → the turn still persists, extraction returns `[]`. `/turns`
returns 201 with the turn id. See [src/extraction.py:155-183](src/extraction.py#L155-L183).

---

## 5. Recall strategy

`POST /recall` does three retrievals and one assembly step.

**Hybrid retrieval (vector + BM25 + RRF).** In a single SQL query (see
[src/recall.py:58-110](src/recall.py#L58-L110)) we:

1. Rank candidates by pgvector cosine distance against the query
   embedding (`embedding <=> CAST(:qvec AS vector)`).
2. Rank candidates by Postgres `ts_rank(content_tsv, plainto_tsquery(...))`.
3. Fuse the two with Reciprocal Rank Fusion (k=60) via a CTE and full
   outer join on memory id.
4. Filter to `active = true` and the target `user_id` at every step —
   inactive memories never leak into recall, and we never blend across
   users.

RRF over weighted-score fusion: vector cosine distance and `ts_rank` are
on different scales and not directly comparable. Rank fusion sidesteps
the calibration problem; RRF k=60 is the value from the original Cormack
et al. paper and is the de facto default in open RAG literature.

If the query embedding fails to compute (no API key, OpenAI down), the
query degrades to BM25-only — `/recall` still returns sensible results.

**Three-section context assembly** (see [src/context_assembly.py](src/context_assembly.py)):

1. **`## Known facts about this user`** — stable facts and preferences,
   ordered by confidence. Always rendered first, regardless of query
   relevance. The agent needs to know who the user *is* before answering
   anything about them.
2. **`## Relevant from recent conversations`** — hybrid-retrieval hits,
   deduplicated against the stable section.
3. **`## Recent in this session`** — terse one-line summaries of the
   last few turns in this `session_id`, for same-session continuity.

**Token budget split: 40 / 45 / 15.** Counted with `tiktoken`
(`cl100k_base`).

Why this priority order:

- Stable facts come first because a query like "good restaurants?" needs
  to surface "vegetarian" even though *"vegetarian"* isn't in the query.
  If we ran out of budget for recall results, we'd still serve a useful
  context. The reverse is not true.
- Recall gets the largest flex allocation because it's the most
  query-dependent — that's where variance lives.
- Session continuity gets the smallest slice because the agent likely
  already has the current session in its own context window. We treat
  this section as redundant-but-helpful rather than primary.

When budget is tight, sections shrink independently from the bottom up:
session continuity is the first to truncate, stable facts the last.

**Citations.** Every fact line in the output corresponds to one
`Citation(turn_id, score, snippet)` entry in the response — so the
caller can pin any claim back to a source turn.

---

## 6. Fact evolution

New memories are classified against existing same-key memories before
insertion (see [src/main.py:140-205](src/main.py#L140-L205) and
[src/contradiction.py](src/contradiction.py)).

**Three-way classification** — not binary:

- `identical` — same fact, different phrasing. Skip the insert, bump
  `updated_at` on the existing row.
- `contradictory` — only one can be true. Insert new with
  `supersedes=old.id`; mark old `active=false`.
- `additive` — both can coexist (skills, hobbies, languages, multiple
  pets). Insert new independently; leave old active.

Two-way "contradicts or not" classification would either over-supersede
(collapse two hobbies into one) or under-supersede (leave two stale jobs
active). The additive case is what lets multi-value keys coexist
correctly.

**Single-valued-key fast path.** ~12 keys
(`current_location`, `employment`, `role`, `company`, `partner`,
`age`, `name`, `gender`, `pronouns`, `nationality`, `hometown`,
`communication_style`) short-circuit to `contradictory` when values
differ — no LLM call. Saves ~500ms per write on the most common updates.
The allowlist is small and conservative; ambiguous keys fall through to
the LLM path.

**LLM judge.** Everything else goes through a small classifier prompt at
temperature 0.0 (`gpt-4o-mini` via `instructor`). The judge sees the
key, the existing value, and the new value, and returns
`identical | contradictory | additive` plus a one-sentence reason.

**Failure default: CONTRADICTORY.** If the judge fails (no API key,
timeout, schema violation), we treat the relationship as contradictory:
mark old inactive, insert new. Worst case is a stale fact gets demoted
from `/recall` but is recoverable from `/users/{id}/memories`. The
reverse default (ADDITIVE) would silently keep two truths active and
pollute agent context — strictly worse.

**Soft delete.** Superseded memories remain in the table with
`active=false`. They're visible in `GET /users/{id}/memories`
(history is preserved for inspection) and invisible in `/recall` and
`/search` (we never surface stale truth to the agent).

**Chain depth 1.** When a new memory contradicts, it points at the most
recent active same-key memory. Earlier rows in that key's history are
already inactive — they were superseded when their successor was
inserted. No need to walk the full chain on every write.

**Atomicity.** Extraction, classification, deactivation of old, and
insertion of new all run in the same SQLAlchemy transaction (see
`session_scope` in [src/db.py:49-58](src/db.py#L49-L58)). A crash mid-write
rolls back cleanly; the turn either lands fully or not at all.

**Opinion arcs — known limitation.** The spec calls out gradually
evolving opinions ("love TypeScript" → "generics are annoying" → "fine
for big projects but Python for scripts"). This implementation treats
each step as supersession: a chain of `opinion` memories with the newest
active. The chain is reconstructable from `/users/{id}/memories`, but
the *trajectory* (sentiment delta, oscillation) isn't surfaced in
`/recall`. A richer model would add a `sentiment_delta` column or a
typed `opinion_arc` table. Deferred and documented.

---

## 7. Tradeoffs

**Optimized for:**

- **Correctness.** Single store, transactional supersession, never serve
  stale truth in `/recall`.
- **Extraction quality.** Controlled vocabulary + Pydantic-validated
  output + verbatim evidence spans.
- **Fixture trajectory.** The self-eval fixture sits at 7/7. Every
  CHANGELOG entry quotes a measured delta.
- **Graceful degradation.** No API key, OpenAI down, embedding fails —
  the turn persists; recall still works (BM25-only mode); 4xx for bad
  input but never 5xx for misconfigured environment.

**Given up:**

- **Multi-hop / multi-fact composed recall.** A query like "what does
  the user's spouse do for work?" requires bridging two facts. Today
  each retrieval is single-hop. Could be addressed with query rewriting
  or a small KG layer; deferred.
- **Cross-encoder reranking.** Hybrid + RRF already saturates the
  fixture at <10k memories. Adding a cross-encoder on the top-20
  candidates would help at scale but isn't needed here.
- **Async ingestion.** `POST /turns` is synchronous and blocks on the
  extraction LLM call (~1–2s on `gpt-4o-mini`). Splitting into "persist
  turn 201" + background extraction would lower p99 write latency, at
  the cost of an inconsistency window where `/recall` doesn't see a
  just-written turn. Not done because the spec's eval is correctness-
  weighted, not latency-weighted.
- **Opinion-arc sentiment trajectories.** See section 6.
- **Horizontal scaling.** Single Postgres. For a "memory per user"
  workload this is fine well into the millions of users; we'd shard by
  `user_id` if we ever hit that wall.
- **Auth.** A `MEMORY_AUTH_TOKEN` env var is wired through config but
  not enforced on the endpoints. Trivial to add; deferred for the
  challenge.

---

## 8. Failure modes

| Condition                              | Behavior                                                                                                              |
|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `OPENAI_API_KEY` unset                 | `POST /turns` persists the turn; `extract_memories` returns `[]`. Returns 201. `/recall` works on whatever's stored.   |
| OpenAI API down                        | Same as above for extraction. For recall, vector arm is skipped; BM25-only ranking serves results.                    |
| Embedding fails for one memory         | Memory persists with `embedding=NULL`. BM25 still surfaces it. Vector arm just skips it (`WHERE embedding IS NOT NULL`). |
| Malformed input (bad JSON, missing field) | FastAPI/Pydantic returns 422. No crash.                                                                              |
| Restart mid-write                      | `session_scope` rolls the transaction back on exception. Either the turn + its memories all land, or none do.         |
| Cold session (`user_id` unknown)       | `/recall` returns `{"context": "", "citations": []}` with 200. Never errors.                                          |
| Empty query string                     | `hybrid_recall` short-circuits to `[]`; stable facts still surface if the user is known.                              |
| Container / DB restart                 | Named volume `memory_db_data` persists Postgres state. Verified in [tests/test_contract.py](tests/test_contract.py) as `test_restart_persistence`. |

---

## 9. Running tests

```bash
# All tests (skipping the slow restart-persistence test):
docker compose exec api pytest tests/ -s -m 'not slow'

# Just the recall-quality fixture:
docker compose exec api pytest tests/test_recall_quality.py -s

# Just the contract tests, including slow ones:
docker compose exec api pytest tests/test_contract.py -s
```

The fixture runner ingests four scenarios (basic recall, fact evolution,
implicit + correction, noise resistance), runs scripted probes against
`/recall`, and prints a per-scenario and overall score. It doesn't
assert a minimum — the trajectory in the CHANGELOG is the signal.

---

## 10. Repository layout

```
memory-service-for-ai/
├── README.md                  # this file
├── CHANGELOG.md               # design history with measured deltas
├── docker-compose.yml         # api + db services, named volume
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── main.py                # FastAPI endpoints
│   ├── config.py              # pydantic-settings
│   ├── db.py                  # async engine, init_db, session_scope
│   ├── models.py              # Turn, Memory ORM
│   ├── extraction.py          # LLM extraction with controlled vocabulary
│   ├── contradiction.py       # identical / contradictory / additive judge
│   ├── embeddings.py          # OpenAI embeddings (batched)
│   ├── recall.py              # hybrid_recall, get_stable_facts, get_recent_session_turns
│   └── context_assembly.py    # 3-section formatter with token budget
├── tests/
│   ├── test_recall_quality.py # fixture-driven recall scoring
│   └── test_contract.py       # spec §7 contract tests
└── fixtures/
    ├── scenario_01_basic_recall.json
    ├── scenario_02_fact_evolution.json
    ├── scenario_03_implicit_and_correction.json
    └── scenario_04_noise_resistance.json
```

See `CHANGELOG.md` for the full design history, including bug
post-mortems and the score trajectory across v0 → v8.
