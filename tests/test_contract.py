"""
Spec §7 contract tests.

Verifies request/response shapes, restart persistence, multi-user isolation,
and malformed-input handling. Uses sync httpx against the running service,
same style as test_recall_quality.py.

test_restart_persistence is marked `slow` and auto-skips when the docker CLI
isn't available (i.e., when this file is being run from inside the api
container). Run it from the host:

    pytest tests/test_contract.py::test_restart_persistence -s
"""
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone

import httpx
import pytest


BASE_URL = os.getenv("MEMORY_SERVICE_URL", "http://localhost:8080")


@pytest.fixture
def client():
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as c:
        yield c


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wait_for_health(client: httpx.Client, timeout: float = 30.0) -> None:
    """Poll /health until ok or timeout. Used after restart."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = client.get("/health")
            if r.status_code == 200 and r.json().get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"service did not become healthy within {timeout}s")


def test_contract_roundtrip(client):
    """POST /turns, POST /recall, GET memories, DELETE /users — all shapes correct."""
    user_id = f"contract-rt-{uuid.uuid4()}"
    session_id = f"rt-session-{uuid.uuid4()}"

    try:
        # POST /turns -> 201 with {id: UUID}
        turn = client.post("/turns", json={
            "session_id": session_id,
            "user_id": user_id,
            "messages": [
                {"role": "user", "content": "I live in Tokyo and work at Anthropic as an engineer."},
                {"role": "assistant", "content": "Nice."},
            ],
            "timestamp": _utc_now_iso(),
            "metadata": {},
        })
        assert turn.status_code == 201, turn.text
        body = turn.json()
        assert "id" in body
        uuid.UUID(body["id"])  # raises if not a UUID

        # POST /recall -> 200 with {context: str, citations: list}
        rec = client.post("/recall", json={
            "query": "Where does the user live?",
            "session_id": session_id,
            "user_id": user_id,
            "max_tokens": 512,
        })
        assert rec.status_code == 200, rec.text
        rec_body = rec.json()
        assert isinstance(rec_body.get("context"), str)
        assert isinstance(rec_body.get("citations"), list)
        for c in rec_body["citations"]:
            assert {"turn_id", "score", "snippet"} <= set(c.keys())

        # GET /users/{id}/memories -> 200 with {memories: list}
        mem = client.get(f"/users/{user_id}/memories")
        assert mem.status_code == 200
        mem_body = mem.json()
        assert isinstance(mem_body.get("memories"), list)
        for m in mem_body["memories"]:
            assert {"id", "type", "key", "value", "active"} <= set(m.keys())

        # DELETE /users/{id} -> 204
        deletion = client.delete(f"/users/{user_id}")
        assert deletion.status_code == 204

        # After delete, memories list is empty.
        after = client.get(f"/users/{user_id}/memories")
        assert after.status_code == 200
        assert after.json()["memories"] == []
    finally:
        client.delete(f"/users/{user_id}")


@pytest.mark.slow
def test_restart_persistence(client):
    """Memories must survive a container restart (named volume + transactional writes).

    Skipped automatically when run from inside the api container (no docker CLI).
    Run from host to actually exercise: pytest tests/test_contract.py -m slow -s
    """
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not available (likely running inside the api container)")

    user_id = f"contract-restart-{uuid.uuid4()}"
    session_id = f"restart-session-{uuid.uuid4()}"

    try:
        turn = client.post("/turns", json={
            "session_id": session_id,
            "user_id": user_id,
            "messages": [
                {"role": "user", "content": "I have a parrot named Mango and my favorite color is teal."},
                {"role": "assistant", "content": "Got it."},
            ],
            "timestamp": _utc_now_iso(),
            "metadata": {},
        })
        assert turn.status_code == 201

        before = client.get(f"/users/{user_id}/memories").json()["memories"]
        # We need extraction to actually have run for this test to be meaningful.
        if not before:
            pytest.skip("no memories were extracted (OPENAI_API_KEY likely unset)")

        subprocess.run(
            ["docker", "compose", "restart", "api"],
            check=True, capture_output=True,
        )
        _wait_for_health(client)

        after = client.get(f"/users/{user_id}/memories").json()["memories"]
        assert len(after) == len(before), (
            f"memories did not survive restart: {len(before)} before, {len(after)} after"
        )
        # Same ids — not just same count
        assert {m["id"] for m in after} == {m["id"] for m in before}

        # Recall still works after restart.
        rec = client.post("/recall", json={
            "query": "What pet does the user have?",
            "session_id": session_id,
            "user_id": user_id,
            "max_tokens": 512,
        })
        assert rec.status_code == 200
    finally:
        client.delete(f"/users/{user_id}")


def test_concurrent_sessions(client):
    """Two user_ids with different facts — no cross-bleed in storage or recall."""
    user_a = f"contract-conc-a-{uuid.uuid4()}"
    user_b = f"contract-conc-b-{uuid.uuid4()}"

    try:
        client.post("/turns", json={
            "session_id": f"sess-a-{uuid.uuid4()}",
            "user_id": user_a,
            "messages": [
                {"role": "user", "content": "I'm Alice. I work at Stripe on payments infrastructure."},
                {"role": "assistant", "content": "Cool."},
            ],
            "timestamp": _utc_now_iso(),
            "metadata": {},
        })
        client.post("/turns", json={
            "session_id": f"sess-b-{uuid.uuid4()}",
            "user_id": user_b,
            "messages": [
                {"role": "user", "content": "I'm Bob. I work at Datadog on observability tooling."},
                {"role": "assistant", "content": "Nice."},
            ],
            "timestamp": _utc_now_iso(),
            "metadata": {},
        })

        mem_a = client.get(f"/users/{user_a}/memories").json()["memories"]
        mem_b = client.get(f"/users/{user_b}/memories").json()["memories"]

        # If extraction didn't run, the test is uninformative — skip.
        if not mem_a or not mem_b:
            pytest.skip("extraction produced no memories (OPENAI_API_KEY likely unset)")

        a_blob = " ".join(m["value"] for m in mem_a).lower()
        b_blob = " ".join(m["value"] for m in mem_b).lower()

        assert "datadog" not in a_blob, f"user A leaked B's data: {a_blob}"
        assert "bob" not in a_blob, f"user A leaked B's data: {a_blob}"
        assert "stripe" not in b_blob, f"user B leaked A's data: {b_blob}"
        assert "alice" not in b_blob, f"user B leaked A's data: {b_blob}"

        # /recall scoped to user A should not surface B's facts.
        rec_a = client.post("/recall", json={
            "query": "Where does the user work?",
            "session_id": f"probe-a-{uuid.uuid4()}",
            "user_id": user_a,
            "max_tokens": 512,
        }).json()
        ctx_a = (rec_a.get("context") or "").lower()
        assert "datadog" not in ctx_a
        assert "bob" not in ctx_a
    finally:
        client.delete(f"/users/{user_a}")
        client.delete(f"/users/{user_b}")


def test_malformed_input(client):
    """Bad input returns 4xx — never crashes the service with a 5xx."""
    user_id = f"contract-mal-{uuid.uuid4()}"

    try:
        # (a) Invalid JSON body.
        bad_json = client.post(
            "/turns",
            content=b"this is not json",
            headers={"content-type": "application/json"},
        )
        assert 400 <= bad_json.status_code < 500, (
            f"invalid JSON should yield 4xx, got {bad_json.status_code}: {bad_json.text}"
        )

        # (b) Missing required fields.
        missing = client.post("/turns", json={"session_id": "incomplete"})
        assert missing.status_code == 422, missing.text

        # (c) Unicode + emoji content — should succeed.
        unicode_resp = client.post("/turns", json={
            "session_id": f"unicode-{uuid.uuid4()}",
            "user_id": user_id,
            "messages": [
                {"role": "user", "content": "私の名前は山田です 🍜 I love ramen."},
                {"role": "assistant", "content": "素晴らしい！"},
            ],
            "timestamp": _utc_now_iso(),
            "metadata": {},
        })
        assert unicode_resp.status_code == 201, unicode_resp.text

        # (d) Oversized payload (~1MB content). Either accepted or 4xx, never 5xx.
        big_text = "x" * 1_000_000
        oversized = client.post("/turns", json={
            "session_id": f"big-{uuid.uuid4()}",
            "user_id": user_id,
            "messages": [
                {"role": "user", "content": big_text},
                {"role": "assistant", "content": "ok"},
            ],
            "timestamp": _utc_now_iso(),
            "metadata": {},
        })
        assert oversized.status_code < 500, (
            f"oversized payload returned 5xx: {oversized.status_code} {oversized.text[:200]}"
        )
    finally:
        client.delete(f"/users/{user_id}")
