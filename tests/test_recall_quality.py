"""
Recall quality fixture runner.

Ingests fixture scenarios, runs probe queries against /recall, scores them.
Prints a per-scenario and overall score. Designed to be re-run after every
recall pipeline change so each CHANGELOG entry has a measured delta.
"""
import json
import os
from pathlib import Path

import httpx
import pytest


BASE_URL = os.getenv("MEMORY_SERVICE_URL", "http://localhost:8080")
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _load_scenarios() -> list[dict]:
    scenarios = []
    for path in sorted(FIXTURES_DIR.glob("scenario_*.json")):
        with open(path) as f:
            scenarios.append(json.load(f))
    return scenarios


def _ingest_scenario(client: httpx.Client, scenario: dict) -> None:
    """POST each turn in the scenario to /turns."""
    for turn in scenario["turns"]:
        resp = client.post(
            "/turns",
            json={
                "session_id": turn["session_id"],
                "user_id": scenario["user_id"],
                "messages": turn["messages"],
                "timestamp": turn["timestamp"],
                "metadata": {},
            },
        )
        assert resp.status_code == 201, (
            f"Ingest failed for {scenario['name']}: {resp.status_code} {resp.text}"
        )


def _cleanup(client: httpx.Client, scenario: dict) -> None:
    client.delete(f"/users/{scenario['user_id']}")


def _score_probe(probe: dict, recall_response: dict) -> tuple[bool, str]:
    """Returns (passed, reason)."""
    context = (recall_response.get("context") or "").lower()

    # Noise resistance: forbidden terms must NOT appear.
    if probe.get("expect_empty_or_irrelevant"):
        forbidden = [t.lower() for t in probe.get("forbidden_terms", [])]
        leaked = [t for t in forbidden if t in context]
        if leaked:
            return False, f"hallucinated forbidden terms: {leaked}"
        return True, "no hallucinated content"

    # expected_any: at least one must appear.
    expected_any = [e.lower() for e in probe.get("expected_any", [])]
    if expected_any and not any(e in context for e in expected_any):
        return False, f"missing all expected: {expected_any}"

    # expected_not: any of these in context is a fail (used for stale-fact detection).
    expected_not = [e.lower() for e in probe.get("expected_not", [])]
    leaked = [e for e in expected_not if e in context]
    if leaked:
        return False, f"stale facts present: {leaked}"

    # expected_not_strict: even substring match is a fail.
    not_strict = [e.lower() for e in probe.get("expected_not_strict", [])]
    leaked = [e for e in not_strict if e in context]
    if leaked:
        return False, f"forbidden terms present (strict): {leaked}"

    return True, "ok"


def test_recall_quality():
    """Run all fixture scenarios and report a recall-quality score."""
    scenarios = _load_scenarios()
    assert scenarios, "No fixture scenarios found in fixtures/"

    overall_passes = 0
    overall_total = 0
    report_lines = ["", "=" * 70, "RECALL QUALITY REPORT", "=" * 70]

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        for scenario in scenarios:
            _cleanup(client, scenario)  # paranoid: clear before run
            _ingest_scenario(client, scenario)

            scenario_passes = 0
            scenario_total = len(scenario["probes"])
            scenario_lines = [f"\n[{scenario['name']}]"]

            for probe in scenario["probes"]:
                resp = client.post(
                    "/recall",
                    json={
                        "query": probe["query"],
                        "session_id": probe.get("session_id", "probe-session"),
                        "user_id": scenario["user_id"],
                        "max_tokens": 1024,
                    },
                )
                assert resp.status_code == 200, (
                    f"/recall failed: {resp.status_code} {resp.text}"
                )
                passed, reason = _score_probe(probe, resp.json())
                if passed:
                    scenario_passes += 1
                    scenario_lines.append(f"  ✓ {probe['query']}")
                else:
                    scenario_lines.append(f"  ✗ {probe['query']}  ({reason})")

            overall_passes += scenario_passes
            overall_total += scenario_total
            scenario_lines.append(
                f"  → {scenario_passes}/{scenario_total} passed"
            )
            report_lines.extend(scenario_lines)

            _cleanup(client, scenario)

    score = overall_passes / overall_total if overall_total else 0.0
    report_lines.extend([
        "",
        "-" * 70,
        f"OVERALL: {overall_passes}/{overall_total} probes passed ({score:.0%})",
        "=" * 70,
        "",
    ])
    print("\n".join(report_lines))

    # Don't assert on the score — this test reports, it doesn't gate.
    # The number is what matters; we track its trajectory in the CHANGELOG.