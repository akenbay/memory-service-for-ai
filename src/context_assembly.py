"""
Assemble the formatted context string returned by /recall.

Priority order (defended in README):
  1. Stable user facts (always included if they exist).
  2. Query-relevant memories from hybrid retrieval (deduplicated against #1).
  3. (Phase 5+) Recent session context.

Token budget: count tokens with tiktoken; drop lowest-priority items
when over budget. Sections shrink independently — we never let recall
results crowd out core user facts.
"""
from dataclasses import dataclass
from typing import Optional

import tiktoken

from src.recall import RetrievedMemory


# Budget split — see Phase 5 for the formal justification.
STABLE_FACTS_BUDGET_FRAC = 0.45
RECALL_BUDGET_FRAC = 0.55

# Tokenizer for budget accounting. cl100k_base is used by GPT-4 family
# and our embedding model — close enough for any frozen LLM.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(s: str) -> int:
    return len(_TOKENIZER.encode(s))


@dataclass
class Citation:
    turn_id: str
    score: float
    snippet: str


@dataclass
class AssembledContext:
    context: str
    citations: list[Citation]


def _format_fact_line(m: RetrievedMemory) -> str:
    """Bullet line for the stable-facts section."""
    return f"- {m.value}"


def _format_recall_line(m: RetrievedMemory) -> str:
    """Bullet line for the recalled-memories section, with date prefix."""
    date = m.created_at[:10]  # YYYY-MM-DD
    return f"- [{date}] {m.value}"


def assemble_context(
    stable: list[RetrievedMemory],
    recalled: list[RetrievedMemory],
    recent_turns: list[dict],
    max_tokens: int,
) -> AssembledContext:
    """
    Build the prose context that goes into /recall's response.
    Respects max_tokens with per-section budgets.

    Priority (when budget is tight):
      1. Stable facts — always included if they exist. The agent must know
         who the user is before anything else.
      2. Query-relevant recall — what the question is about.
      3. Recent session turns — same-session continuity, smallest slice.
    """
    if not stable and not recalled and not recent_turns:
        return AssembledContext(context="", citations=[])

    # Deduplicate stable vs recalled.
    stable_ids = {m.id for m in stable}
    recalled = [m for m in recalled if m.id not in stable_ids]

    stable_budget = int(max_tokens * STABLE_FACTS_BUDGET_FRAC)
    recall_budget = int(max_tokens * RECALL_BUDGET_FRAC)
    session_budget = max_tokens - stable_budget - recall_budget

    sections: list[str] = []
    citations: list[Citation] = []

    # Section 1: stable user facts.
    if stable:
        header = "## Known facts about this user"
        lines = [header]
        used = count_tokens(header)
        for m in stable:
            line = _format_fact_line(m)
            cost = count_tokens(line) + 1
            if used + cost > stable_budget:
                break
            lines.append(line)
            used += cost
            citations.append(Citation(
                turn_id=m.source_turn_id,
                score=m.score,
                snippet=m.value,
            ))
        if len(lines) > 1:
            sections.append("\n".join(lines))

    # Section 2: query-relevant recall.
    if recalled:
        header = "## Relevant from recent conversations"
        lines = [header]
        used = count_tokens(header)
        for m in recalled:
            line = _format_recall_line(m)
            cost = count_tokens(line) + 1
            if used + cost > recall_budget:
                break
            lines.append(line)
            used += cost
            citations.append(Citation(
                turn_id=m.source_turn_id,
                score=m.score,
                snippet=m.value,
            ))
        if len(lines) > 1:
            sections.append("\n".join(lines))

    # Section 3: same-session continuity.
    if recent_turns:
        header = "## Recent in this session"
        lines = [header]
        used = count_tokens(header)
        for turn in recent_turns:
            summary = _summarize_turn(turn)
            if summary is None:
                continue
            cost = count_tokens(summary) + 1
            if used + cost > session_budget:
                break
            lines.append(summary)
            used += cost
        if len(lines) > 1:
            sections.append("\n".join(lines))

    context = "\n\n".join(sections)
    return AssembledContext(context=context, citations=citations)

SESSION_CONTEXT_BUDGET_FRAC = 0.15  # third section gets a small slice
# Adjust the existing splits to leave room for it
STABLE_FACTS_BUDGET_FRAC = 0.40
RECALL_BUDGET_FRAC = 0.45
# (the three should sum to 1.0)


def _summarize_turn(turn: dict) -> str | None:
    """One-line summary: '[YYYY-MM-DD] first user message, truncated.'"""
    messages = turn.get("messages") or []
    user_msg = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"),
        None,
    )
    if not user_msg:
        return None
    user_msg = user_msg.strip().replace("\n", " ")
    if len(user_msg) > 100:
        user_msg = user_msg[:97] + "..."
    date = turn.get("timestamp", "")[:10]
    return f"- [{date}] {user_msg}"