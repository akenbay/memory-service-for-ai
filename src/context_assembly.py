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
    max_tokens: int,
) -> AssembledContext:
    """
    Build the prose context that goes into /recall's response.
    Respects max_tokens with per-section budgets.
    """
    if not stable and not recalled:
        return AssembledContext(context="", citations=[])

    # Deduplicate: don't repeat a memory in both sections.
    stable_ids = {m.id for m in stable}
    recalled = [m for m in recalled if m.id not in stable_ids]

    stable_budget = int(max_tokens * STABLE_FACTS_BUDGET_FRAC)
    recall_budget = max_tokens - stable_budget  # remainder; never undershoot

    sections: list[str] = []
    citations: list[Citation] = []

    # Section 1: stable user facts.
    if stable:
        header = "## Known facts about this user"
        lines = [header]
        used = count_tokens(header)
        for m in stable:
            line = _format_fact_line(m)
            cost = count_tokens(line) + 1  # +1 for newline
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

    context = "\n\n".join(sections)
    return AssembledContext(context=context, citations=citations)