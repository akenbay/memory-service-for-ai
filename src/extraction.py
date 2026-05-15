"""LLM-based structured extraction from conversation turns."""
from typing import Optional
from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel, Field

from src.config import settings
from src.models import MemoryType


# Controlled vocabulary for the `key` field. The LLM is allowed to coin
# new keys but is told to prefer these.
KEY_VOCABULARY = [
    # identity
    "name", "age", "gender", "pronouns", "nationality",
    # location
    "current_location", "hometown", "previous_location",
    # work
    "employment", "role", "company", "industry", "work_history",
    # family
    "family_member", "partner", "children", "parents", "siblings",
    # pets
    "pet",
    # preferences
    "dietary_restriction", "allergy", "communication_style",
    "favorite_food", "favorite_music", "favorite_tool",
    # opinions
    "opinion_tool", "opinion_topic", "opinion_general",
    # skills
    "skill", "language_spoken", "language_programming",
    # events
    "upcoming_event", "past_event", "recent_activity",
    # goals
    "goal", "current_project",
    # health
    "medical_condition", "medication",
    # misc
    "hobby", "interest", "misc_fact",
]


class ExtractedMemory(BaseModel):
    """A single structured memory extracted from a turn."""
    type: MemoryType = Field(
        description="fact (objective truth), preference (likes/dislikes), "
                    "opinion (subjective view, mutable), event (time-bound)."
    )
    key: str = Field(
        description=f"Canonical key from the controlled vocabulary. "
                    f"Use snake_case. Preferred: {', '.join(KEY_VOCABULARY)}. "
                    f"Coin a new snake_case key only if nothing fits."
    )
    value: str = Field(
        description="The canonical statement, written third-person about the user. "
                    "E.g., 'Works at Notion as a PM' or 'Has a dog named Biscuit'."
    )
    evidence: str = Field(
        description="The exact span from the user's message that supports this memory. "
                    "Quote verbatim, no paraphrasing."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="0.95+ if user stated directly. 0.7-0.9 if strongly implied. "
                    "0.5-0.7 if inferred from context.",
    )


class ExtractionResult(BaseModel):
    """The full extraction output for one turn."""
    memories: list[ExtractedMemory] = Field(
        default_factory=list,
        description="Empty list if the turn contains nothing memorable."
    )


_EXTRACTION_SYSTEM_PROMPT = """\
You are a memory extraction system for an AI agent. Your job is to read a
conversation turn and extract structured, persistent facts about the USER
(not the assistant).

Extract memories that would be useful to remember in future conversations:
- Personal facts: name, age, location, employment, family, pets
- Preferences and dietary/communication preferences
- Opinions about tools, topics, things (these may evolve over time)
- Allergies, medical info
- Skills, hobbies, ongoing projects
- Events: upcoming travel, recent activities

Rules:
1. Extract memories ONLY from user messages, not assistant messages. The
   assistant is not a source of truth about the user. Exception: if the
   assistant restates a user fact ("So you live in Berlin?"), and the user
   doesn't correct it, you can rely on it — but only if the original user
   statement was extracted too.
2. Capture IMPLICIT facts. "Walking Biscuit this morning" → the user has a
   pet named Biscuit. "My partner and I" → the user has a partner.
3. Capture CORRECTIONS. If a user says "actually I meant X, not Y", extract
   X with high confidence and note the correction in the evidence span.
4. Do NOT extract conversational filler ("thanks", "got it", "interesting").
5. Do NOT extract assistant claims about itself.
6. If a turn has no memorable user content, return an empty memories list.
7. Use the controlled vocabulary for `key` whenever possible. Two memories
   about the same aspect of the user (e.g., where they work) must share
   the same key so we can detect contradictions later.
8. `value` should be a canonical third-person statement, not a quote.
9. `evidence` must be the verbatim span from the message — used for audit.

Examples:

Input: "I just moved from NYC to Berlin last month. Loving it so far."
Output:
  - {type: fact, key: current_location, value: "Lives in Berlin", evidence: "I just moved from NYC to Berlin", confidence: 0.95}
  - {type: fact, key: previous_location, value: "Previously lived in NYC", evidence: "moved from NYC to Berlin", confidence: 0.95}

Input: "Walking Biscuit this morning was nice."
Output:
  - {type: fact, key: pet, value: "Has a pet named Biscuit (likely a dog, walks it)", evidence: "Walking Biscuit this morning", confidence: 0.85}

Input: "Thanks, that was helpful!"
Output:
  - (empty)

Input: "I love TypeScript. Actually wait, I find its generics annoying."
Output:
  - {type: opinion, key: opinion_tool, value: "Finds TypeScript generics annoying (corrected from 'loves TypeScript')", evidence: "Actually wait, I find its generics annoying", confidence: 0.85}
"""


_extractor_client: instructor.AsyncInstructor | None = None


def get_extractor() -> instructor.AsyncInstructor:
    global _extractor_client
    if _extractor_client is None:
        _extractor_client = instructor.from_openai(
            AsyncOpenAI(api_key=settings.openai_api_key)
        )
    return _extractor_client


def _format_turn_for_extraction(messages: list[dict]) -> str:
    """Render the turn as a transcript the LLM can read."""
    lines = []
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        name = m.get("name")
        if name:
            lines.append(f"[{role}:{name}] {content}")
        else:
            lines.append(f"[{role}] {content}")
    return "\n".join(lines)


async def extract_memories(messages: list[dict]) -> list[ExtractedMemory]:
    """
    Extract structured memories from a turn. Returns [] on any failure
    (no API key, LLM error, timeout) so /turns still succeeds.
    """
    if not settings.openai_api_key:
        return []
    if not messages:
        return []

    transcript = _format_turn_for_extraction(messages)
    if not transcript.strip():
        return []

    try:
        client = get_extractor()
        result = await client.chat.completions.create(
            model=settings.extraction_model,
            response_model=ExtractionResult,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Conversation turn:\n\n{transcript}"},
            ],
            temperature=0.1,
            max_retries=1,
        )
        return result.memories
    except Exception:
        # Graceful degradation: turn is still persisted, just without extraction.
        return []