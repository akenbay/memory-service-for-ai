"""
Contradiction judge: given two memories for the same key, classify their
relationship as identical / contradictory / additive.

Most keys go through a small LLM call. A few obviously-single-valued keys
short-circuit to 'contradictory' to save latency on the hot path.
"""
from enum import Enum
from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel, Field

from src.config import settings


# Keys where a new value always replaces the old. The user only has one
# current location, one age, one name at a time.
SINGLE_VALUED_KEYS = {
    "name", "age", "gender", "pronouns", "nationality",
    "current_location", "hometown",
    "employment", "role", "company",
    "partner",
    "communication_style",
}


class Relationship(str, Enum):
    IDENTICAL = "identical"
    CONTRADICTORY = "contradictory"
    ADDITIVE = "additive"


class JudgeOutput(BaseModel):
    relationship: Relationship = Field(
        description=(
            "identical: same fact, just paraphrased. "
            "contradictory: new fact replaces old (mutually exclusive). "
            "additive: both facts can be true at once about the user."
        )
    )
    reasoning: str = Field(
        description="One short sentence explaining the call."
    )


_JUDGE_SYSTEM_PROMPT = """\
You compare two pieces of structured knowledge about the same user.
Both share a category (e.g., 'employment', 'pet', 'opinion_tool'). Decide
how they relate:

- identical: same fact, different phrasing.
  Example: "Lives in Berlin" vs "Currently based in Berlin" -> identical.

- contradictory: only one can be true at the same time.
  Example: "Works at Stripe" vs "Works at Notion" -> contradictory.
  Example: "Lives in Berlin" vs "Lives in NYC" -> contradictory.
  Example: "Loves TypeScript" vs "Finds TypeScript annoying" -> contradictory (opinion changed).

- additive: both can be true simultaneously about the user.
  Example: "Speaks Python" vs "Speaks Rust" -> additive (people speak multiple languages).
  Example: "Has a dog named Biscuit" vs "Has a cat named Mango" -> additive.
  Example: "Hobby: rock climbing" vs "Hobby: woodworking" -> additive.

When in doubt between contradictory and identical, choose identical.
When in doubt between contradictory and additive, look at the category:
single-valued aspects (where/when/who-am-I) lean contradictory; collections
(skills, hobbies, pets, languages) lean additive.
"""


_judge_client: instructor.AsyncInstructor | None = None


def get_judge() -> instructor.AsyncInstructor:
    global _judge_client
    if _judge_client is None:
        _judge_client = instructor.from_openai(
            AsyncOpenAI(api_key=settings.openai_api_key)
        )
    return _judge_client


async def classify(
    key: str,
    existing_value: str,
    new_value: str,
) -> Relationship:
    """
    Decide how a new memory relates to an existing memory with the same key.

    Fast path for known single-valued keys; LLM for everything else.
    Returns CONTRADICTORY on any failure — safer default than ADDITIVE,
    since misclassifying additive as contradictory only loses the older
    fact (still in DB), but the reverse leaves two stale truths active.
    """
    # Trivial case: textually identical values.
    if existing_value.strip().lower() == new_value.strip().lower():
        return Relationship.IDENTICAL

    # Fast path: known single-valued keys are always contradictory when
    # the values differ.
    if key in SINGLE_VALUED_KEYS:
        return Relationship.CONTRADICTORY

    # LLM path for everything else.
    if not settings.openai_api_key:
        return Relationship.CONTRADICTORY  # safer default; see docstring

    user_prompt = (
        f"Category: {key}\n"
        f"Existing: {existing_value}\n"
        f"New: {new_value}"
    )
    try:
        client = get_judge()
        result = await client.chat.completions.create(
            model=settings.extraction_model,
            response_model=JudgeOutput,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_retries=1,
        )
        return result.relationship
    except Exception:
        return Relationship.CONTRADICTORY