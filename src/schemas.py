"""Pydantic schemas for structured agent outputs and handoffs.

Using schemas (rather than free-text) for the agent-to-agent handoff is what makes
the routing in graph.py reliable: the Triage agent must return one of a fixed set
of categories, and downstream nodes branch on that validated value.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    """The five buckets every incoming ticket is sorted into."""

    BILLING = "billing"
    TECHNICAL = "technical"
    REFUND = "refund"
    ABUSE = "abuse"            # abusive / threatening language -> guardrail refusal
    OUT_OF_SCOPE = "out_of_scope"  # not something support handles -> polite decline


# Categories that should NOT proceed to retrieval + drafting + sending.
# They are handled by the guardrail (policy_response) node instead.
BLOCKED_CATEGORIES = {Category.ABUSE, Category.OUT_OF_SCOPE}


class Triage(BaseModel):
    """Structured output of the Triage/Router agent (Agent 1)."""

    category: Category = Field(description="The single best category for this ticket.")
    reason: str = Field(description="One short sentence explaining the classification.")
    order_id: str | None = Field(
        default=None,
        description="The order ID mentioned by the customer, if any (e.g. 'ORD-1001').",
    )
    sentiment: str = Field(
        default="neutral",
        description="Customer sentiment: 'angry', 'frustrated', 'neutral', or 'happy'.",
    )
