"""The shared graph state.

Every node receives this dict, reads what it needs, and returns a partial update.
LangGraph merges the returned partial back into the running state. This single
object is how information flows (and is reused) across the whole system — the
rubric's "state, memory, and context design".
"""

from __future__ import annotations

from typing import TypedDict


class SupportState(TypedDict, total=False):
    # --- input ---
    ticket: str                 # the raw customer message

    # --- set by Triage (Agent 1) ---
    category: str               # one of schemas.Category values
    triage_reason: str          # why it was classified that way
    order_id: str | None        # extracted order id, if any
    sentiment: str              # angry / frustrated / neutral / happy

    # --- set by Retriever (Agent 2, RAG) ---
    retrieved_docs: list[str]   # the help-doc snippets used for grounding
    retrieved_sources: list[str]  # which files those snippets came from

    # --- set by Drafter (Agent 3) ---
    draft_reply: str            # the proposed customer reply
    tools_used: list[str]       # e.g. ["order_lookup"] — for observability

    # --- human-in-the-loop ---
    approval_decision: str      # "approve" | "reject" (set by the human)
    final_reply: str            # the approved (possibly edited) reply

    # --- terminal outcome ---
    status: str                 # "sent" | "refused" | "rejected" | "declined"
    send_result: str            # confirmation string from the send_reply tool

    # --- observability ---
    log: list[str]              # human-readable trace of what each node did
