"""LangGraph wiring: nodes, edges, conditional routing, and the HITL checkpointer.

Graph shape
-----------
                          ┌─ abuse / out_of_scope ─► policy_response ─► END
   START ─► triage ─►(route)
                          └─ billing/technical/refund ─► retrieve ─► draft ─► human_review
                                                                                   │
                                                          (route) ┌─ approve ─► send ─► END
                                                                  └─ reject ──────────► END

- `triage` is the single conditional branch point required by the brief.
- `human_review` calls interrupt(), so the graph PAUSES there; a MemorySaver
  checkpointer persists state across the pause/resume (needed for HITL).
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents import (
    drafter_agent,
    human_review,
    policy_response,
    retriever_agent,
    send_node,
    triage_agent,
)
from .schemas import BLOCKED_CATEGORIES, Category
from .state import SupportState


def route_after_triage(state: SupportState) -> str:
    """Conditional edge: blocked categories skip the pipeline; others proceed."""
    if state["category"] in {c.value for c in BLOCKED_CATEGORIES}:
        return "policy_response"
    return "retrieve"


def route_after_review(state: SupportState) -> str:
    """Conditional edge: send only if the human approved."""
    if state.get("approval_decision") == "approve":
        return "send"
    return END


def build_graph(checkpointer: MemorySaver | None = None):
    """Construct and compile the support graph.

    A checkpointer is REQUIRED for the interrupt()/resume HITL flow to work.
    Callers that don't pass one get a fresh MemorySaver.
    """
    graph = StateGraph(SupportState)

    # Register nodes.
    graph.add_node("triage", triage_agent)
    graph.add_node("retrieve", retriever_agent)
    graph.add_node("draft", drafter_agent)
    graph.add_node("policy_response", policy_response)
    graph.add_node("human_review", human_review)
    graph.add_node("send", send_node)

    # Edges.
    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {"policy_response": "policy_response", "retrieve": "retrieve"},
    )
    graph.add_edge("policy_response", END)
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "human_review")
    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {"send": "send", END: END},
    )
    graph.add_edge("send", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
