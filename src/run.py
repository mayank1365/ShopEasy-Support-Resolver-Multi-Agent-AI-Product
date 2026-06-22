"""Programmatic driver for the support graph (used by evals and the CLI).

`resolve()` runs a ticket all the way through, automatically supplying a human
decision when the graph pauses at the approval gate. The Streamlit app does the
same dance interactively instead.
"""

from __future__ import annotations

from langgraph.types import Command

from .graph import build_graph
from .state import SupportState


def resolve(
    ticket: str,
    decision: str = "approve",
    edited_reply: str | None = None,
    thread_id: str = "run-1",
) -> SupportState:
    """Run one ticket end-to-end and return the final state.

    decision: what the simulated human does at the approval gate
              ("approve" or "reject"). Ignored for abuse/out-of-scope tickets,
              which never reach the gate.
    """
    app = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    result = app.invoke({"ticket": ticket, "log": []}, config)

    # If the graph paused at human_review, supply the decision and resume.
    if "__interrupt__" in result:
        result = app.invoke(
            Command(resume={"decision": decision, "edited_reply": edited_reply}),
            config,
        )
    return result


def pretty_print(state: SupportState) -> None:
    """Human-readable dump of a resolved ticket (for the CLI / observability)."""
    print("=" * 70)
    print(f"TICKET: {state['ticket']}")
    print("-" * 70)
    print("TRACE:")
    for line in state.get("log", []):
        print("  " + line)
    print("-" * 70)
    print(f"CATEGORY : {state.get('category')}")
    print(f"STATUS   : {state.get('status')}")
    if state.get("retrieved_sources"):
        print(f"SOURCES  : {', '.join(state['retrieved_sources'])}")
    print("-" * 70)
    print("FINAL REPLY:")
    print(state.get("final_reply") or state.get("draft_reply") or "(none)")
    print("=" * 70)
    print()


if __name__ == "__main__":
    import sys

    ticket = " ".join(sys.argv[1:]) or "I want a refund for order ORD-1001, it arrived broken."
    pretty_print(resolve(ticket))
