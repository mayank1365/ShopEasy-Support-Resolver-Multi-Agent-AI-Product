"""Streamlit demo UI for the Multi-Agent E-commerce Support Resolver.

Run:  streamlit run app.py

The UI makes the graph's work *visible*:
  1. You enter (or pick) a customer ticket and click Resolve.
  2. A pipeline stepper shows which agents ran (Triage → Retrieve → Draft → Review → Send).
  3. The retrieved policy snippets and the full agent trace are shown.
  4. For tickets that produce an outbound reply, the graph PAUSES for human
     approval (Approve / Edit / Reject) before anything is "sent".
  5. Abusive / out-of-scope tickets are handled by the guardrail and never sent.
"""

from __future__ import annotations

import streamlit as st
from langgraph.types import Command

from src.graph import build_graph
from src.llm import MODEL, tracing_enabled

st.set_page_config(page_title="ShopEasy Support Resolver", page_icon="🛍️", layout="wide")

EXAMPLES = {
    "Billing — double charge": "I think I was charged twice for my order, can you help?",
    "Technical — can't log in": "I can't log into my account, the password reset email never arrives.",
    "Refund — broken item, in window (ORD-1001)": "I want a refund for order ORD-1001, it arrived broken.",
    "Refund — high value > $200 (ORD-1005)": "Please refund order ORD-1005, the 4K monitor — I changed my mind.",
    "Refund — outside 30 days (ORD-1007)": "I'd like to return ORD-1007, the office chair I bought a couple months ago.",
    "Refund — clearance / final sale (ORD-1003)": "I'd like to return ORD-1003, the clearance USB-C cable.",
    "Refund — not delivered yet (ORD-1006)": "Cancel and refund ORD-1006, the laptop stand — it hasn't arrived.",
    "Abuse — must refuse": "You people are idiots, give me my money back you morons!",
    "Out of scope — decline": "What's the weather in Mumbai and can you recommend a pizza place?",
}

CONFIG = {"configurable": {"thread_id": "streamlit-session"}}


def _reset() -> None:
    for key in ("phase", "interrupt", "state", "app"):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Sidebar — what this is + how it works (so the screen is never empty)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🛍️ ShopEasy Support Resolver")
    st.caption(f"Multi-agent · LangGraph · Claude `{MODEL}`")

    st.markdown(
        "**Pipeline**\n\n"
        "1. **Triage** — classify & route\n"
        "2. **Retrieve** — RAG over help-docs\n"
        "3. **Draft** — write reply (+ `order_lookup`)\n"
        "4. **Human review** — approve / edit / reject\n"
        "5. **Send** — only after approval\n\n"
        "Abuse & out-of-scope → **guardrail** (never sent)."
    )

    if tracing_enabled():
        st.success("🔭 LangSmith tracing: ON")
    else:
        st.info("🔭 LangSmith tracing: off\n\n(add LANGSMITH_API_KEY + "
                "LANGSMITH_TRACING=true to .env)")

    with st.expander("Graph structure"):
        try:
            mermaid = build_graph().get_graph().draw_mermaid()
            st.code(mermaid, language="mermaid")
        except Exception:
            st.text("triage → (route) → retrieve → draft → human_review → send\n"
                    "         ↘ policy_response (guardrail)")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Customer support, resolved by a team of AI agents")
st.caption("Triage → Retrieve (RAG) → Draft (tool use) → **Human approval** → Send")

st.session_state.setdefault("phase", "idle")


# ---------------------------------------------------------------------------
# Pipeline stepper — makes the agents' work visible
# ---------------------------------------------------------------------------
def show_pipeline(state: dict) -> None:
    category = state.get("category")
    status = state.get("status")
    blocked = category in ("abuse", "out_of_scope")

    if blocked:
        stages = [("Triage", True), ("Guardrail", True)]
    else:
        stages = [
            ("Triage", bool(category)),
            ("Retrieve", bool(state.get("retrieved_docs") is not None)),
            ("Draft", bool(state.get("draft_reply"))),
            ("Review", status in ("sent", "rejected")),
            ("Send", status == "sent"),
        ]

    cols = st.columns(len(stages))
    for col, (name, done) in zip(cols, stages):
        col.markdown(f"### {'✅' if done else '⏳'}")
        col.caption(name)


def show_details(state: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Category", state.get("category", "—"))
    c2.metric("Sentiment", state.get("sentiment", "—"))
    c3.metric("Tools used", ", ".join(state.get("tools_used") or []) or "—")

    if state.get("retrieved_docs"):
        with st.expander(f"📚 Retrieved policy ({len(state['retrieved_docs'])} snippet(s) "
                         f"from {', '.join(state.get('retrieved_sources', []))})"):
            for doc in state["retrieved_docs"]:
                st.markdown(doc)
                st.divider()

    with st.expander("🔍 Full agent trace (observability)", expanded=True):
        for line in state.get("log", []):
            st.text(line)


# ---------------------------------------------------------------------------
# Phase: input
# ---------------------------------------------------------------------------
if st.session_state["phase"] == "idle":
    st.subheader("1 · Enter a customer ticket")
    choice = st.selectbox("Pick an example, or write your own", ["(write my own)"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, "")
    ticket = st.text_area("Customer ticket", value=default, height=120,
                          placeholder="e.g. I want a refund for order ORD-1001, it arrived broken.")

    if st.button("▶ Resolve ticket", type="primary", disabled=not ticket.strip()):
        app = build_graph()
        st.session_state["app"] = app
        with st.spinner("Agents working: triage → retrieve → draft ..."):
            result = app.invoke({"ticket": ticket.strip(), "log": []}, CONFIG)
        if "__interrupt__" in result:
            st.session_state["interrupt"] = result["__interrupt__"][0].value
            st.session_state["state"] = result
            st.session_state["phase"] = "awaiting_approval"
        else:
            st.session_state["state"] = result
            st.session_state["phase"] = "done"
        st.rerun()
    else:
        st.info("👈 Pick an example on the left of the dropdown or type a ticket, then click **Resolve**.")


# ---------------------------------------------------------------------------
# Phase: approval gate (human-in-the-loop)
# ---------------------------------------------------------------------------
if st.session_state["phase"] == "awaiting_approval":
    st.subheader("2 · Agents ran — here's what happened")
    show_pipeline(st.session_state["state"])
    show_details(st.session_state["state"])

    payload = st.session_state["interrupt"]
    st.subheader("3 · ✋ Human approval required before sending")
    if payload.get("category") == "refund":
        st.warning("This is a **refund** reply — review carefully before it goes out.")

    edited = st.text_area("Proposed reply (edit if needed)", value=payload["draft_reply"], height=240)

    c1, c2, c3 = st.columns(3)
    if c1.button("✅ Approve & send", type="primary"):
        result = st.session_state["app"].invoke(
            Command(resume={"decision": "approve", "edited_reply": edited}), CONFIG)
        st.session_state["state"] = result
        st.session_state["phase"] = "done"
        st.rerun()
    if c2.button("❌ Reject"):
        result = st.session_state["app"].invoke(Command(resume={"decision": "reject"}), CONFIG)
        st.session_state["state"] = result
        st.session_state["phase"] = "done"
        st.rerun()
    if c3.button("↩︎ Start over"):
        _reset()
        st.rerun()


# ---------------------------------------------------------------------------
# Phase: final outcome
# ---------------------------------------------------------------------------
if st.session_state["phase"] == "done":
    st.subheader("Result")
    state = st.session_state["state"]
    show_pipeline(state)

    status = state.get("status", "—")
    banner = {
        "sent": ("✅ Reply sent to the customer", st.success),
        "refused": ("🛡️ Refused by guardrail (abusive ticket) — nothing sent", st.error),
        "declined": ("↪︎ Politely declined (out of scope) — nothing sent", st.info),
        "rejected": ("❌ Rejected by reviewer — nothing sent", st.warning),
    }.get(status, (f"Status: {status}", st.info))
    banner[1](banner[0])

    if status == "sent":
        st.markdown("#### 📤 Reply sent to customer")
        st.write(state.get("final_reply") or state.get("draft_reply") or "(none)")
        if state.get("send_result"):
            st.caption(state["send_result"])
    elif status == "rejected":
        st.markdown("#### 🗑️ Rejected draft (discarded — NOT sent)")
        st.caption("This is the reply the reviewer rejected. It was never sent to the customer.")
        st.write(state.get("draft_reply") or "(none)")
    else:  # refused / declined — the guardrail's own message
        st.markdown("#### Guardrail response")
        st.write(state.get("final_reply") or state.get("draft_reply") or "(none)")

    show_details(state)

    if st.button("↩︎ Resolve another ticket", type="primary"):
        _reset()
        st.rerun()
