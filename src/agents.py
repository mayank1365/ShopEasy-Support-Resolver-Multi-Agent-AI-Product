"""The agent node functions used by the LangGraph graph.

Each function is a graph node: it takes the shared SupportState and returns a
partial update. The three meaningful agents are:

    triage_agent    (Agent 1) — classify + route + extract order id  [structured output]
    retriever_agent (Agent 2) — RAG over help docs for grounding
    drafter_agent   (Agent 3) — write the reply, calling order_lookup when needed

Plus two control nodes that are not "agents" but complete the product workflow:
    policy_response — guardrail: safe refusal/decline for abuse & out-of-scope
    send_node       — executes the high-impact send AFTER human approval
"""

from __future__ import annotations

import json

from langgraph.types import interrupt

from .llm import MODEL, get_client
from .rag import retrieve
from .schemas import Category, Triage
from .state import SupportState
from .tools import ORDER_LOOKUP_TOOL, order_lookup, send_reply


def _append_log(state: SupportState, message: str) -> list[str]:
    """Return the running log with one line appended (observability)."""
    return [*state.get("log", []), message]


# ---------------------------------------------------------------------------
# Agent 1 — Triage / Router
# ---------------------------------------------------------------------------
# Forced tool-use is used to GUARANTEE structured output: Claude must call the
# `classify` tool, whose schema mirrors the Triage pydantic model. We then
# validate the arguments with pydantic before trusting them.
_CLASSIFY_TOOL = {
    "name": "classify",
    "description": "Record the classification of a customer support ticket.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [c.value for c in Category],
                "description": (
                    "billing = payments/charges/invoices; technical = login/app/website/order-changes; "
                    "refund = wants money back or to return an item; "
                    "abuse = threats, harassment, or abusive language; "
                    "out_of_scope = unrelated to ShopEasy support (jokes, general questions, other companies)."
                ),
            },
            "reason": {"type": "string", "description": "One short sentence justifying the category."},
            "order_id": {
                "type": ["string", "null"],
                "description": "The order ID the customer mentions, e.g. 'ORD-1001', else null.",
            },
            "sentiment": {
                "type": "string",
                "enum": ["angry", "frustrated", "neutral", "happy"],
            },
        },
        "required": ["category", "reason", "sentiment"],
    },
}

_TRIAGE_SYSTEM = (
    "You are the expert Triage Agent for ShopEasy customer support.\n"
    "Your role is to analyze the customer's support ticket, determine the single most appropriate category, "
    "extract the order ID if mentioned, and identify customer sentiment. You must call the `classify` tool to submit your analysis.\n\n"
    "CRITICAL CATEGORY GUIDELINES:\n"
    "1. billing: Focuses on payments, charges, duplicate billing, pending holds, invoices, receipts, subscription billing (ShopEasy Plus), and coupons/promo codes.\n"
    "2. technical: Focuses on login issues, password resets, account locking, app/website bugs, page load/checkout errors, and changes to active orders (address changes or cancellation requests before shipping).\n"
    "3. refund: Customer explicitly wants a refund, return, reimbursement, or their money back for an order (e.g., damaged/broken item, unwanted item, return shipping fee requests).\n"
    "4. abuse: Strictly reserved for genuine threats, harassment, abusive/offensive/hateful language, or personal attacks. Do NOT classify mere frustration or angry complaints about a service issue as abuse.\n"
    "5. out_of_scope: Entirely unrelated to ShopEasy support, products, or services (e.g., general knowledge questions, jokes, questions about other companies, weather, recommendations).\n\n"
    "ORDER ID EXTRACTION:\n"
    "- Search for patterns matching 'ORD-' followed by digits (e.g., 'ORD-1001').\n"
    "- If a valid order ID is found, extract it exactly.\n"
    "- If no order ID is found or it is highly ambiguous, set it to null.\n\n"
    "SENTIMENT ANALYSIS:\n"
    "- 'angry': Hostile, aggressive, using capital letters/exclamation marks to vent, demanding immediate action.\n"
    "- 'frustrated': Impatient, annoyed by bugs, delays, or repetitive steps, but not hostile/abusive.\n"
    "- 'neutral': Standard, matter-of-fact queries, clear descriptions of issues without emotional tone.\n"
    "- 'happy': Expressing gratitude, satisfaction, or positive feedback.\n\n"
    "Ensure you prioritize accuracy and strictly follow the tool schema parameters."
)


def triage_agent(state: SupportState) -> SupportState:
    client = get_client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=_TRIAGE_SYSTEM,
        tools=[_CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "classify"},
        messages=[{"role": "user", "content": state["ticket"]}],
    )
    tool_block = next(b for b in resp.content if b.type == "tool_use")
    triage = Triage.model_validate(tool_block.input)  # pydantic validation of the handoff

    return {
        "category": triage.category.value,
        "triage_reason": triage.reason,
        "order_id": triage.order_id,
        "sentiment": triage.sentiment,
        "log": _append_log(
            state,
            f"[Triage] category={triage.category.value} | sentiment={triage.sentiment} "
            f"| order_id={triage.order_id} | {triage.reason}",
        ),
    }


# ---------------------------------------------------------------------------
# Agent 2 — Knowledge Retriever (RAG)
# ---------------------------------------------------------------------------
def retriever_agent(state: SupportState) -> SupportState:
    # Retrieve against the ticket text plus the category for sharper matching.
    query = f"{state['category']} {state['ticket']}"
    hits = retrieve(query, k=3)

    docs = [chunk for _, chunk in hits]
    sources = sorted({src for src, _ in hits})

    if docs:
        log_line = f"[Retriever] grounded on {len(docs)} snippet(s) from: {', '.join(sources)}"
    else:
        log_line = "[Retriever] no relevant policy found — drafter will answer cautiously"

    return {
        "retrieved_docs": docs,
        "retrieved_sources": sources,
        "log": _append_log(state, log_line),
    }


# ---------------------------------------------------------------------------
# Agent 3 — Reply Drafter (with order_lookup tool-use)
# ---------------------------------------------------------------------------
_DRAFTER_SYSTEM = (
    "You are a professional, empathetic, and concise customer-support agent for ShopEasy.\n"
    "Your objective is to write a high-quality, helpful response to the customer using ONLY the provided Policy Grounding.\n\n"
    "CORE INSTRUCTIONS:\n"
    "1. Grounding constraint: Do NOT invent or assume any policies, timelines, refund amounts, or procedures. If the provided Policy Grounding does not cover the customer's query, politely inform them that you will escalate their ticket to a human teammate.\n"
    "2. Order verification: If the customer mentions an order or if their issue is order-specific, you MUST call the `order_lookup` tool to retrieve real order facts before writing the final reply. Do not guess the order status, items, or refund eligibility.\n"
    "3. Sentiment adjustment:\n"
    "   - If sentiment is 'angry' or 'frustrated': Be exceptionally empathetic, acknowledge their frustration, and avoid generic boilerplate language.\n"
    "   - If sentiment is 'happy' or 'neutral': Be professional, warm, and clear.\n"
    "4. Specific Policy Rules (apply ONLY when confirmed via grounding and tool results):\n"
    "   - Refund Window: Within 30 days of delivery. Beyond 30 days is ineligible (suggest escalation for goodwill review). Final-sale/clearance items are never eligible for refunds/returns.\n"
    "   - Damaged/Wrong Item: Full refund or free replacement, including refunding shipping and free prepaid return label.\n"
    "   - Refund Threshold: Refunds ABOVE $200 require manager approval. Inform the customer they are eligible and that you are submitting it for quick manager sign-off.\n"
    "   - Delivery/Late packages: Ask to check neighbors/reception if marked delivered but not received; wait 48 hours before filing claims. If package is >3 business days late, offer carrier investigation.\n"
    "   - Billing holds: Double charges are usually pending holds that drop off in 3-5 business days. Ask them to verify if pending before escalating.\n"
    "5. Formatting Constraints:\n"
    "   - Keep your final reply clear, structured, and under 120 words.\n"
    "   - Do NOT output any internal thinking or scratchpad in your final text response. Just output the final message to the customer.\n"
    "   - Always sign off as 'ShopEasy Support'."
)


def drafter_agent(state: SupportState) -> SupportState:
    client = get_client()
    grounding = "\n\n---\n\n".join(state.get("retrieved_docs", [])) or "(no policy snippets retrieved)"
    user_msg = (
        f"Customer ticket:\n{state['ticket']}\n\n"
        f"Customer Sentiment: {state.get('sentiment', 'neutral')}\n\n"
        f"Order ID (if any): {state.get('order_id') or 'None'}\n\n"
        f"Policy grounding:\n{grounding}\n\n"
        "Write the customer reply now."
    )

    messages = [{"role": "user", "content": user_msg}]
    tools_used: list[str] = []

    # Tool-use loop: let Claude call order_lookup as many times as it needs.
    for _ in range(4):  # safety bound on iterations
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=_DRAFTER_SYSTEM,
            tools=[ORDER_LOOKUP_TOOL],
            messages=messages,
        )

        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            return {
                "draft_reply": text,
                "tools_used": tools_used,
                "log": _append_log(
                    state,
                    f"[Drafter] wrote draft ({len(text)} chars)"
                    + (f", tools used: {tools_used}" if tools_used else ", no tools needed"),
                ),
            }

        # Execute every requested tool call and feed the results back.
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if block.type == "tool_use" and block.name == "order_lookup":
                tools_used.append("order_lookup")
                result = order_lookup(**block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    # Fell through the loop bound — return whatever text we can.
    return {
        "draft_reply": "I'm escalating this to a human teammate who will follow up shortly.\n\nShopEasy Support",
        "tools_used": tools_used,
        "log": _append_log(state, "[Drafter] hit tool-loop bound; escalating to human"),
    }


# ---------------------------------------------------------------------------
# Guardrail node — abuse & out-of-scope handling (no send)
# ---------------------------------------------------------------------------
def policy_response(state: SupportState) -> SupportState:
    """Safe, fixed handling for tickets that must NOT enter the normal pipeline."""
    category = state["category"]
    if category == Category.ABUSE.value:
        reply = (
            "We want to help, but we can't continue while the conversation includes "
            "abusive language. If you'd like to rephrase your concern, a support "
            "teammate will be glad to assist.\n\nShopEasy Support"
        )
        status = "refused"
        log_line = "[Guardrail] abusive ticket -> refusal, not sent automatically"
    else:  # OUT_OF_SCOPE
        reply = (
            "Thanks for reaching out! This doesn't look like something ShopEasy "
            "support can help with, but if you have a question about an order, "
            "billing, or your account, we're happy to help.\n\nShopEasy Support"
        )
        status = "declined"
        log_line = "[Guardrail] out-of-scope ticket -> polite decline"

    return {
        "draft_reply": reply,
        "final_reply": reply,
        "status": status,
        "log": _append_log(state, log_line),
    }


# ---------------------------------------------------------------------------
# Human-in-the-loop gate
# ---------------------------------------------------------------------------
def human_review(state: SupportState) -> SupportState:
    """Pause the graph and hand the draft to a human for Approve / Edit / Reject.

    `interrupt(...)` suspends execution and surfaces the payload to the caller
    (the Streamlit app or the eval harness). Execution resumes when the caller
    sends a Command(resume=<decision dict>) back into the graph.
    """
    decision = interrupt(
        {
            "kind": "approval_request",
            "category": state["category"],
            "draft_reply": state["draft_reply"],
            "order_id": state.get("order_id"),
        }
    )
    # `decision` is whatever the human supplied on resume.
    choice = decision.get("decision", "reject")
    edited = decision.get("edited_reply") or state["draft_reply"]

    if choice == "approve":
        return {
            "approval_decision": "approve",
            "final_reply": edited,
            "log": _append_log(state, "[Human] approved the reply" + (" (edited)" if edited != state["draft_reply"] else "")),
        }
    return {
        "approval_decision": "reject",
        "status": "rejected",
        "log": _append_log(state, "[Human] rejected the reply — nothing sent"),
    }


# ---------------------------------------------------------------------------
# Send node — the high-impact action, runs only after approval
# ---------------------------------------------------------------------------
def send_node(state: SupportState) -> SupportState:
    result = send_reply(channel="email", message=state["final_reply"])
    return {
        "status": "sent",
        "send_result": result,
        "log": _append_log(state, f"[Send] {result}"),
    }
