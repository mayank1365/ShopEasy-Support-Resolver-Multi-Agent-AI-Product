# Individual Contributions — ShopEasy Support Resolver

This document outlines the specific contributions of each team member to the **ShopEasy Support Resolver** project. 

---

## 👤 Mayank Gupta
**Focus:** Triage Agent & LangGraph Orchestration (Orchestration & Workflow Wiring)

*   **LangGraph Orchestration (`src/graph.py`):** Structured and implemented the `StateGraph` which wires all nodes (agents, guardrails, review, and send) together. Integrated two conditional routers (`route_after_triage` and `route_after_review`) and set up the checkpointer to support state persistence for human-in-the-loop pauses.
*   **Triage Agent (`src/agents.py`):** Built the `triage_agent` which classifies incoming customer tickets into one of the designated categories (`billing`, `technical`, `refund`, `abuse`, or `out_of_scope`) and extracts order IDs and sentiments.
*   **Structured State & Schemas (`src/state.py`, `src/schemas.py`):** Designed the shared `SupportState` (a `TypedDict` passed between all nodes) and implemented the Pydantic models for strict data validation (such as enforcing classification enum schemas).

---

## 👤 Kumar Kartikay
**Focus:** Retriever Agent, RAG Knowledge Base, Evaluation Suite, & Observability

*   **RAG Knowledge Base & Retriever (`src/rag.py`, `src/knowledge_base/`):** Built the RAG system that loads and splits help documents (`refunds.md`, `shipping.md`, `billing.md`, `account_technical.md`) into sections, indexes them using TF-IDF, and retrieves top matching snippets using cosine similarity.
*   **Retriever Agent (`src/agents.py`):** Implemented the `retriever_agent` node that queries the RAG index and attaches relevant policy documentation snippets and sources to the state.
*   **Evaluation Suite (`evals.py`):** Developed a 5-case evaluation test suite covering each branch of the graph to programmatically verify category classification and final ticket status.
*   **Observability & Tracing (`src/llm.py`):** Configured LangSmith tracing to wrap the Claude client (tracing every agent and LLM call as spans) and set up the per-node trace logs for debugging.

---

## 👤 Abhishek Kumar Shah
**Focus:** Drafter Agent, Tools, Guardrails, Human-in-the-Loop, & Streamlit Demo

*   **Drafter Agent (`src/agents.py`):** Built the `drafter_agent` which writes the customer response grounded in retrieved policy, and runs an agentic tool-use loop to lookup order information when referenced.
*   **Tools & Mock Data (`src/tools.py`):** Implemented the `order_lookup` and `send_reply` tools, along with their JSON schemas and a 7-order mock database simulating different refund rules (e.g. within/out-of window, final sale, value over $200).
*   **Guardrails & HITL Nodes (`src/agents.py`):** Developed the `policy_response` node to act as a safety guardrail (refusing abuse and declining out-of-scope queries) and the `human_review` node using LangGraph's `interrupt()` to pause execution for human verification.
*   **Streamlit Demo UI (`app.py`):** Developed the interactive Streamlit user interface that demonstrates the entire pipeline, including step-by-step trace logs, retrieved policy files, and the Approve / Edit / Reject interactive review panel.
