"""Thin wrapper around the Anthropic Claude client, with optional LangSmith tracing.

Centralises the model id and client construction so every agent uses the same
configuration. We use the official Anthropic SDK directly inside LangGraph nodes.

Observability: if LangSmith is configured (LANGSMITH_TRACING=true + LANGSMITH_API_KEY
in .env), we wrap the Anthropic client with `wrap_anthropic`, which records every
Claude call as an LLM span in LangSmith. LangGraph also auto-traces each node as a
step when those env vars are set. If LangSmith is NOT configured, everything still
runs normally — tracing is simply off.
"""

from __future__ import annotations

import functools
import os

from anthropic import Anthropic
from dotenv import load_dotenv

# Load env (ANTHROPIC_API_KEY, and optionally LANGSMITH_*) from .env once.
load_dotenv()

# Anthropic's most capable model. Do not change without reason.
MODEL = "claude-opus-4-8"


def tracing_enabled() -> bool:
    """True only when LangSmith is both turned on and has an API key."""
    on = os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")
    return on and bool(os.getenv("LANGSMITH_API_KEY"))


@functools.lru_cache(maxsize=1)
def get_client() -> Anthropic:
    """Return a process-wide singleton Anthropic client.

    The API key is read from ANTHROPIC_API_KEY by the SDK automatically. When
    LangSmith tracing is enabled, the client is wrapped so each call is traced.
    """
    client = Anthropic()
    if tracing_enabled():
        try:
            from langsmith.wrappers import wrap_anthropic

            client = wrap_anthropic(client)
        except Exception:
            # Never let an observability dependency break the core product.
            pass
    return client
