"""Helpers for Databricks-hosted chat models."""

from __future__ import annotations

import json
from typing import Any

from databricks_langchain import ChatDatabricks
from langchain_core.messages import AIMessage


def chat(endpoint: str, temperature: float = 0, max_tokens: int = 2000) -> ChatDatabricks:
    """Chat model with low reasoning effort (honoured by reasoning models such as gpt-oss; ignored by others)."""
    return ChatDatabricks(endpoint=endpoint, temperature=temperature, max_tokens=max_tokens, extra_params={"reasoning_effort": "low"})


def message_text(msg: AIMessage) -> str:
    """Text content of a reply; gpt-oss returns reasoning + text blocks serialized as JSON in `content`."""
    content: Any = msg.content
    if isinstance(content, str) and content.startswith('[{"type"'):
        content = json.loads(content)
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return content
