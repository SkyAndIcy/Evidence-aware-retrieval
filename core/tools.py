"""OpenAI function-calling tool schemas and response normalization.

Defines the wiki_search / wiki_lookup tools used by ReAct and EARA, and
normalizes OpenAI tool_call responses into a uniform {name, arguments} shape.
"""
from __future__ import annotations
import json


WIKI_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "wiki_search",
        "description": "Search the Wikipedia corpus for passages relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "search query"}},
            "required": ["query"],
        },
    },
}

WIKI_LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "wiki_lookup",
        "description": "Look up a passage by its title.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string", "description": "passage title"}},
            "required": ["title"],
        },
    },
}

ALL_TOOLS = [WIKI_SEARCH_TOOL, WIKI_LOOKUP_TOOL]


def normalize_tool_calls(choice_msg) -> list[dict]:
    """Normalize OpenAI tool_calls to [{name, arguments(dict)}]."""
    out = []
    if not getattr(choice_msg, "tool_calls", None):
        return out
    for tc in choice_msg.tool_calls:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        out.append({"name": tc.function.name, "arguments": args})
    return out
