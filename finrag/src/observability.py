"""LangSmith spans and bounded serialization for FinRAG components.

LangChain and LangGraph trace their model and tool runs when the LangSmith
environment variables are enabled. This module adds named spans around the
FinRAG Python stages between those calls and summarizes their payloads so large
document bodies are not duplicated by these custom spans.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import inspect
from functools import wraps
from collections.abc import Mapping
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from src.config import LANGSMITH_PROJECT
from src.telemetry import usage_stage
from src.request_runtime import check_request

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree


_SECRET_KEY_RE = re.compile(
    r"(?:api.?key|secret|password|credential|authorization|"
    r"(?:access|refresh).?token|bearer)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|lsv2_[A-Za-z0-9_-]{16,}|co_[A-Za-z0-9_-]{16,})\b"
)
_CONTENT_KEYS = frozenset(
    {
        "answer",
        "body",
        "candidate_answer",
        "content",
        "context",
        "contexts",
        "document_text",
        "evidence_excerpt",
        "evidence_text",
        "full_evidence_text",
        "generated_answer",
        "ground_truth",
        "page_content",
        "prompt",
        "reference",
        "reference_answer",
        "response",
        "text",
    }
)
_PREVIEW_KEYS = frozenset({"query", "question"})
_MAX_DEPTH = 6
_MAX_FIELDS = 40
_MAX_ITEMS = 16
_MAX_PREVIEW_CHARS = 240


def _safe_string(value: str, *, preview: bool = False) -> dict[str, Any]:
    cleaned = _SECRET_VALUE_RE.sub("[REDACTED]", value)
    result: dict[str, Any] = {"chars": len(cleaned)}
    if preview and cleaned:
        result["preview"] = cleaned[:_MAX_PREVIEW_CHARS]
    return result


def summarize_trace_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """Keep useful metadata while excluding long prompts and evidence bodies."""
    if _SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if depth > _MAX_DEPTH:
        return {"type": type(value).__name__}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return {"file_name": value.name}
    if isinstance(value, str):
        if key.casefold() in _CONTENT_KEYS:
            return _safe_string(value)
        if key.casefold() in _PREVIEW_KEYS:
            return _safe_string(value, preview=True)
        cleaned = _SECRET_VALUE_RE.sub("[REDACTED]", value)
        return cleaned[:_MAX_PREVIEW_CHARS]

    # LangChain Documents are summarized before their `page_content` can be
    # serialized as part of the object.
    if hasattr(value, "page_content") and hasattr(value, "metadata"):
        return {
            "type": "Document",
            "content_chars": len(str(getattr(value, "page_content", ""))),
            "metadata": summarize_trace_value(
                getattr(value, "metadata", {}), key="metadata", depth=depth + 1
            ),
        }

    if isinstance(value, Mapping):
        items = list(value.items())
        summarized = {
            str(item_key): summarize_trace_value(
                item_value, key=str(item_key), depth=depth + 1
            )
            for item_key, item_value in items[:_MAX_FIELDS]
        }
        if len(items) > _MAX_FIELDS:
            summarized["_omitted_fields"] = len(items) - _MAX_FIELDS
        return summarized

    if isinstance(value, argparse.Namespace):
        return summarize_trace_value(vars(value), key=key, depth=depth + 1)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return summarize_trace_value(
                dataclasses.asdict(value), key=key, depth=depth + 1
            )
        except Exception:
            return {"type": type(value).__name__}

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return summarize_trace_value(
                model_dump(mode="json"), key=key, depth=depth + 1
            )
        except Exception:
            return {"type": type(value).__name__}

    # Scope keys and other short records are much easier to read in LangSmith
    # as ordinary arrays than as nested {count, items} summaries.
    if isinstance(value, tuple) and len(value) <= 8 and all(
        item is None or isinstance(item, (str, bool, int, float, Enum, Decimal))
        for item in value
    ):
        return [
            summarize_trace_value(item, key=key, depth=depth + 1)
            for item in value
        ]

    if isinstance(value, (list, tuple, set, frozenset)):
        values = (
            sorted(value, key=str)
            if isinstance(value, (set, frozenset))
            else list(value)
        )
        result = {
            "count": len(values),
            "items": [
                summarize_trace_value(item, key=key, depth=depth + 1)
                for item in values[:_MAX_ITEMS]
            ],
        }
        if len(values) > _MAX_ITEMS:
            result["omitted_items"] = len(values) - _MAX_ITEMS
        return result

    return {"type": type(value).__name__}


def summarize_trace_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Processor used by LangSmith decorators for function arguments."""
    try:
        return summarize_trace_value(inputs, key="inputs")
    except Exception:
        return {"serialization": "failed", "type": type(inputs).__name__}


def summarize_trace_outputs(output: Any) -> Any:
    """Processor used by LangSmith decorators for component results."""
    try:
        return summarize_trace_value(output, key="output")
    except Exception:
        return {"serialization": "failed", "type": type(output).__name__}


def trace_component(
    name: str,
    *,
    run_type: str = "chain",
    root: bool = False,
) -> Callable:
    """Decorate a FinRAG stage with a named, bounded LangSmith span."""
    options: dict[str, Any] = {
        "name": name,
        "run_type": run_type,
        "tags": ["finrag", f"component:{name}"],
        "metadata": {"application": "finrag", "component": name},
        "process_inputs": summarize_trace_inputs,
        "process_outputs": summarize_trace_outputs,
    }
    if root:
        options["project_name"] = LANGSMITH_PROJECT
    def decorate(function: Callable) -> Callable:
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def measured(*args, **kwargs):
                check_request()
                with usage_stage(name):
                    result = await function(*args, **kwargs)
                check_request()
                return result
        else:
            @wraps(function)
            def measured(*args, **kwargs):
                check_request()
                with usage_stage(name):
                    result = function(*args, **kwargs)
                check_request()
                return result
        return traceable(**options)(measured)
    return decorate


def annotate_current_trace(
    metadata: Mapping[str, Any], *, tags: tuple[str, ...] = ()
) -> None:
    """Add searchable request facts to the active run without affecting work."""
    try:
        run = get_current_run_tree()
        if run is None:
            return
        run.add_metadata(
            {
                str(key): summarize_trace_value(value, key=str(key))
                for key, value in metadata.items()
            }
        )
        if tags:
            run.add_tags(tags)
    except Exception:
        # Observability must never make a financial request fail.
        return
