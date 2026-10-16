"""Shared classification for upstream, request, and internal failures."""

from __future__ import annotations

from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError


@dataclass(frozen=True)
class RuntimeErrorInfo:
    code: str
    retryable: bool
    detail: str | None = None


def classify_runtime_error(error: Exception) -> RuntimeErrorInfo:
    """Give every runtime failure one stable code and retry policy."""
    if isinstance(error, RateLimitError):
        body = error.body if isinstance(error.body, dict) else {}
        details = body.get("error")
        if isinstance(details, dict) and (
            details.get("type") == "insufficient_quota"
            or details.get("code") in {"insufficient_quota", "credit_balance_exhausted"}
        ):
            return RuntimeErrorInfo("provider_quota_exhausted", False)
        return RuntimeErrorInfo("provider_rate_limited", True)
    if isinstance(error, APITimeoutError):
        return RuntimeErrorInfo("provider_timeout", True)
    if isinstance(error, APIConnectionError):
        return RuntimeErrorInfo("provider_unavailable", True)
    if isinstance(error, APIStatusError):
        retryable = error.status_code >= 500
        return RuntimeErrorInfo(
            "provider_server_error" if retryable else "provider_request_rejected",
            retryable,
        )
    if isinstance(error, ValueError):
        detail = " ".join(str(error).split())[:240]
        return RuntimeErrorInfo("request_rejected", False, detail or None)
    return RuntimeErrorInfo("internal_error", False)


def tool_error_response(operation: str, error: Exception) -> dict[str, object]:
    """Build the compact, consistent failure payload consumed by tools and evals."""
    info = classify_runtime_error(error)
    response: dict[str, object] = {
        "error": f"{operation} failed ({info.code}).",
        "error_code": info.code,
        "error_type": type(error).__name__,
        "retryable": info.retryable,
    }
    if info.detail:
        response["error_detail"] = info.detail
    return response
