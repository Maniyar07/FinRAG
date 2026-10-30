"""Shared classification for upstream, request, and internal failures."""

from __future__ import annotations

import ssl
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from src.request_runtime import RequestAborted


@dataclass(frozen=True)
class RuntimeErrorInfo:
    code: str
    retryable: bool
    detail: str | None = None


def classify_runtime_error(error: Exception) -> RuntimeErrorInfo:
    """Give every runtime failure one stable code and retry policy."""
    if isinstance(error, RequestAborted):
        return RuntimeErrorInfo("request_aborted", False, str(error)[:240])
    if isinstance(error, RateLimitError):
        body = error.body if isinstance(error.body, dict) else {}
        # The SDK normally unwraps the outer "error" object before raising.
        details = body.get("error", body)
        if isinstance(details, dict) and (
            details.get("type") == "insufficient_quota"
            or details.get("code") in {"insufficient_quota", "credit_balance_exhausted"}
        ):
            return RuntimeErrorInfo("provider_quota_exhausted", False)
        return RuntimeErrorInfo("provider_rate_limited", True)
    if isinstance(error, APITimeoutError):
        return RuntimeErrorInfo("provider_timeout", True)
    if isinstance(error, APIConnectionError):
        current: BaseException | None = error
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, RequestAborted):
                return RuntimeErrorInfo("request_aborted", False, str(current)[:240])
            if (
                isinstance(current, ssl.SSLCertVerificationError)
                or "CERTIFICATE_VERIFY_FAILED" in str(current)
            ):
                return RuntimeErrorInfo("provider_tls_error", False)
            current = current.__cause__ or current.__context__
        return RuntimeErrorInfo("provider_unavailable", True)
    if isinstance(error, APIStatusError):
        retryable = error.status_code >= 500
        body = error.body if isinstance(error.body, dict) else {}
        body = body.get("error", body)
        detail = f"HTTP {error.status_code}"
        if isinstance(body, dict):
            detail += "".join(f"; {key}={str(body[key])[:80]}" for key in ("code", "param") if body.get(key))
        return RuntimeErrorInfo(
            "provider_server_error" if retryable else "provider_request_rejected",
            retryable,
            detail,
        )
    if isinstance(error, ValueError):
        detail = " ".join(str(error).split())[:240]
        return RuntimeErrorInfo("request_rejected", False, detail or None)
    return RuntimeErrorInfo("internal_error", False)


def runtime_error_message(info: RuntimeErrorInfo) -> str:
    """Return an actionable message without exposing raw provider exceptions."""
    return {
        "request_aborted": "The request stopped because it was cancelled or reached a configured execution limit.",
        "provider_tls_error": (
            "FinRAG could not verify the model provider's HTTPS certificate. "
            "Check the trusted certificates for your network or configure SSL_CERT_FILE, "
            "then restart FinRAG."
        ),
        "provider_unavailable": (
            "FinRAG could not connect to the model provider. "
            "Check your network, proxy, firewall, and provider endpoint, then try again."
        ),
        "provider_timeout": "The model provider timed out. Please try again shortly.",
        "provider_rate_limited": (
            "The model provider is rate limiting requests. Please wait briefly and try again."
        ),
        "provider_quota_exhausted": (
            "The model provider reports exhausted API credits or quota. "
            "Check billing and usage limits for the configured API key before trying again."
        ),
        "provider_server_error": (
            "The model provider encountered a server error. Please try again shortly."
        ),
        "provider_request_rejected": (
            "The model provider rejected this request. Check the configured API key, "
            "model access, and provider settings."
        ),
    }.get(
        info.code,
        "The financial research agent could not complete this request. Please try again.",
    )


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
