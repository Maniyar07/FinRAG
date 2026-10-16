from __future__ import annotations

import unittest

import httpx
from openai import APIConnectionError, RateLimitError

from src.runtime_errors import classify_runtime_error, tool_error_response


class RuntimeErrorTests(unittest.TestCase):
    def test_request_errors_are_not_marked_retryable(self) -> None:
        info = classify_runtime_error(ValueError("  bad   filing year  "))

        self.assertEqual(info.code, "request_rejected")
        self.assertFalse(info.retryable)
        self.assertEqual(info.detail, "bad filing year")

    def test_provider_connection_failure_is_retryable_and_structured(self) -> None:
        error = APIConnectionError(
            message="connection failed",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        response = tool_error_response("Document search", error)

        self.assertEqual(response["error_code"], "provider_unavailable")
        self.assertEqual(response["error_type"], "APIConnectionError")
        self.assertTrue(response["retryable"])

    def test_rate_limit_is_explicitly_retryable(self) -> None:
        response = httpx.Response(
            429,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        info = classify_runtime_error(
            RateLimitError("rate limited", response=response, body={})
        )

        self.assertEqual(info.code, "provider_rate_limited")
        self.assertTrue(info.retryable)

    def test_exhausted_provider_quota_is_not_retryable(self) -> None:
        response = httpx.Response(
            429,
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        )
        error = RateLimitError(
            "no credits",
            response=response,
            body={"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted"}},
        )

        info = classify_runtime_error(error)

        self.assertEqual(info.code, "provider_quota_exhausted")
        self.assertFalse(info.retryable)


if __name__ == "__main__":
    unittest.main()
