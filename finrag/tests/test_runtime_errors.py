from __future__ import annotations

import ssl
import unittest

import httpx
from openai import APIConnectionError, RateLimitError

from src.runtime_errors import classify_runtime_error, runtime_error_message, tool_error_response


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

    def test_sdk_unwrapped_quota_body_is_not_treated_as_rate_limit(self) -> None:
        response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
        error = RateLimitError(
            "no credits", response=response,
            body={"type": "insufficient_quota", "code": "insufficient_quota"},
        )

        info = classify_runtime_error(error)

        self.assertEqual(info.code, "provider_quota_exhausted")
        self.assertFalse(info.retryable)
        self.assertIn("billing", runtime_error_message(info))

    def test_tls_failure_is_actionable_without_leaking_exception_details(self) -> None:
        error = APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
        transport_error = httpx.ConnectError("private network and secret diagnostic")
        transport_error.__cause__ = ssl.SSLCertVerificationError(1, "certificate verify failed")
        error.__cause__ = transport_error

        info = classify_runtime_error(error)

        self.assertEqual(info.code, "provider_tls_error")
        self.assertFalse(info.retryable)
        self.assertIn("certificate", runtime_error_message(info))
        self.assertNotIn("secret", runtime_error_message(info))

    def test_httpcore_certificate_marker_is_recognized_without_ssl_exception(self) -> None:
        error = APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
        error.__cause__ = httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate")

        self.assertEqual(classify_runtime_error(error).code, "provider_tls_error")

    def test_connection_failure_is_not_presented_as_a_billing_problem(self) -> None:
        error = APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))

        message = runtime_error_message(classify_runtime_error(error))

        self.assertIn("could not connect", message)
        self.assertNotIn("billing", message)


if __name__ == "__main__":
    unittest.main()
