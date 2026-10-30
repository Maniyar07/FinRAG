"""Verified provider connections using the operating system's trusted CAs."""

from __future__ import annotations

import atexit
import ssl
from functools import lru_cache

from openai import DefaultHttpxClient


@lru_cache(maxsize=1)
def provider_ssl_context() -> ssl.SSLContext:
    # On Windows, Python includes certificates from the Windows trust store.
    # OpenSSL's SSL_CERT_FILE / SSL_CERT_DIR overrides are also respected.
    return ssl.create_default_context()


@lru_cache(maxsize=1)
def model_http_client() -> DefaultHttpxClient:
    """Reuse the synchronous chat transport across short-lived model wrappers."""
    client = DefaultHttpxClient(verify=provider_ssl_context())
    atexit.register(client.close)
    return client
