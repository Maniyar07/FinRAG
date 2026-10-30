from __future__ import annotations

import ssl
from pathlib import Path

import certifi

from src.provider_http import provider_ssl_context


def test_provider_tls_requires_valid_certificates_and_matching_hostnames() -> None:
    context = provider_ssl_context()

    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert context.cert_store_stats()["x509_ca"] > 0


def test_configured_ca_bundle_is_loaded_without_disabling_verification(tmp_path, monkeypatch) -> None:
    # Use a real CA from HTTPX's dependency; no network or private keys are needed.
    bundle = Path(certifi.where()).read_text(encoding="ascii")
    start = bundle.index("-----BEGIN CERTIFICATE-----")
    end = bundle.index("-----END CERTIFICATE-----", start) + len("-----END CERTIFICATE-----")
    certificate = bundle[start:end]
    ca_file = tmp_path / "approved-ca.pem"
    ca_file.write_text(certificate, encoding="ascii")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_file))
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    provider_ssl_context.cache_clear()
    try:
        context = provider_ssl_context()

        assert ssl.PEM_cert_to_DER_cert(certificate) in context.get_ca_certs(binary_form=True)
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
    finally:
        provider_ssl_context.cache_clear()
