from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from src.app.chat_service import ChatService
from src.config import ACTIVE_INDEX_VERSION, DEBUG_TRACES, normalize_index_version
from src.constants import COMPANY_NAMES


LOGGER = logging.getLogger(__name__)
ServiceFactory = Callable[..., ChatService]


def configured_index_versions() -> tuple[str, ...]:
    """Return the explicitly allowed indexes, with the active index first."""
    configured = os.getenv("FINRAG_API_INDEX_VERSIONS", "").strip()
    candidates = [value.strip() for value in configured.split(",") if value.strip()]
    if ACTIVE_INDEX_VERSION:
        candidates.insert(0, ACTIVE_INDEX_VERSION)
    if not candidates:
        candidates.append(normalize_index_version(None))
    return tuple(dict.fromkeys(normalize_index_version(value) for value in candidates))


class ServiceManager:
    """Own and reuse one ChatService for each configured index version."""

    def __init__(
        self,
        *,
        service_factory: ServiceFactory = ChatService,
        index_versions: tuple[str, ...] | list[str] | None = None,
        allow_traces: bool | None = None,
    ) -> None:
        versions = tuple(index_versions or configured_index_versions())
        if not versions:
            raise ValueError("At least one index version must be configured.")
        self.index_versions = tuple(
            dict.fromkeys(normalize_index_version(value) for value in versions)
        )
        self.default_index = self.index_versions[0]
        self.allow_traces = DEBUG_TRACES if allow_traces is None else bool(allow_traces)
        self._service_factory = service_factory
        self._services: dict[str, ChatService] = {}

    @property
    def ready(self) -> bool:
        return len(self._services) == len(self.index_versions)

    def start(self) -> None:
        if self._services:
            return
        opened: dict[str, ChatService] = {}
        try:
            for version in self.index_versions:
                opened[version] = self._service_factory(index_version=version)
        except Exception:
            for service in opened.values():
                try:
                    service.close()
                except Exception:
                    LOGGER.exception("Failed to close a partially initialized ChatService.")
            raise
        self._services = opened

    def close(self) -> None:
        services, self._services = self._services, {}
        for service in services.values():
            try:
                service.close()
            except Exception:
                LOGGER.exception("Failed to close ChatService during shutdown.")

    def get(self, index_version: str) -> ChatService:
        version = normalize_index_version(index_version)
        if version not in self._services:
            raise KeyError(version)
        return self._services[version]

    def catalogue(self) -> list[dict[str, Any]]:
        indexes: list[dict[str, Any]] = []
        for version in self.index_versions:
            service = self.get(version)
            keys = sorted(
                (str(ticker), str(year), str(doc_type))
                for ticker, year, doc_type in service.available_keys
            )
            tickers = sorted({ticker for ticker, _, _ in keys})
            indexes.append(
                {
                    "version": version,
                    "companies": [
                        {"ticker": ticker, "name": COMPANY_NAMES.get(ticker, ticker)}
                        for ticker in tickers
                    ],
                    "fiscal_years": sorted({year for _, year, _ in keys}),
                    "document_types": sorted({doc_type for _, _, doc_type in keys}),
                    "availability": [
                        {
                            "company": ticker,
                            "fiscal_year": year,
                            "document_type": doc_type,
                        }
                        for ticker, year, doc_type in keys
                    ],
                }
            )
        return indexes

