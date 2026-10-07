"""Isolated, audit-first dataset and LoRA utilities for FinRAG.

This package is intentionally additive.  It reads the existing immutable index and
gold-builder artifacts, but it does not modify the runtime application, its indexes,
or its canonical evaluation files.
"""

from .dataset_factory.models import MasterExample

__all__ = ["MasterExample"]
