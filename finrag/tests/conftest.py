"""Test process safeguards shared by the FinRAG suite."""

from __future__ import annotations

import os


# A developer may enable LangSmith in their local .env. Keep unit-test prompts,
# fixtures, and synthetic document text local unless a test explicitly opts in.
os.environ.setdefault("LANGSMITH_TRACING", "false")
