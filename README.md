# FinRAG Analyst

Evidence-grounded financial research over SEC 10-K filings and Q4 earnings-call
transcripts.

The application lives in [`finrag/`](finrag/). Start with the complete
[`finrag/README.md`](finrag/README.md), which includes:

- a five-minute setup for Windows, macOS, and Linux;
- system and data-ingestion architecture diagrams;
- the complete indexing and question-answering workflows;
- corpus coverage, configuration, API examples, testing, and troubleshooting.

For HTTP integration details, see the
[`FastAPI reference`](finrag/README_FASTAPI.md).

For a presentation-ready explanation of the business problem, design decisions,
end-to-end request flow, worked examples, evaluation results, limitations, roadmap,
and manager Q&A, see the
[`manager presentation and project walkthrough`](finrag/PROJECT_WALKTHROUGH.md).

## Quick start

```powershell
cd finrag
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Add OPENAI_API_KEY to .env
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

> FinRAG is a document-research system, not a live-market-data service or an
> investment adviser. Verify material conclusions against the cited source.
