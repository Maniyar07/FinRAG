import sys
from pathlib import Path

# Ensure root directory is in the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.app.chat_service import ChatService

# Comprehensive test suite covering all complex financial categories
complex_test_queries = [
    {
        "category": "1. Domain Abbreviations & Scale Normalization",
        "query": (
            "Using TSLA's 2025 annual 10-K, find net cash provided by operating "
            "activities and capital expenditures, then calculate free cash flow "
            "as operating cash flow minus capital expenditures with the financial "
            "calculator. Also report 2025 automotive revenue."
        )
    },
    {
        "category": "2. Multi-Hop Parallel Percentage Change",
        "query": (
            "Using Tesla's annual 10-Ks, compare total revenue for fiscal 2024 "
            "and fiscal 2025, then calculate 2025 year-over-year revenue growth "
            "with the financial calculator."
        )
    },
    {
        "category": "3. Multi-Company Cross-Comparison",
        "query": (
            "Compare Microsoft (MSFT) and Tesla (TSLA) total revenue in their "
            "2025 annual 10-Ks using parallel searches."
        )
    },
    {
        "category": "4. Multi-Period CAGR Calculation",
        "query": (
            "Using Microsoft's annual 10-Ks, calculate the CAGR for MSFT revenue "
            "from fiscal 2024 to fiscal 2025 over 1 elapsed period with the "
            "financial calculator."
        )
    },
    {
        "category": "5. Structured Statement Table Lookup",
        "query": (
            "Show me Microsoft's 2025 annual 10-K income statement table using "
            "the statement lookup tool."
        )
    }
]

def main():
    print("Initializing FinRAG backends for comprehensive evaluation...")
    service = ChatService()
    try:
        for index, item in enumerate(complex_test_queries, start=1):
            print(f"\nTEST CATEGORY: {item['category']}\nQUERY: {item['query']}")
            response = service.ask(item["query"], thread_id=f"complex_eval_{index}")
            print(f"DECISION: {response.decision.value}\nAGENT ANSWER:\n{response.answer}")
            print(f"SOURCES: {len(response.sources)}")
            print(f"[Execution Trace]: {response.trace.get('agent', {})}")
    finally:
        service.close()


if __name__ == "__main__":
    main()
