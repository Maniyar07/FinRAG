from __future__ import annotations

import os

import streamlit as st

from src.config import DEBUG_TRACES
from src.app.chat_service import ChatService
from src.generation.citations import citation_label, escape_currency_for_markdown
from src.schemas import Decision, Scope

st.set_page_config(page_title="FinRAG Analyst", page_icon="FR", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 1.5rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #d7dce2;}
    .scope-line {color: #4b5563; font-size: 0.9rem; margin-bottom: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_service(index_version: str) -> ChatService:
    return ChatService(index_version=index_version)


def ui_scope() -> Scope:
    company = st.session_state.company_filter
    year = st.session_state.year_filter
    doc_type = st.session_state.doc_type_filter
    return Scope(
        tickers=() if company == "All companies" else (company,),
        years=() if year == "All years" else (year,),
        doc_type=None if doc_type == "Both document types" else doc_type,
    )


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Evidence sources ({len(sources)})"):
        rows = [
            {
                "ID": source["id"],
                "Citation": citation_label(source),
                "Company": source["ticker"],
                "Fiscal year": source["fiscal_year"],
                "Period": source.get("fiscal_period"),
                "Type": source["doc_type"],
                "Section": source["section"],
                "PDF page": source["pdf_page"],
                "File": source["source"],
            }
            for source in sources
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)
        for source in sources:
            st.caption(f"{source['id']}: retrieved passage")
            st.code(source.get("evidence_text", "Passage text unavailable."), language="markdown")


def render_chat_markdown(content: str) -> None:
    """Render financial text without treating currency markers as LaTeX."""
    st.markdown(escape_currency_for_markdown(content), unsafe_allow_html=False)


st.title("FinRAG Analyst")
st.caption("Cited analysis of the filings and calls in the selected index")

with st.sidebar:
    st.subheader("Search scope")
    index_version = st.text_input(
        "Index version", value=os.getenv("FINRAG_INDEX_VERSION", "")
    ).strip()
    service = None
    if index_version:
        try:
            service = get_service(index_version)
        except Exception as error:
            st.error(f"Cannot open index: {error}")
    keys = service.available_keys if service else set()
    st.selectbox(
        "Company",
        ["All companies"] + sorted({t for t, y, d in keys}),
        key="company_filter",
    )
    st.selectbox(
        "Fiscal year",
        ["All years"] + sorted({y for t, y, d in keys}),
        key="year_filter",
    )
    st.selectbox(
        "Document type",
        ["Both document types"] + sorted({d for t, y, d in keys}),
        key="doc_type_filter",
    )
    selection = (index_version, st.session_state.company_filter,
                 st.session_state.year_filter, st.session_state.doc_type_filter)
    if st.session_state.get("conversation_selection", selection) != selection:
        st.session_state.active_scope = Scope()
        st.session_state.pending_clarification = None
        st.session_state.history_start = len(st.session_state.get("messages", []))
    st.session_state.conversation_selection = selection
    st.divider()
    show_debug = st.checkbox("Show retrieval trace", value=DEBUG_TRACES)
    active_scope = st.session_state.get("active_scope", Scope())
    st.caption("Conversation scope")
    st.write(active_scope.label() if active_scope.complete else "Not established")
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_scope = Scope()
        st.session_state.pending_clarification = None
        st.session_state.history_start = 0
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_scope" not in st.session_state:
    st.session_state.active_scope = Scope()
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

if not index_version:
    st.info("Enter an index version in the sidebar after running ingestion.")
    st.stop()

try:
    service = get_service(index_version)
except Exception as error:
    st.error(f"FinRAG could not open the selected index: {error}")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("scope_note"):
            st.caption(message["scope_note"])
        render_chat_markdown(message["content"])
        render_sources(message.get("sources", []))
        if show_debug and message.get("trace"):
            with st.expander("Retrieval trace"):
                st.json(message["trace"])

question = st.chat_input("Ask about a company, fiscal year, filing, or earnings call")
if question:
    prior_history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[st.session_state.get("history_start", 0):]
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        render_chat_markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking scope and evidence..."):
            try:
                result = service.ask(
                    question,
                    ui_scope=ui_scope(),
                    previous_scope=st.session_state.active_scope,
                    history=prior_history,
                    pending_clarification=st.session_state.pending_clarification,
                )
            except Exception as error:
                result = None
                st.error(f"The request could not be completed: {error}")

        if result is not None:
            scope_note = ""
            if result.inherited_fields:
                scope_note = (
                    f"Using previous {', '.join(result.inherited_fields)}: "
                    f"{result.scope.label()}"
                )
                st.caption(scope_note)
            render_chat_markdown(result.answer)
            render_sources(result.sources)
            if show_debug and result.trace:
                with st.expander("Retrieval trace"):
                    st.json(result.trace)
            if result.scope.complete and result.decision in {
                Decision.ANSWERED,
                Decision.INSUFFICIENT_EVIDENCE,
                Decision.VALIDATION_FAILED,
            }:
                st.session_state.active_scope = result.scope
            st.session_state.pending_clarification = result.pending_clarification
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.sources,
                    "decision": result.decision.value,
                    "scope_note": scope_note,
                    "trace": result.trace,
                }
            )
