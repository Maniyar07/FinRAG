"use strict";

const STORAGE_KEY = "finrag.chat-history.v1";
const THEME_KEY = "finrag.theme";
const SIDEBAR_KEY = "finrag.sidebar-collapsed";
const MAX_SAVED_CHATS = 12;
const MAX_SAVED_MESSAGES = 40;

const emptyScope = () => ({
  tickers: [],
  years: [],
  doc_type: null,
  requested_groups: [],
  required_doc_types: [],
});

function cleanPendingClarification(candidate) {
  if (!candidate || typeof candidate !== "object" || !candidate.scope) return null;
  const originalQuestion = String(candidate.original_question || "").trim().slice(0, 4000);
  if (!originalQuestion) return null;
  return {
    original_question: originalQuestion,
    scope: {
      tickers: Array.isArray(candidate.scope.tickers) ? candidate.scope.tickers.slice(0, 12) : [],
      years: Array.isArray(candidate.scope.years) ? candidate.scope.years.slice(0, 12) : [],
      doc_type: candidate.scope.doc_type || null,
      requested_groups: Array.isArray(candidate.scope.requested_groups)
        ? candidate.scope.requested_groups.slice(0, 24)
        : [],
      required_doc_types: Array.isArray(candidate.scope.required_doc_types)
        ? candidate.scope.required_doc_types.slice(0, 2)
        : [],
    },
    candidate_tickers: Array.isArray(candidate.candidate_tickers)
      ? candidate.candidate_tickers.slice(0, 3)
      : [],
    missing_fields: Array.isArray(candidate.missing_fields)
      ? candidate.missing_fields.slice(0, 4)
      : [],
    query_expansions: Array.isArray(candidate.query_expansions)
      ? candidate.query_expansions.slice(0, 3)
      : [],
  };
}

const state = {
  catalogue: null,
  chats: [],
  activeChatId: null,
  controller: null,
  busy: false,
  lastAttempt: null,
  theme: localStorage.getItem(THEME_KEY) || "system",
};

const elements = {
  appShell: document.getElementById("appShell"),
  sidebar: document.getElementById("sidebar"),
  collapseSidebar: document.getElementById("collapseSidebar"),
  mobileMenuButton: document.getElementById("mobileMenuButton"),
  sidebarScrim: document.getElementById("sidebarScrim"),
  newChatButton: document.getElementById("newChatButton"),
  clearHistoryButton: document.getElementById("clearHistoryButton"),
  historySearch: document.getElementById("historySearch"),
  historyList: document.getElementById("historyList"),
  indexFilter: document.getElementById("indexFilter"),
  companyFilter: document.getElementById("companyFilter"),
  companyPicker: document.getElementById("companyPicker"),
  companyPickerButton: document.getElementById("companyPickerButton"),
  companySelectedText: document.getElementById("companySelectedText"),
  companyMenu: document.getElementById("companyMenu"),
  yearFilter: document.getElementById("yearFilter"),
  documentFilter: document.getElementById("documentFilter"),
  scopeStatus: document.getElementById("scopeStatus"),
  traceControl: document.getElementById("traceControl"),
  traceToggle: document.getElementById("traceToggle"),
  themePicker: document.getElementById("themePicker"),
  themeButton: document.getElementById("themeButton"),
  themeIcon: document.getElementById("themeIcon"),
  themeMenu: document.getElementById("themeMenu"),
  themeChoices: [...document.querySelectorAll("[data-theme-choice]")],
  healthDot: document.getElementById("healthDot"),
  healthText: document.getElementById("healthText"),
  conversationTitle: document.getElementById("conversationTitle"),
  activeScope: document.getElementById("activeScope"),
  chatScroll: document.getElementById("chatScroll"),
  welcomePanel: document.getElementById("welcomePanel"),
  promptGrid: document.getElementById("promptGrid"),
  messages: document.getElementById("messages"),
  stageBar: document.getElementById("stageBar"),
  stageText: document.getElementById("stageText"),
  stopButton: document.getElementById("stopButton"),
  retryBar: document.getElementById("retryBar"),
  retryButton: document.getElementById("retryButton"),
  composerForm: document.getElementById("composerForm"),
  questionInput: document.getElementById("questionInput"),
  sendButton: document.getElementById("sendButton"),
  evidenceDialog: document.getElementById("evidenceDialog"),
  evidenceList: document.getElementById("evidenceList"),
  closeEvidenceButton: document.getElementById("closeEvidenceButton"),
  traceDialog: document.getElementById("traceDialog"),
  traceOutput: document.getElementById("traceOutput"),
  closeTraceButton: document.getElementById("closeTraceButton"),
  toastRegion: document.getElementById("toastRegion"),
};

function makeId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function newChatRecord(filters = null) {
  const now = new Date().toISOString();
  return {
    id: makeId(),
    title: "New financial analysis",
    createdAt: now,
    updatedAt: now,
    messages: [],
    scope: emptyScope(),
    pending_clarification: null,
    filters: filters || defaultFilters(),
  };
}

function defaultFilters() {
  return {
    index_version: state.catalogue?.default_index || "",
    company: "",
    fiscal_year: "",
    document_type: "",
  };
}

function currentChat() {
  return state.chats.find((chat) => chat.id === state.activeChatId) || null;
}

function cleanStoredChat(candidate) {
  if (!candidate || typeof candidate !== "object" || !Array.isArray(candidate.messages)) {
    return null;
  }
  const messages = candidate.messages
    .filter((message) => message && ["user", "assistant"].includes(message.role))
    .slice(-MAX_SAVED_MESSAGES)
    .map((message) => ({
      role: message.role,
      content: String(message.content || "").slice(0, 100_000),
      decision: message.decision ? String(message.decision) : undefined,
      sources: Array.isArray(message.sources) ? message.sources.slice(0, 12) : [],
      trace: message.trace && typeof message.trace === "object" ? message.trace : null,
      inherited_fields: Array.isArray(message.inherited_fields) ? message.inherited_fields : [],
    }));
  const filters = candidate.filters && typeof candidate.filters === "object"
    ? {
        index_version: String(candidate.filters.index_version || ""),
        company: String(candidate.filters.company || ""),
        fiscal_year: String(candidate.filters.fiscal_year || ""),
        document_type: String(candidate.filters.document_type || ""),
      }
    : defaultFilters();
  const scope = candidate.scope && typeof candidate.scope === "object"
    ? {
        tickers: Array.isArray(candidate.scope.tickers) ? candidate.scope.tickers : [],
        years: Array.isArray(candidate.scope.years) ? candidate.scope.years : [],
        doc_type: candidate.scope.doc_type || null,
        requested_groups: Array.isArray(candidate.scope.requested_groups) ? candidate.scope.requested_groups : [],
        required_doc_types: Array.isArray(candidate.scope.required_doc_types) ? candidate.scope.required_doc_types : [],
        label: candidate.scope.label ? String(candidate.scope.label) : "",
        complete: Boolean(candidate.scope.complete),
      }
    : emptyScope();
  return {
    id: String(candidate.id || makeId()),
    title: String(candidate.title || "Financial analysis").slice(0, 90),
    createdAt: String(candidate.createdAt || new Date().toISOString()),
    updatedAt: String(candidate.updatedAt || new Date().toISOString()),
    messages,
    scope,
    pending_clarification: cleanPendingClarification(candidate.pending_clarification),
    filters,
  };
}

function loadHistory() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (Array.isArray(stored)) {
      state.chats = stored.map(cleanStoredChat).filter(Boolean).slice(0, MAX_SAVED_CHATS);
    }
  } catch (_error) {
    localStorage.removeItem(STORAGE_KEY);
  }
  if (!state.chats.length) {
    const chat = newChatRecord();
    state.chats = [chat];
  }
  state.activeChatId = state.chats[0].id;
}

function compactForStorage(chat) {
  return {
    ...chat,
    messages: chat.messages.slice(-MAX_SAVED_MESSAGES).map((message) => ({
      ...message,
      sources: (message.sources || []).slice(0, 12).map((source) => ({
        ...source,
        evidence_text: String(source.evidence_text || "").slice(0, 20_000),
      })),
    })),
  };
}

function saveHistory() {
  state.chats.sort((left, right) => String(right.updatedAt).localeCompare(String(left.updatedAt)));
  state.chats = state.chats.slice(0, MAX_SAVED_CHATS);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.chats.map(compactForStorage)));
  } catch (_error) {
    showToast("Chat history could not be saved in this browser.", "error");
  }
  renderHistory();
}

function setOptions(select, items, selectedValue = "") {
  select.replaceChildren();
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    select.append(option);
  }
  const valid = items.some((item) => item.value === selectedValue);
  select.value = valid ? selectedValue : (items[0]?.value || "");
}

function closeCompanyMenu({ restoreFocus = false } = {}) {
  elements.companyMenu.hidden = true;
  elements.companyPickerButton.setAttribute("aria-expanded", "false");
  if (restoreFocus) elements.companyPickerButton.focus();
}

function openCompanyMenu({ focusSelected = false } = {}) {
  if (elements.companyPickerButton.disabled) return;
  elements.companyMenu.hidden = false;
  elements.companyPickerButton.setAttribute("aria-expanded", "true");
  if (focusSelected) {
    const selected = elements.companyMenu.querySelector('[aria-selected="true"]');
    (selected || elements.companyMenu.querySelector("button"))?.focus();
  }
}

function selectCompany(value) {
  elements.companyFilter.value = value;
  syncCompanyPicker();
  closeCompanyMenu({ restoreFocus: true });
  elements.companyFilter.dispatchEvent(new Event("change", { bubbles: true }));
}

function syncCompanyPicker() {
  const value = elements.companyFilter.value;
  elements.companySelectedText.textContent = value || "All";
  for (const option of elements.companyMenu.querySelectorAll(".company-option")) {
    const selected = option.dataset.value === value;
    option.setAttribute("aria-selected", String(selected));
    const check = option.querySelector(".option-check");
    if (check) check.textContent = selected ? "\u2713" : "";
  }
}

function populateCompanyPicker(companies) {
  elements.companyMenu.replaceChildren();
  const options = [
    { ticker: "", name: "Search every available company", shortLabel: "All" },
    ...companies.map((company) => ({
      ticker: company.ticker,
      name: company.name,
      shortLabel: company.ticker,
    })),
  ];
  for (const company of options) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "company-option";
    button.setAttribute("role", "option");
    button.dataset.value = company.ticker;
    const ticker = document.createElement("strong");
    ticker.textContent = company.shortLabel;
    const name = document.createElement("small");
    name.textContent = company.name;
    const check = document.createElement("span");
    check.className = "option-check";
    check.setAttribute("aria-hidden", "true");
    button.append(ticker, name, check);
    button.addEventListener("click", () => selectCompany(company.ticker));
    elements.companyMenu.append(button);
  }
  syncCompanyPicker();
}

function selectedIndex() {
  if (!state.catalogue) return null;
  return state.catalogue.indexes.find((item) => item.version === elements.indexFilter.value)
    || state.catalogue.indexes[0]
    || null;
}

function populateIndexFilters(desired = null) {
  if (!state.catalogue) return;
  const indexItems = state.catalogue.indexes.map((item) => ({
    value: item.version,
    label: item.version,
  }));
  const indexValue = desired?.index_version || elements.indexFilter.value || state.catalogue.default_index;
  setOptions(elements.indexFilter, indexItems, indexValue);
  elements.indexFilter.disabled = false;

  const index = selectedIndex();
  if (!index) return;
  setOptions(
    elements.companyFilter,
    [
      { value: "", label: "All companies" },
      ...index.companies.map((company) => ({
        value: company.ticker,
        label: `${company.ticker} · ${company.name}`,
      })),
    ],
    desired?.company || "",
  );
  populateCompanyPicker(index.companies);
  setOptions(
    elements.yearFilter,
    [
      { value: "", label: "All years" },
      ...index.fiscal_years.map((year) => ({ value: year, label: year })),
    ],
    desired?.fiscal_year || "",
  );
  setOptions(
    elements.documentFilter,
    [
      { value: "", label: "10-K and transcript" },
      ...index.document_types.map((docType) => ({
        value: docType,
        label: docType === "10K" ? "Annual 10-K" : "Q4 earnings transcript",
      })),
    ],
    desired?.document_type || "",
  );
  elements.companyFilter.disabled = false;
  elements.companyPickerButton.disabled = false;
  elements.yearFilter.disabled = false;
  elements.documentFilter.disabled = false;
}

function readFilters() {
  return {
    index_version: elements.indexFilter.value,
    company: elements.companyFilter.value,
    fiscal_year: elements.yearFilter.value,
    document_type: elements.documentFilter.value,
  };
}

function handleFilterChange(event) {
  if (event.target === elements.indexFilter) {
    closeCompanyMenu();
    populateIndexFilters({ index_version: elements.indexFilter.value });
  }
  if (event.target === elements.companyFilter) syncCompanyPicker();
  const chat = currentChat();
  if (!chat) return;
  chat.filters = readFilters();
  chat.scope = emptyScope();
  chat.pending_clarification = null;
  chat.updatedAt = new Date().toISOString();
  updateScopeDisplay(chat.scope);
  saveHistory();
}

function renderHistory() {
  elements.historyList.replaceChildren();
  const query = elements.historySearch.value.trim().toLocaleLowerCase();
  const visibleChats = state.chats.filter((chat) => chat.title.toLocaleLowerCase().includes(query));

  if (!visibleChats.length) {
    const empty = document.createElement("li");
    empty.className = "history-empty";
    empty.textContent = "No matching conversations";
    elements.historyList.append(empty);
    return;
  }

  for (const chat of visibleChats) {
    const item = document.createElement("li");
    item.className = `history-item${chat.id === state.activeChatId ? " active" : ""}`;

    const selectButton = document.createElement("button");
    selectButton.type = "button";
    selectButton.className = "history-main";
    selectButton.setAttribute("aria-label", `Open conversation: ${chat.title}`);
    const title = document.createElement("strong");
    title.textContent = chat.title;
    selectButton.append(title);
    selectButton.addEventListener("click", () => activateChat(chat.id));

    const actions = document.createElement("div");
    actions.className = "history-actions";

    const menuButton = document.createElement("button");
    menuButton.type = "button";
    menuButton.className = "history-action";
    menuButton.textContent = "\u22ee";
    menuButton.setAttribute("aria-label", `Conversation actions: ${chat.title}`);
    menuButton.setAttribute("aria-haspopup", "menu");
    menuButton.setAttribute("aria-expanded", "false");

    const menu = document.createElement("div");
    menu.className = "history-menu";
    menu.setAttribute("role", "menu");
    menu.hidden = true;

    const renameButton = document.createElement("button");
    renameButton.type = "button";
    renameButton.setAttribute("role", "menuitem");
    renameButton.textContent = "Rename";
    renameButton.addEventListener("click", () => startRenameChat(chat.id, item));

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "delete";
    deleteButton.setAttribute("role", "menuitem");
    deleteButton.textContent = "Delete";
    deleteButton.addEventListener("click", () => deleteChat(chat.id));
    menu.append(renameButton, deleteButton);

    menuButton.addEventListener("click", () => {
      const opening = menu.hidden;
      closeHistoryMenus();
      menu.hidden = !opening;
      menuButton.setAttribute("aria-expanded", String(opening));
      if (opening) renameButton.focus();
    });
    menu.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeHistoryMenus();
        menuButton.focus();
      }
    });

    actions.append(menuButton);
    item.append(selectButton, actions, menu);
    elements.historyList.append(item);
  }
}

function closeHistoryMenus() {
  for (const menu of elements.historyList.querySelectorAll(".history-menu")) menu.hidden = true;
  for (const button of elements.historyList.querySelectorAll('.history-action[aria-expanded="true"]')) {
    button.setAttribute("aria-expanded", "false");
  }
}

function startRenameChat(chatId, item) {
  const chat = state.chats.find((candidate) => candidate.id === chatId);
  if (!chat || item.querySelector(".history-rename")) return;
  closeHistoryMenus();

  const input = document.createElement("input");
  input.className = "history-rename";
  input.type = "text";
  input.maxLength = 90;
  input.value = chat.title;
  input.setAttribute("aria-label", "Conversation name");
  item.append(input);
  input.focus();
  input.select();

  let finished = false;
  const finish = (save) => {
    if (finished) return;
    finished = true;
    const title = input.value.trim();
    if (save && title) {
      chat.title = title;
      chat.updatedAt = new Date().toISOString();
      saveHistory();
      renderConversation();
    } else {
      input.remove();
    }
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      finish(true);
    }
    if (event.key === "Escape") {
      event.preventDefault();
      finish(false);
    }
  });
  input.addEventListener("blur", () => finish(true));
}

function activateChat(chatId) {
  if (state.busy) abortRequest();
  const chat = state.chats.find((item) => item.id === chatId);
  if (!chat) return;
  state.activeChatId = chat.id;
  if (state.catalogue) populateIndexFilters(chat.filters);
  renderConversation();
  renderHistory();
  closeMobileSidebar();
}

function createNewChat() {
  if (state.busy) abortRequest();
  elements.historySearch.value = "";
  const chat = newChatRecord(state.catalogue ? defaultFilters() : null);
  state.chats.unshift(chat);
  state.activeChatId = chat.id;
  if (state.catalogue) populateIndexFilters(chat.filters);
  saveHistory();
  renderConversation();
  closeMobileSidebar();
  elements.questionInput.focus();
}

function deleteChat(chatId) {
  const wasActive = state.activeChatId === chatId;
  state.chats = state.chats.filter((chat) => chat.id !== chatId);
  if (!state.chats.length) state.chats.push(newChatRecord());
  if (wasActive) state.activeChatId = state.chats[0].id;
  saveHistory();
  activateChat(state.activeChatId);
}

function clearHistory() {
  if (!window.confirm("Clear all locally saved FinRAG conversations?")) return;
  if (state.busy) abortRequest();
  elements.historySearch.value = "";
  const chat = newChatRecord();
  state.chats = [chat];
  state.activeChatId = chat.id;
  saveHistory();
  if (state.catalogue) populateIndexFilters(chat.filters);
  renderConversation();
}

function updateScopeDisplay(scope) {
  const complete = Boolean(scope?.complete || (scope?.tickers?.length && scope?.years?.length));
  const label = scope?.label || "Scope not established";
  const scopeLabel = elements.activeScope.querySelector("span:last-child");
  scopeLabel.textContent = complete ? label : "Scope not established";
  elements.activeScope.classList.toggle("established", complete);
  elements.scopeStatus.textContent = complete ? "Active" : "Ready";
}

function renderConversation() {
  const chat = currentChat();
  if (!chat) return;
  elements.messages.replaceChildren();
  const hasMessages = chat.messages.length > 0;
  elements.welcomePanel.hidden = hasMessages;
  elements.conversationTitle.textContent = "Financial research workspace";
  updateScopeDisplay(chat.scope);

  for (const message of chat.messages) {
    elements.messages.append(renderMessage(message));
  }
  requestAnimationFrame(() => {
    elements.chatScroll.scrollTop = elements.chatScroll.scrollHeight;
  });
}

function answerPresentation(value, sources) {
  const original = String(value || "");
  const availableIds = new Set(sources.map((source) => String(source.id).toUpperCase()));
  const citedIds = [];
  for (const match of original.matchAll(/\[(S\d+)(?::[^\]]+)?\]/gi)) {
    const sourceId = match[1].toUpperCase();
    if (availableIds.has(sourceId) && !citedIds.includes(sourceId)) citedIds.push(sourceId);
  }

  const withoutAppendix = original.replace(
    /\n{2,}\*\*Sources\*\*\s*\n[\s\S]*$/i,
    "",
  );
  const displayText = withoutAppendix
    .split(/\r?\n/)
    .filter((line) => !/^\s*(?:source|sources|citation|citations)\s*:\s*(?:\[S\d+\]\s*)+\.?\s*$/i.test(line))
    .join("\n")
    .trim();
  const visibleIds = new Set(
    [...displayText.matchAll(/\[(S\d+)\]/gi)].map((match) => match[1].toUpperCase()),
  );
  return { displayText, citedIds, hasVisibleCitations: visibleIds.size > 0 };
}

function createCitationChip(sourceId, sources) {
  const citation = document.createElement("button");
  citation.type = "button";
  citation.className = "citation-chip";
  citation.textContent = sourceId;
  citation.setAttribute("aria-label", `Open evidence ${sourceId}`);
  citation.addEventListener("click", () => openEvidence(sources, sourceId));
  return citation;
}

function renderMessage(message) {
  const article = document.createElement("article");
  article.className = `message ${message.role}`;

  if (message.role === "user") {
    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = message.content;
    article.append(body);
    return article;
  }

  const heading = document.createElement("div");
  heading.className = "message-heading";
  const label = document.createElement("span");
  label.className = "assistant-label";
  label.textContent = "FinRAG";
  const decision = document.createElement("span");
  decision.className = "decision-badge";
  const decisionLabels = {
    answered: "Cited answer",
    clarify: "Needs details",
    insufficient_evidence: "Limited evidence",
    validation_failed: "Answer could not be verified",
    data_unavailable: "Data unavailable",
    scope_conflict: "Scope conflict",
    out_of_scope: "Outside collection",
    error: "Could not complete",
  };
  decision.textContent = decisionLabels[message.decision]
    || String(message.decision || "Response").replaceAll("_", " ");
  heading.append(label, decision);
  article.append(heading);

  if (message.inherited_fields?.length) {
    const note = document.createElement("p");
    note.className = "scope-note";
    note.textContent = `Continued from the previous ${message.inherited_fields.join(", ")} scope.`;
    article.append(note);
  }

  const sources = message.sources || [];
  const presentation = answerPresentation(message.content, sources);
  article.append(renderMarkdown(presentation.displayText, sources));

  if (!presentation.hasVisibleCitations && presentation.citedIds.length) {
    const citations = document.createElement("div");
    citations.className = "message-citations";
    const citationsLabel = document.createElement("span");
    citationsLabel.textContent = "Cited evidence";
    citations.append(citationsLabel);
    for (const sourceId of presentation.citedIds) {
      citations.append(createCitationChip(sourceId, sources));
    }
    article.append(citations);
  }

  if (sources.length || message.trace) {
    const actions = document.createElement("div");
    actions.className = "message-actions";
    if (sources.length) {
      const evidenceButton = document.createElement("button");
      evidenceButton.type = "button";
      evidenceButton.className = "source-action";
      evidenceButton.textContent = `View sources (${sources.length})`;
      evidenceButton.addEventListener("click", () => openEvidence(sources));
      actions.append(evidenceButton);
    }
    if (message.trace) {
      const traceButton = document.createElement("button");
      traceButton.type = "button";
      traceButton.className = "source-action";
      traceButton.textContent = "View retrieval trace";
      traceButton.addEventListener("click", () => openTrace(message.trace));
      actions.append(traceButton);
    }
    article.append(actions);
  }
  return article;
}

function unescapeDisplayText(value) {
  return String(value)
    .replace(/<\/?(?:u|b|strong|em|i)>/gi, "")
    .replaceAll("\\$", "$")
    .replaceAll("\\|", "|");
}

function appendInline(parent, value, sources) {
  const text = unescapeDisplayText(value);
  const sourceIds = new Set(sources.map((source) => String(source.id).toUpperCase()));
  const tokenPattern = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*|\[(S\d+)\])/gi;
  let cursor = 0;
  for (const match of text.matchAll(tokenPattern)) {
    if (match.index > cursor) parent.append(document.createTextNode(text.slice(cursor, match.index)));
    const token = match[0];
    if (token.startsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      parent.append(strong);
    } else if (token.startsWith("`")) {
      const code = document.createElement("code");
      code.textContent = token.slice(1, -1);
      parent.append(code);
    } else if (token.startsWith("*")) {
      const emphasis = document.createElement("em");
      emphasis.textContent = token.slice(1, -1);
      parent.append(emphasis);
    } else {
      const sourceId = match[2].toUpperCase();
      if (sourceIds.has(sourceId)) {
        parent.append(createCitationChip(sourceId, sources));
      } else {
        parent.append(document.createTextNode(token));
      }
    }
    cursor = match.index + token.length;
  }
  if (cursor < text.length) parent.append(document.createTextNode(text.slice(cursor)));
}

function splitTableRow(line) {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  const cells = [];
  let current = "";
  let escaped = false;
  for (const character of trimmed) {
    if (escaped) {
      current += character;
      escaped = false;
    } else if (character === "\\") {
      escaped = true;
      current += character;
    } else if (character === "|") {
      cells.push(current.trim());
      current = "";
    } else {
      current += character;
    }
  }
  cells.push(current.trim());
  return cells;
}

function isTableSeparator(line) {
  if (!line?.trim().startsWith("|")) return false;
  const cells = splitTableRow(line);
  return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.replaceAll(" ", "")));
}

function startsBlock(lines, index) {
  const line = lines[index] || "";
  return /^#{1,4}\s+/.test(line)
    || /^```/.test(line.trim())
    || /^\s*[-*]\s+/.test(line)
    || /^\s*\d+\.\s+/.test(line)
    || /^>\s?/.test(line)
    || (/^\s*\|/.test(line) && isTableSeparator(lines[index + 1]));
}

function renderMarkdown(value, sources) {
  const container = document.createElement("div");
  container.className = "markdown-body";
  const lines = String(value || "").replaceAll("\r\n", "\n").replaceAll("\r", "\n").split("\n");
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    if (line.trim().startsWith("```")) {
      const codeLines = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = unescapeDisplayText(codeLines.join("\n"));
      pre.append(code);
      container.append(pre);
      continue;
    }

    const headingMatch = /^(#{1,4})\s+(.+)$/.exec(line);
    if (headingMatch) {
      const level = Math.min(headingMatch[1].length + 1, 4);
      const heading = document.createElement(`h${level}`);
      appendInline(heading, headingMatch[2], sources);
      container.append(heading);
      index += 1;
      continue;
    }

    if (/^\s*\|/.test(line) && isTableSeparator(lines[index + 1])) {
      const headerCells = splitTableRow(line);
      index += 2;
      const rows = [];
      while (index < lines.length && /^\s*\|/.test(lines[index])) {
        const cells = splitTableRow(lines[index]);
        if (cells.length === headerCells.length) rows.push(cells);
        index += 1;
      }
      const wrapper = document.createElement("div");
      wrapper.className = "table-scroll";
      wrapper.tabIndex = 0;
      wrapper.setAttribute("role", "region");
      wrapper.setAttribute("aria-label", "Financial data table");
      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      for (const cellValue of headerCells) {
        const cell = document.createElement("th");
        cell.scope = "col";
        appendInline(cell, cellValue, sources);
        headerRow.append(cell);
      }
      thead.append(headerRow);
      const tbody = document.createElement("tbody");
      for (const rowValues of rows) {
        const row = document.createElement("tr");
        for (const cellValue of rowValues) {
          const cell = document.createElement("td");
          appendInline(cell, cellValue, sources);
          row.append(cell);
        }
        tbody.append(row);
      }
      table.append(thead, tbody);
      wrapper.append(table);
      container.append(wrapper);
      continue;
    }

    const unordered = /^\s*[-*]\s+(.+)$/.exec(line);
    const ordered = /^\s*\d+\.\s+(.+)$/.exec(line);
    if (unordered || ordered) {
      const list = document.createElement(unordered ? "ul" : "ol");
      const pattern = unordered ? /^\s*[-*]\s+(.+)$/ : /^\s*\d+\.\s+(.+)$/;
      while (index < lines.length) {
        const itemMatch = pattern.exec(lines[index]);
        if (!itemMatch) break;
        const item = document.createElement("li");
        appendInline(item, itemMatch[1], sources);
        list.append(item);
        index += 1;
        // Markdown permits blank lines between list items. Keep them in one
        // semantic list so repeated model-authored markers such as `1.` are
        // displayed as 1, 2, 3 instead of restarting at 1 each time.
        let nextItem = index;
        while (nextItem < lines.length && !lines[nextItem].trim()) nextItem += 1;
        if (nextItem < lines.length && pattern.test(lines[nextItem])) {
          index = nextItem;
        }
      }
      container.append(list);
      continue;
    }

    if (/^>\s?/.test(line)) {
      const quote = document.createElement("blockquote");
      const quoteLines = [];
      while (index < lines.length && /^>\s?/.test(lines[index])) {
        quoteLines.push(lines[index].replace(/^>\s?/, ""));
        index += 1;
      }
      appendInline(quote, quoteLines.join(" "), sources);
      container.append(quote);
      continue;
    }

    const paragraphLines = [line];
    index += 1;
    while (index < lines.length && lines[index].trim() && !startsBlock(lines, index)) {
      paragraphLines.push(lines[index]);
      index += 1;
    }
    const paragraph = document.createElement("p");
    appendInline(paragraph, paragraphLines.join(" "), sources);
    container.append(paragraph);
  }
  return container;
}

function sourceTitle(source) {
  const docType = source.doc_type === "10K" ? "10-K" : (source.doc_type || "Document");
  return `${source.id} · ${source.ticker || "Unknown"} ${source.fiscal_year || ""} ${docType}`.trim();
}

function openEvidence(sources, targetId = null) {
  elements.evidenceList.replaceChildren();
  if (!sources.length) {
    const empty = document.createElement("p");
    empty.className = "empty-dialog";
    empty.textContent = "No evidence sources were returned for this response.";
    elements.evidenceList.append(empty);
  }
  let targetCard = null;
  for (const source of sources) {
    const card = document.createElement("article");
    card.className = "evidence-card";
    card.dataset.sourceId = String(source.id);
    if (targetId && String(source.id).toUpperCase() === targetId.toUpperCase()) {
      card.classList.add("highlighted");
      targetCard = card;
    }
    const title = document.createElement("h3");
    title.textContent = sourceTitle(source);
    const metadata = document.createElement("div");
    metadata.className = "source-metadata";
    const values = [
      source.section,
      source.pdf_page && source.pdf_page !== "not available" ? `Page ${source.pdf_page}` : null,
      source.call_date ? `Call ${source.call_date}` : null,
      source.fiscal_period,
      source.speaker ? `${source.speaker}${source.speaker_role ? ` · ${source.speaker_role}` : ""}` : null,
      source.source,
    ].filter(Boolean);
    for (const value of values) {
      const chip = document.createElement("span");
      chip.textContent = String(value);
      metadata.append(chip);
    }
    const evidence = document.createElement("pre");
    evidence.textContent = String(source.evidence_text || "Evidence text unavailable.");
    card.append(title, metadata, evidence);
    elements.evidenceList.append(card);
  }
  if (!elements.evidenceDialog.open) elements.evidenceDialog.showModal();
  if (targetCard) requestAnimationFrame(() => targetCard.scrollIntoView({ block: "start" }));
}

function openTrace(trace) {
  elements.traceOutput.textContent = JSON.stringify(trace, null, 2);
  if (!elements.traceDialog.open) elements.traceDialog.showModal();
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  elements.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), 5_000);
}

function setBusy(busy, stage = "Preparing analysis…") {
  state.busy = busy;
  elements.stageBar.hidden = !busy;
  elements.stageText.textContent = stage;
  elements.sendButton.disabled = busy || !elements.questionInput.value.trim() || !state.catalogue;
  elements.questionInput.disabled = busy;
  for (const select of [elements.indexFilter, elements.companyFilter, elements.yearFilter, elements.documentFilter]) {
    select.disabled = busy || !state.catalogue;
  }
  elements.companyPickerButton.disabled = busy || !state.catalogue;
  if (busy) closeCompanyMenu();
}

function setStage(stage) {
  const labels = {
    request_validated: "Request and scope controls validated",
    finrag_pipeline: "Resolving scope and retrieving grounded evidence…",
    response_ready: "Checking citations and preparing the response…",
  };
  elements.stageText.textContent = labels[stage.stage] || stage.message || "Processing…";
}

function autoResizeInput() {
  elements.questionInput.style.height = "auto";
  elements.questionInput.style.height = `${Math.min(elements.questionInput.scrollHeight, 150)}px`;
  elements.sendButton.disabled = state.busy || !elements.questionInput.value.trim() || !state.catalogue;
}

class ApiClientError extends Error {
  constructor(message, details = null) {
    super(message);
    this.name = "ApiClientError";
    this.details = details;
  }
}

async function httpError(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch (_error) {
    // The status code is still enough to return a safe browser message.
  }
  const error = payload?.error;
  const reference = error?.trace_id ? ` Reference: ${error.trace_id}.` : "";
  return new ApiClientError(`${error?.message || `Request failed (${response.status}).`}${reference}`, error);
}

function parseEventFrame(frame) {
  let event = "message";
  const dataLines = [];
  for (const line of frame.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (!dataLines.length) return null;
  return { event, data: JSON.parse(dataLines.join("\n")) };
}

async function streamChat(payload, signal) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) throw await httpError(response);
  if (!response.body) throw new ApiClientError("This browser cannot read the response stream.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult = null;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    buffer = buffer.replaceAll("\r\n", "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (frame.trim() && !frame.trimStart().startsWith(":")) {
        const parsed = parseEventFrame(frame);
        if (parsed?.event === "stage") setStage(parsed.data);
        if (parsed?.event === "result") finalResult = parsed.data;
        if (parsed?.event === "error") {
          const error = parsed.data?.error;
          const reference = error?.trace_id ? ` Reference: ${error.trace_id}.` : "";
          throw new ApiClientError(`${error?.message || "The stream failed."}${reference}`, error);
        }
      }
      boundary = buffer.indexOf("\n\n");
    }
    if (done) break;
  }
  if (!finalResult) throw new ApiClientError("The response ended before a final result arrived.");
  return finalResult;
}

function requestPayload(question, chat, history) {
  const filters = readFilters();
  return {
    index_version: filters.index_version,
    question,
    history: history.slice(-24).map((message) => ({
      role: message.role,
      content: message.content,
    })),
    previous_scope: {
      tickers: chat.scope?.tickers || [],
      years: chat.scope?.years || [],
      doc_type: chat.scope?.doc_type || null,
      requested_groups: chat.scope?.requested_groups || [],
      required_doc_types: chat.scope?.required_doc_types || [],
    },
    pending_clarification: cleanPendingClarification(chat.pending_clarification),
    filters: {
      company: filters.company || null,
      fiscal_year: filters.fiscal_year || null,
      document_type: filters.document_type || null,
    },
    include_trace: Boolean(state.catalogue?.trace_available && elements.traceToggle.checked),
  };
}

async function submitQuestion(question, { retry = false } = {}) {
  const normalized = String(question || "").trim();
  if (!normalized || state.busy || !state.catalogue) return;
  const chat = currentChat();
  if (!chat) return;

  const history = retry && state.lastAttempt
    ? state.lastAttempt.history
    : chat.messages.map((message) => ({ role: message.role, content: message.content }));
  const payload = retry && state.lastAttempt
    ? state.lastAttempt.payload
    : requestPayload(normalized, chat, history);

  if (!retry) {
    chat.messages.push({ role: "user", content: normalized });
    if (chat.messages.filter((message) => message.role === "user").length === 1) {
      chat.title = normalized.length > 76 ? `${normalized.slice(0, 75)}…` : normalized;
    }
    chat.filters = readFilters();
    chat.updatedAt = new Date().toISOString();
    state.lastAttempt = { chatId: chat.id, question: normalized, history, payload };
    elements.questionInput.value = "";
    autoResizeInput();
    saveHistory();
    renderConversation();
  }

  elements.retryBar.hidden = true;
  state.controller = new AbortController();
  setBusy(true, "Connecting to the FinRAG pipeline…");
  try {
    const result = await streamChat(payload, state.controller.signal);
    const targetChat = state.chats.find((item) => item.id === (state.lastAttempt?.chatId || chat.id));
    if (!targetChat) return;
    targetChat.messages.push({
      role: "assistant",
      content: String(result.answer || ""),
      decision: String(result.decision || "response"),
      sources: Array.isArray(result.sources) ? result.sources : [],
      trace: result.trace || null,
      inherited_fields: Array.isArray(result.inherited_fields) ? result.inherited_fields : [],
    });
    targetChat.pending_clarification = cleanPendingClarification(result.pending_clarification);
    const scopeDecisions = new Set(["answered", "insufficient_evidence", "validation_failed"]);
    if (result.scope?.complete && scopeDecisions.has(result.decision)) targetChat.scope = result.scope;
    targetChat.updatedAt = new Date().toISOString();
    state.lastAttempt = null;
    saveHistory();
    if (targetChat.id === state.activeChatId) renderConversation();
  } catch (error) {
    if (error.name === "AbortError") {
      showToast("Request stopped.");
    } else {
      if (
        error.details?.code === "invalid_scope"
        && String(error.details?.message || "").includes("pending_clarification")
      ) {
        chat.pending_clarification = null;
        if (state.lastAttempt?.payload) {
          state.lastAttempt.payload.pending_clarification = null;
        }
        saveHistory();
      }
      showToast(error.message || "The request could not be completed.", "error");
      elements.retryBar.hidden = false;
    }
  } finally {
    state.controller = null;
    setBusy(false);
    elements.questionInput.focus();
  }
}

function abortRequest() {
  state.controller?.abort();
}

function retryLastRequest() {
  if (!state.lastAttempt || state.busy) return;
  if (state.lastAttempt.chatId !== state.activeChatId) activateChat(state.lastAttempt.chatId);
  submitQuestion(state.lastAttempt.question, { retry: true });
}

function applyTheme(theme) {
  const allowed = new Set(["system", "light", "dark"]);
  state.theme = allowed.has(theme) ? theme : "system";
  document.documentElement.dataset.theme = state.theme;
  localStorage.setItem(THEME_KEY, state.theme);
  const themeDetails = {
    system: { label: "System", icon: "\u263c" },
    light: { label: "Light", icon: "\u2600" },
    dark: { label: "Dark", icon: "\u263e" },
  }[state.theme];
  elements.themeIcon.textContent = themeDetails.icon;
  elements.themeButton.setAttribute("aria-label", `${themeDetails.label} theme. Choose color theme`);
  elements.themeButton.title = `${themeDetails.label} theme`;
  for (const button of elements.themeChoices) {
    const active = button.dataset.themeChoice === state.theme;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
}

function closeThemeMenu({ restoreFocus = false } = {}) {
  elements.themeMenu.hidden = true;
  elements.themeButton.setAttribute("aria-expanded", "false");
  if (restoreFocus) elements.themeButton.focus();
}

function toggleThemeMenu() {
  const opening = elements.themeMenu.hidden;
  elements.themeMenu.hidden = !opening;
  elements.themeButton.setAttribute("aria-expanded", String(opening));
  if (opening) elements.themeMenu.querySelector("button.active")?.focus();
}

function setSidebarCollapsed(collapsed) {
  elements.appShell.classList.toggle("sidebar-collapsed", collapsed);
  elements.collapseSidebar.setAttribute("aria-expanded", String(!collapsed));
  elements.collapseSidebar.setAttribute("aria-label", collapsed ? "Expand sidebar" : "Collapse sidebar");
  localStorage.setItem(SIDEBAR_KEY, String(collapsed));
}

function openMobileSidebar() {
  document.body.classList.add("sidebar-open");
  elements.mobileMenuButton.setAttribute("aria-expanded", "true");
}

function closeMobileSidebar() {
  document.body.classList.remove("sidebar-open");
  elements.mobileMenuButton.setAttribute("aria-expanded", "false");
}

async function loadServerState() {
  try {
    const [healthResponse, catalogueResponse] = await Promise.all([
      fetch("/api/health", { headers: { Accept: "application/json" } }),
      fetch("/api/catalogue", { headers: { Accept: "application/json" } }),
    ]);
    if (!healthResponse.ok) throw await httpError(healthResponse);
    if (!catalogueResponse.ok) throw await httpError(catalogueResponse);
    const [health, catalogue] = await Promise.all([healthResponse.json(), catalogueResponse.json()]);
    if (!Array.isArray(catalogue.indexes) || !catalogue.indexes.length) {
      throw new ApiClientError("No indexes are available.");
    }
    state.catalogue = catalogue;
    elements.healthDot.classList.add("ready");
    elements.healthText.textContent = health.service_ready ? "Index ready" : "Starting";
    elements.scopeStatus.textContent = "Ready";
    elements.traceControl.hidden = !catalogue.trace_available;

    for (const chat of state.chats) {
      if (!catalogue.indexes.some((index) => index.version === chat.filters.index_version)) {
        chat.filters = { ...chat.filters, index_version: catalogue.default_index };
        chat.scope = emptyScope();
        chat.pending_clarification = null;
      }
    }
    populateIndexFilters(currentChat()?.filters || defaultFilters());
    renderConversation();
    saveHistory();
    autoResizeInput();
  } catch (error) {
    elements.healthDot.classList.add("error");
    elements.healthText.textContent = "Unavailable";
    elements.scopeStatus.textContent = "Offline";
    showToast(error.message || "FinRAG could not connect to the API.", "error");
  }
}

function bindEvents() {
  elements.newChatButton.addEventListener("click", createNewChat);
  elements.clearHistoryButton.addEventListener("click", clearHistory);
  elements.historySearch.addEventListener("input", renderHistory);
  elements.collapseSidebar.addEventListener("click", () => {
    setSidebarCollapsed(!elements.appShell.classList.contains("sidebar-collapsed"));
  });
  elements.mobileMenuButton.addEventListener("click", openMobileSidebar);
  elements.sidebarScrim.addEventListener("click", closeMobileSidebar);
  elements.themeButton.addEventListener("click", toggleThemeMenu);
  elements.themeMenu.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      closeThemeMenu({ restoreFocus: true });
    }
  });
  for (const button of elements.themeChoices) {
    button.addEventListener("click", () => {
      applyTheme(button.dataset.themeChoice);
      closeThemeMenu({ restoreFocus: true });
    });
  }
  elements.companyPickerButton.addEventListener("click", () => {
    if (elements.companyMenu.hidden) openCompanyMenu();
    else closeCompanyMenu();
  });
  elements.companyPickerButton.addEventListener("keydown", (event) => {
    if (["ArrowDown", "ArrowUp"].includes(event.key)) {
      event.preventDefault();
      openCompanyMenu({ focusSelected: true });
    }
    if (event.key === "Escape") closeCompanyMenu();
  });
  elements.companyMenu.addEventListener("keydown", (event) => {
    const options = [...elements.companyMenu.querySelectorAll(".company-option")];
    const currentIndex = options.indexOf(document.activeElement);
    if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      let nextIndex = currentIndex;
      if (event.key === "Home") nextIndex = 0;
      if (event.key === "End") nextIndex = options.length - 1;
      if (event.key === "ArrowDown") nextIndex = Math.min(currentIndex + 1, options.length - 1);
      if (event.key === "ArrowUp") nextIndex = Math.max(currentIndex - 1, 0);
      options[nextIndex]?.focus();
    }
    if (event.key === "Escape" || event.key === "Tab") {
      closeCompanyMenu({ restoreFocus: event.key === "Escape" });
    }
  });
  elements.stopButton.addEventListener("click", abortRequest);
  elements.retryButton.addEventListener("click", retryLastRequest);
  elements.closeEvidenceButton.addEventListener("click", () => elements.evidenceDialog.close());
  elements.closeTraceButton.addEventListener("click", () => elements.traceDialog.close());
  for (const dialog of [elements.evidenceDialog, elements.traceDialog]) {
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
  }

  for (const select of [elements.indexFilter, elements.companyFilter, elements.yearFilter, elements.documentFilter]) {
    select.addEventListener("change", handleFilterChange);
  }

  elements.questionInput.addEventListener("input", autoResizeInput);
  elements.questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      elements.composerForm.requestSubmit();
    }
  });
  elements.composerForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitQuestion(elements.questionInput.value);
  });

  elements.promptGrid.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-prompt]");
    if (!button || state.busy) return;
    elements.questionInput.value = button.dataset.prompt;
    autoResizeInput();
    submitQuestion(button.dataset.prompt);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeMobileSidebar();
      closeCompanyMenu();
      closeThemeMenu();
      closeHistoryMenus();
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      elements.questionInput.focus();
    }
  });
  document.addEventListener("pointerdown", (event) => {
    if (!elements.companyPicker.contains(event.target)) closeCompanyMenu();
    if (!elements.themePicker.contains(event.target)) closeThemeMenu();
    if (!elements.historyList.contains(event.target)) closeHistoryMenus();
  });
}

function initialize() {
  applyTheme(state.theme);
  setSidebarCollapsed(localStorage.getItem(SIDEBAR_KEY) === "true");
  loadHistory();
  bindEvents();
  renderHistory();
  renderConversation();
  loadServerState();
}

initialize();
