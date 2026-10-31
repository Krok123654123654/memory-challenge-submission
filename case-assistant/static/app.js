// Case assistant front end: plain JavaScript, no build step. All case text is inserted as text, never as HTML.

const ROLE_COLOURS = [["suspect", "#d92d20"], ["defendant", "#d92d20"], ["interest", "#e04f16"], ["victim", "#dc6803"],
                      ["claimant", "#dc6803"], ["witness", "#3651d4"], ["officer", "#079455"], ["expert", "#0e9384"],
                      ["forensic", "#0e9384"]];
const KIND_ICON = {statement: "i-statement", interview: "i-interview", messages: "i-messages", email: "i-email",
                   report: "i-report", letter: "i-letter"};
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const state = {config: null, caseName: null, summary: null, chats: {}, timeline: {from: "", to: "", person: ""}};
const $ = (sel) => document.querySelector(sel);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function icon(id, cls = "icon") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", cls);
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#${id}`);
  svg.append(use);
  return svg;
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

function niceDate(iso, withYear = true) {
  if (!iso) return "";
  const [y, m, d] = iso.split("-").map(Number);
  return `${d} ${MONTHS[m - 1]}${withYear ? " " + y : ""}`;
}
function weekday(iso) { return new Date(`${iso}T12:00:00`).toLocaleDateString("en-GB", {weekday: "long"}); }
function roleColour(role) { return (ROLE_COLOURS.find(([w]) => (role || "").toLowerCase().includes(w)) || [, "#667085"])[1]; }
function roleTag(text, role = text) {
  const c = roleColour(role);
  return el("span", {class: "role", style: `background:${c}1a;color:${c}`}, text || "involved");
}
function initials(name) {
  return name.replace(/^(DS|PC|Dr|Mr|Ms|Mrs)\s+/, "").split(/\s+/).map(w => w[0]).slice(0, 2).join("").toUpperCase();
}
function avatar(p, large = false) {
  return el("div", {class: `avatar${large ? " lg" : ""}`, style: `background:${roleColour(p.role)}`}, initials(p.name));
}
function personById(id) { return state.summary.people.find(p => p.id === id); }
function docLink(docId, line, text) {
  return el("a", {href: `#documents/${docId}`, onclick: (e) => { e.preventDefault(); openDoc(docId, line); }}, text);
}

// ---------- document drawer ----------

async function openDoc(docId, line) {
  const body = $("#drawer-body");
  body.replaceChildren(el("p", {class: "note"}, "Opening…"));
  $("#drawer-title").replaceChildren();
  $("#drawer").classList.add("open");
  $("#scrim").classList.add("open");
  $("#drawer").setAttribute("aria-hidden", "false");
  const d = await api(`/api/cases/${state.caseName}/documents/${docId}`);
  const summary = state.summary.documents.find(x => x.id === docId);
  const author = d.author && personById(d.author) ? personById(d.author).name : "—";
  $("#drawer-title").replaceChildren(
    el("div", {class: "doc-title"}, icon(KIND_ICON[d.kind] || "i-file"), el("h2", {}, d.title)),
    el("div", {class: "source-meta"}, `${d.kind} · written ${niceDate(d.date)} · author: ${author}`),
    el("div", {class: "chips", style: "margin:8px 0 0"}, (summary?.people || []).map(pid => {
      const p = personById(pid);
      const tag = roleTag(p.name, p.role);
      tag.addEventListener("click", () => { closeDoc(); location.hash = `#people/${pid}`; });
      return tag;
    })));
  const firstSpeaker = d.units.find(u => u.speaker)?.speaker;
  let target = null;
  body.replaceChildren(...d.units.map(u => {
    const node = ["interview", "messages"].includes(d.kind) && u.speaker_label
      ? el("div", {class: `bubble${u.speaker === firstSpeaker ? "" : " other"}`},
          el("div", {class: "who"}, u.speaker_label, el("span", {class: "line", style: "margin-left:8px"}, u.line)), u.text)
      : el("div", {class: "para"}, el("span", {class: "line"}, u.line), u.text);
    if (line && u.line === line) { node.classList.add("target"); target = node; }
    return node;
  }));
  if (target) setTimeout(() => target.scrollIntoView({block: "center", behavior: "smooth"}), 60);
}
function closeDoc() {
  $("#drawer").classList.remove("open");
  $("#scrim").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
}

// ---------- chat ----------

function chatState() {
  if (!state.chats[state.caseName]) {
    state.chats[state.caseName] = {session: crypto.randomUUID(), messages: [], scope: null, busy: false};
  }
  return state.chats[state.caseName];
}

function windowText(w) {
  const [a, b] = w;
  if (a && b) return a === b ? niceDate(a) : `${niceDate(a)} – ${niceDate(b)}`;
  return a ? `from ${niceDate(a)}` : `up to ${niceDate(b)}`;
}

function renderReply(r) {
  const byLabel = Object.fromEntries(r.evidence.map(e => [e.label, e]));
  const kind = r.verdict === "yes" ? "yes" : r.verdict === "no" ? "no" : r.verdict === "disputed" ? "disputed"
    : (r.verdict === "not in file" || (r.text || "").toUpperCase().startsWith("NOT IN FILE")) ? "none" : "";
  const reply = el("div", {class: `reply ${kind}`});
  if (r.verdict) reply.append(el("div", {class: `verdict ${kind}`}, r.verdict === "not in file" ? "Not in file" : r.verdict));
  const text = el("div", {class: "text"});
  if (r.error) {
    text.append(el("span", {class: "error"}, `The model failed: ${r.error}`));
  } else if (r.text !== null && r.text !== undefined) {
    for (const piece of r.text.split(/(\[?S\d+\]?)/)) {
      const m = piece.match(/^\[?(S\d+)\]?$/);
      const e = m && byLabel[m[1]];
      text.append(e ? el("button", {class: "cite", title: `${e.title}, line ${e.line}`, onclick: () => openDoc(e.doc_id, e.line)}, m[1])
                    : document.createTextNode(piece));
    }
  } else if (r.evidence.length) {
    text.append(`Here ${r.evidence.length === 1 ? "is the passage" : `are the ${r.evidence.length} passages`} most relevant to your question` +
      `${r.window ? ", in the order things happened" : ""}. Switch "Answers" to Reactor for a written answer.`);
  } else {
    text.append(r.window ? "Nothing in the file is dated inside that period." : "Nothing in the file matches that question.");
  }
  reply.append(text);
  const meta = el("div", {class: "meta-line"});
  if (r.window) meta.append(el("span", {class: "pill"}, icon("i-calendar"), windowText(r.window)));
  if (r.person) meta.append(el("span", {class: "pill"}, icon("i-people"), r.person));
  if (r.text !== null && r.text !== undefined) meta.append(`${r.backend} · ${r.seconds}s · check every claim against its source`);
  if (meta.childNodes.length) reply.append(meta);
  if (r.evidence.length) {
    const cited = new Set(r.cited);
    const sources = el("details", {class: "sources", open: r.text === null || r.text === undefined ? "" : null},
      el("summary", {}, `${r.evidence.length} source${r.evidence.length > 1 ? "s" : ""}${cited.size ? ` · ${cited.size} cited` : ""}`),
      r.evidence.map(e => el("div", {class: `src${cited.has(e.label) ? " cited" : ""}`, onclick: () => openDoc(e.doc_id, e.line)},
        el("span", {class: "label"}, e.label),
        el("div", {},
          el("div", {class: "t"}, el("b", {}, e.event_date && e.date_source !== "document"
            ? `${niceDate(e.event_date)}${e.event_time ? " " + e.event_time : ""} · ` : ""), e.title, ` · line ${e.line}`,
            e.speaker ? ` · ${e.speaker}` : ""),
          el("div", {class: "x"}, e.text)))));
    reply.append(sources);
  }
  return reply;
}

function renderChat() {
  if (!state.summary) return;  // the case is still loading
  const chat = chatState();
  const view = $("#view-chat");
  const textarea = el("textarea", {rows: 1, placeholder: chat.scope
    ? `Ask about ${personById(chat.scope).name}…` : "Ask anything about this case… e.g. What happened between 1 and 31 August 2025?"});
  const send = el("button", {class: "primary sendbtn", type: "submit", title: "Send", disabled: chat.busy}, icon("i-send"));
  const grow = () => { textarea.style.height = "auto"; textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`; };
  textarea.addEventListener("input", grow);
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });
  const form = el("form", {onsubmit: (e) => { e.preventDefault(); sendMessage(textarea.value); }}, textarea, send);
  const scope = chat.scope ? el("div", {class: "scope"}, el("span", {class: "pill"}, icon("i-people"),
    `About ${personById(chat.scope).name}`),
    el("button", {class: "link", type: "button", onclick: () => { chat.scope = null; renderChat(); }}, "ask about the whole case")) : null;
  const composer = el("div", {class: "composer"}, scope, form);

  if (!chat.messages.length) {
    view.replaceChildren(el("div", {class: "chat"},
      el("div", {class: "chat-empty"},
        el("h2", {}, chat.scope ? `Ask about ${personById(chat.scope).name}` : "Ask about this case"),
        el("div", {}, "Answers cite the documents they come from; click a citation to open the document at that line."),
        el("div", {class: "examples"}, (state.summary.examples || []).map(q =>
          el("button", {class: "chip", type: "button", onclick: () => sendMessage(q)}, q)))),
      el("div", {style: "flex:1"}), composer));
    textarea.focus();
    return;
  }
  const backendName = $("#backend").selectedOptions[0]?.textContent || "";
  const messages = el("div", {class: "messages"}, chat.messages.map(m => m.role === "user"
    ? el("div", {class: "msg user"}, el("div", {class: "bubble-text"}, m.text))
    : el("div", {class: "msg assistant"}, el("div", {class: "bot"}, icon("i-scale")),
        el("div", {class: "body"}, m.pending
          ? el("div", {class: "reply"}, el("span", {class: "typing"}, el("i"), el("i"), el("i"),
              m.backend === "retrieval" ? " Searching the file…" : " Reading the file and asking the model… this can take a minute"))
          : renderReply(m.result)))));
  view.replaceChildren(el("div", {class: "chat"},
    el("div", {class: "chat-toolbar"},
      el("p", {class: "note"}, `Answers from ${backendName}`),
      el("button", {class: "btn-ghost", type: "button", onclick: newChat, disabled: chat.busy}, icon("i-plus"), "New chat")),
    messages, composer));
  if (!chat.busy) textarea.focus();
  composer.scrollIntoView({block: "end"});
}

async function sendMessage(text) {
  const chat = chatState();
  text = text.trim();
  if (!text || chat.busy) return;
  const backend = $("#backend").value;
  chat.busy = true;
  chat.messages.push({role: "user", text}, {role: "assistant", pending: true, backend});
  renderChat();
  let result;
  try {
    result = await api(`/api/cases/${state.caseName}/chat`, {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({session: chat.session, message: text, person: chat.scope, backend, semantic: $("#semantic").checked})});
  } catch (err) {
    result = {question: text, evidence: [], cited: [], text: null, error: err.message, window: null};
  }
  chat.messages[chat.messages.length - 1] = {role: "assistant", result};
  chat.busy = false;
  if (location.hash.startsWith("#chat") || !location.hash) renderChat();
}

async function newChat() {
  const chat = chatState();
  await api(`/api/cases/${state.caseName}/chat/reset`, {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({session: chat.session})}).catch(() => {});
  state.chats[state.caseName] = {session: crypto.randomUUID(), messages: [], scope: chat.scope, busy: false};
  renderChat();
}

// ---------- other views ----------

async function renderTimeline() {
  const t = state.timeline;
  const from = el("input", {type: "date", value: t.from});
  const to = el("input", {type: "date", value: t.to});
  const person = el("select", {}, el("option", {value: ""}, "Everyone"),
    state.summary.people.map(p => el("option", {value: p.id, selected: p.id === t.person}, p.name)));
  const list = el("div", {}, el("p", {class: "note"}, "Loading…"));
  const load = async () => {
    Object.assign(state.timeline, {from: from.value, to: to.value, person: person.value});
    const qs = new URLSearchParams(Object.entries({date_from: from.value, date_to: to.value, person: person.value})
      .filter(([, v]) => v));
    const {events} = await api(`/api/cases/${state.caseName}/timeline?${qs}`);
    if (!events.length) { list.replaceChildren(el("p", {class: "note"}, "No dated events.")); return; }
    const days = new Map();
    for (const e of events) (days.get(e.date) || days.set(e.date, []).get(e.date)).push(e);
    list.replaceChildren(el("div", {class: "tl"}, [...days].map(([day, items]) => el("div", {class: "tl-group"},
      el("div", {class: "tl-day"}, niceDate(day), el("small", {}, `${weekday(day)} · ${items.length} event${items.length > 1 ? "s" : ""}`)),
      items.map(e => el("div", {class: "tl-item"}, el("div", {class: "tl-time"}, e.time || ""),
        el("div", {},
          el("div", {class: "source-meta"}, icon(KIND_ICON[e.kind] || "i-file"), docLink(e.doc_id, e.line, e.title),
            el("span", {class: "dot"}, `line ${e.line}`), e.speaker ? el("span", {class: "dot"}, e.speaker) : null),
          el("div", {}, e.text))))))));
  };
  for (const input of [from, to, person]) input.addEventListener("change", load);
  $("#view-timeline").replaceChildren(
    el("div", {class: "filters"}, icon("i-calendar"), "Show", from, "to", to, person,
      el("button", {class: "link", type: "button", onclick: () => { from.value = ""; to.value = ""; person.value = ""; load(); }}, "show all")),
    list);
  await load();
}

async function renderPeople(personId) {
  const view = $("#view-people");
  if (!personId) {
    const filter = el("input", {type: "text", placeholder: "Find a person", style: "margin-bottom:14px;width:100%;max-width:360px"});
    const grid = el("div", {class: "grid"});
    const draw = () => grid.replaceChildren(...state.summary.people
      .filter(p => `${p.name} ${p.role} ${p.aliases.join(" ")}`.toLowerCase().includes(filter.value.toLowerCase()))
      .map(p => el("div", {class: "card person", onclick: () => (location.hash = `#people/${p.id}`)},
        avatar(p),
        el("div", {}, el("h3", {}, p.name), roleTag(p.role),
          el("div", {class: "counts"}, `Speaks in ${p.said_docs} · mentioned in ${p.mentioned_docs} more`)))));
    filter.addEventListener("input", draw);
    draw();
    view.replaceChildren(filter, grid);
    return;
  }
  view.replaceChildren(el("p", {class: "note"}, "Loading…"));
  const p = await api(`/api/cases/${state.caseName}/people/${personId}`);
  const docList = (groups, empty) => groups.length ? groups.map(g => el("div", {class: "card"},
    el("div", {class: "doc-title"}, icon(KIND_ICON[g.kind] || "i-file"), docLink(g.id, g.units[0]?.line, g.title)),
    el("div", {class: "source-meta"}, `${niceDate(g.date)} · ${g.units.length} passage${g.units.length > 1 ? "s" : ""}`),
    g.units.map(u => el("div", {class: "para"}, el("span", {class: "line"}, u.line),
      ["interview", "messages"].includes(g.kind) && u.speaker_label ? el("b", {}, `${u.speaker_label}: `) : null, u.text))))
    : [el("p", {class: "note"}, empty)];
  const askAbout = () => {
    const chat = chatState();
    if (chat.scope !== p.id && chat.messages.length) newChat();
    chatState().scope = p.id;
    location.hash = "#chat";
  };
  view.replaceChildren(
    el("button", {class: "back", onclick: () => (location.hash = "#people")}, icon("i-back"), "All people"),
    el("div", {class: "profile"}, avatar(p, true),
      el("div", {style: "flex:1"}, el("h2", {}, p.name), roleTag(p.role),
        el("div", {class: "note", style: "margin:6px 0 0"},
          [p.aliases.length ? `Also called ${p.aliases.join(", ")}` : "", p.notes].filter(Boolean).join(" · "))),
      el("button", {class: "primary btn-ghost", type: "button", onclick: askAbout}, icon("i-chat"), `Ask about ${p.name.split(" ")[0]}`)),
    el("div", {class: "cols"},
      el("div", {}, el("h3", {class: "section"}, `Their own words · ${p.own_words.length}`),
        docList(p.own_words, "No statement or turn by this person in the file.")),
      el("div", {}, el("h3", {class: "section"}, `Mentioned by others · ${p.mentioned_by_others.length}`),
        docList(p.mentioned_by_others, "Nobody else names this person in the file."))));
}

async function renderDocuments(docId) {
  const docs = state.summary.documents;
  docId = docId || docs[0]?.id;
  const filter = el("input", {type: "text", placeholder: "Find a document", style: "width:100%;margin-bottom:8px"});
  const list = el("div", {class: "doclist"});
  const drawList = () => list.replaceChildren(filter, ...docs
    .filter(d => `${d.title} ${d.kind} ${d.date}`.toLowerCase().includes(filter.value.toLowerCase()))
    .map(d => el("button", {class: d.id === docId ? "active" : "", onclick: () => (location.hash = `#documents/${d.id}`)},
      icon(KIND_ICON[d.kind] || "i-file"), el("span", {}, d.title, el("small", {}, `${niceDate(d.date)} · ${d.kind}`)))));
  filter.addEventListener("input", () => { drawList(); filter.focus(); });
  drawList();
  const viewer = el("div", {class: "viewer"}, el("p", {class: "note"}, "Loading…"));
  $("#view-documents").replaceChildren(el("div", {class: "docs"}, list, viewer));
  if (!docId) return;
  const d = await api(`/api/cases/${state.caseName}/documents/${docId}`);
  const summary = docs.find(x => x.id === docId);
  const author = d.author && personById(d.author) ? personById(d.author).name : "—";
  const firstSpeaker = d.units.find(u => u.speaker)?.speaker;
  viewer.replaceChildren(
    el("div", {class: "doc-title"}, icon(KIND_ICON[d.kind] || "i-file"), el("h2", {}, d.title)),
    el("div", {class: "source-meta"}, `${d.kind} · ${niceDate(d.date)} · author: ${author}`),
    el("div", {class: "chips"}, (summary?.people || []).map(pid => {
      const p = personById(pid);
      const tag = roleTag(p.name, p.role);
      tag.addEventListener("click", () => (location.hash = `#people/${pid}`));
      return tag;
    })),
    ...d.units.map(u => ["interview", "messages"].includes(d.kind) && u.speaker_label
      ? el("div", {class: `bubble${u.speaker === firstSpeaker ? "" : " other"}`},
          el("div", {class: "who"}, u.speaker_label, el("span", {class: "line", style: "margin-left:8px"}, u.line)), u.text)
      : el("div", {class: "para"}, el("span", {class: "line"}, u.line), u.text)));
}

// ---------- routing and loading ----------

async function route() {
  if (!state.summary) return;
  const [hashTab, id] = location.hash.replace(/^#/, "").split("/");
  const tab = ["chat", "timeline", "people", "documents"].includes(hashTab) ? hashTab : "chat";
  for (const a of document.querySelectorAll("#tabs a")) a.classList.toggle("active", a.dataset.tab === tab);
  for (const v of document.querySelectorAll(".view")) v.classList.toggle("active", v.id === `view-${tab}`);
  try {
    if (tab === "people") await renderPeople(id);
    else if (tab === "documents") await renderDocuments(id);
    else if (tab === "timeline") await renderTimeline();
    else renderChat();
  } catch (err) {
    $(`#view-${tab}`).replaceChildren(el("p", {class: "error"}, `Error: ${err.message}`));
  }
}

async function loadCase(name) {
  state.caseName = name;
  state.timeline = {from: "", to: "", person: ""};
  $("#case-title").textContent = "Loading…";
  state.summary = await api(`/api/cases/${name}?semantic=${$("#semantic").checked}`);
  const s = state.summary;
  $("#case-title").textContent = s.title;
  $("#case-type").textContent = [s.type, s.reference].filter(Boolean).join(" · ");
  $("#case-summary").textContent = s.summary;
  $("#stats").replaceChildren(...[[s.documents.length, "documents"], [s.people.length, "people"],
    [s.passages, "passages"], [s.dates, "period"]].map(([v, k]) => el("div", {class: "stat"}, el("b", {}, v), el("span", {}, k))));
  await route();
}

async function init() {
  state.config = await api("/api/config");
  const caseSelect = $("#case");
  caseSelect.replaceChildren(...state.config.cases.map(c => el("option", {value: c}, c.replace(/-/g, " "))));
  const backendSelect = $("#backend");
  for (const option of backendSelect.options) {
    const ready = state.config.backends[option.value];
    option.disabled = !ready;
    if (!ready) option.textContent += " (not configured)";
  }
  const router = [...backendSelect.options].find(o => o.value === "openrouter");
  if (state.config.openrouter_model) router.textContent = `${state.config.openrouter_model.replace(/^openai\//, "")} (OpenRouter)`;
  backendSelect.value = state.config.default_backend;
  backendSelect.addEventListener("change", () => { if (location.hash.startsWith("#chat") || !location.hash) renderChat(); });
  caseSelect.addEventListener("change", () => { location.hash = "#chat"; loadCase(caseSelect.value); });
  $("#semantic").addEventListener("change", () => loadCase(state.caseName));
  $("#drawer-close").addEventListener("click", closeDoc);
  $("#scrim").addEventListener("click", closeDoc);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDoc(); });
  window.addEventListener("hashchange", route);
  if (state.config.cases.length) await loadCase(state.config.cases[0]);
}

init().catch(err => { $("#case-title").textContent = `Could not start: ${err.message}`; });
