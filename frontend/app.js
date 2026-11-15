// RAG demo frontend: plain JavaScript, one fetch() per pipeline step.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d", "#475569", "#ea580c", "#4f46e5", "#0d9488"];

// Global index settings (header dropdowns)
const cfg = () => ({ doc_strategy: $("cfg-doc").value, sql_strategy: $("cfg-sql").value, embedder: $("cfg-emb").value });

async function api(path, body) {
  const res = await fetch("/api" + path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
}

// Disable the button while a request runs, show errors inline
async function run(btn, outEl, fn) {
  btn.disabled = true; const label = btn.textContent; btn.textContent = "Working...";
  try { await fn(); }
  catch (e) { outEl.innerHTML = `<p class="error">Error: ${esc(e.message)}</p>`; }
  finally { btn.disabled = false; btn.textContent = label; }
}

// A chunk/result card
function card(r, opts = {}) {
  const c = r.chunk || r;
  const score = opts.scoreKey && r[opts.scoreKey] !== undefined ? `<span class="score">${r[opts.scoreKey]}</span>` : "";
  const extra = opts.extra ? `<div class="meta">${opts.extra(r)}</div>` : "";
  const bar = opts.bar ? `<div class="bar" style="width:${Math.max(2, Math.min(100, opts.bar(r) * 100))}%"></div>` : "";
  return `<div class="card ${opts.cls || ""}">
    <div class="meta"><span>${opts.rank ? `<b>#${opts.rank}</b> ` : ""}${esc(c.id)}</span>${score}</div>
    <div>${esc(c.text)}</div>${extra}${bar}</div>`;
}

// ---------------- Tabs ----------------
document.querySelectorAll("#tabs button").forEach((b) => b.onclick = () => {
  document.querySelectorAll("#tabs button, .tab").forEach((el) => el.classList.remove("active"));
  b.classList.add("active"); $("tab-" + b.dataset.tab).classList.add("active");
});

// ---------------- 0. Data ----------------
async function loadData() {
  const [docs, tables] = await Promise.all([api("/ingest/docs"), api("/ingest/sql")]);
  $("data-docs").innerHTML = docs.map((d) =>
    `<div class="doc"><b>${esc(d.name)}</b> <small>(${d.chars} chars)</small><pre>${esc(d.text)}</pre></div>`).join("");
  $("data-sql").innerHTML = tables.map((t) => {
    const cols = Object.keys(t.rows[0] || {});
    return `<b>${esc(t.table)}</b><table><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${
      t.rows.map((r) => `<tr>${cols.map((c) => `<td>${esc(r[c])}</td>`).join("")}</tr>`).join("")}</table>`;
  }).join("");
  const s = await api("/status");
  $("azure-status").textContent = s.azure_openai_configured ? "Azure OpenAI: connected" : "Azure OpenAI: not configured";
  $("azure-status").classList.toggle("ok", s.azure_openai_configured);
}

// ---------------- 1. Chunking ----------------
const STRATS = { docs: ["paragraph", "sentence", "fixed_size"], sql: ["row", "entity", "schema"] };
function syncChunkControls() {
  const src = $("ch-source").value, cur = $("ch-strategy").value;
  if (!STRATS[src].includes(cur)) $("ch-strategy").innerHTML = STRATS[src].map((s) => `<option>${s}</option>`).join("");
  const st = $("ch-strategy").value;
  document.querySelectorAll(".fixed-only").forEach((e) => e.style.display = st === "fixed_size" ? "" : "none");
  document.querySelectorAll(".sent-only").forEach((e) => e.style.display = st === "sentence" ? "" : "none");
}
$("ch-source").onchange = syncChunkControls;
$("ch-strategy").onchange = syncChunkControls;
$("ch-run").onclick = () => run($("ch-run"), $("ch-out"), async () => {
  const d = await api("/chunk", {
    source: $("ch-source").value, strategy: $("ch-strategy").value,
    size: +$("ch-size").value, overlap: +$("ch-overlap").value, sentences: +$("ch-sent").value,
  });
  $("ch-stats").textContent = `${d.count} chunks · avg ${d.avg_chars} chars · min ${d.min_chars} · max ${d.max_chars}`;
  $("ch-out").innerHTML = d.chunks.map((c) => card(c, { extra: () => `<span>${esc(c.source)}</span><span>${c.text.length} chars</span>` })).join("");
});

// ---------------- 2. Embedding ----------------
function matrix(texts, m) {
  const head = texts.map((_, i) => `<th>S${i + 1}</th>`).join("");
  const rows = m.map((row, i) => `<tr><th>S${i + 1}</th>${row.map((v) => {
    const a = Math.max(0, v); return `<td style="background:rgba(37,99,235,${a * 0.75});color:${a > 0.55 ? "#fff" : "inherit"}">${v.toFixed(2)}</td>`;
  }).join("")}</tr>`).join("");
  return `<table class="matrix"><tr><th></th>${head}</tr>${rows}</table>`;
}
$("em-run").onclick = () => run($("em-run"), $("em-out"), async () => {
  const texts = $("em-texts").value.split("\n").map((t) => t.trim()).filter(Boolean);
  const d = await api("/embed/compare", { texts });
  const legend = `<ol style="padding-left:22px">${texts.map((t) => `<li>${esc(t)}</li>`).join("")}</ol>`;
  $("em-out").innerHTML = ["tfidf", "minilm"].map((k) => `<div class="step">
    <h4>${k === "tfidf" ? "TF-IDF (keywords)" : "MiniLM (semantic)"} · ${d[k].dim} dimensions</h4>
    <small>S1 vector starts: [${d[k].preview.join(", ")}, ...]</small>
    ${matrix(texts, d[k].similarity)}</div>`).join("") + `<div class="step"><h4>Sentences</h4>${legend}</div>`;
});

$("map-run").onclick = () => run($("map-run"), $("map-info"), async () => {
  const d = await api("/embed/map", { ...cfg(), query: $("map-query").value || null, n_clusters: +$("map-k").value });
  const all = d.points.concat(d.query ? [d.query] : []);
  const xs = all.map((p) => p.x), ys = all.map((p) => p.y);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const W = 700, H = 420, P = 30;
  const sx = (x) => P + (x - x0) / (x1 - x0 || 1) * (W - 2 * P);
  const sy = (y) => H - P - (y - y0) / (y1 - y0 || 1) * (H - 2 * P);
  let svg = d.points.map((p, i) => `<circle data-i="${i}" cx="${sx(p.x)}" cy="${sy(p.y)}" r="7" fill="${COLORS[p.cluster % COLORS.length]}" fill-opacity=".8"><title>${esc(p.id)}</title></circle>`).join("");
  svg += d.centroids.map(([x, y], c) => `<text x="${sx(x)}" y="${sy(y)}" font-size="18" text-anchor="middle" dominant-baseline="middle" fill="${COLORS[c % COLORS.length]}">✕</text>`).join("");
  if (d.query) svg += `<g><circle cx="${sx(d.query.x)}" cy="${sy(d.query.y)}" r="10" fill="none" stroke="#111" stroke-width="3"/><text x="${sx(d.query.x) + 14}" y="${sy(d.query.y) + 4}" font-size="12" font-weight="700">query</text></g>`;
  $("map").innerHTML = svg;
  $("map-info").innerHTML = `<p>${d.points.length} chunks, ${d.dim}-dimensional vectors squashed to 2D.
    Colours = k-means clusters, ✕ = cluster centre${d.query ? ", black ring = your question" : ""}. Click a dot to read it.</p><div id="map-pick"></div>`;
  $("map").querySelectorAll("circle[data-i]").forEach((el) => el.onclick = () => {
    const p = d.points[+el.dataset.i];
    $("map-pick").innerHTML = card(p, { extra: () => `<span>cluster ${p.cluster}</span><span>${esc(p.source)}</span>` });
  });
});

// ---------------- 3. Retrieval ----------------
$("rt-run").onclick = () => run($("rt-run"), $("rt-out"), async () => {
  const d = await api("/retrieve/compare", { ...cfg(), query: $("rt-query").value, top_n: +$("rt-n").value });
  $("rt-out").innerHTML = ["dense", "bm25", "hybrid", "kmeans"].map((m) => {
    const r = d[m]; let info = "";
    if (m === "kmeans") info = `<p><small>Searched cluster ${r.info.chosen_cluster}: ${r.info.searched_chunks} of ${r.info.total_chunks} chunks. Cluster sizes: [${r.info.cluster_sizes.join(", ")}]</small></p>`;
    if (!r.results.length) info += "<p><small>No keyword matches.</small></p>";
    const max = Math.max(...r.results.map((x) => x.score), 1e-9);
    return `<div><h3>${m}</h3>${info}${r.results.map((x, i) => card(x, {
      rank: i + 1, scoreKey: "score", bar: (y) => y.score / max,
      extra: m === "hybrid" ? (y) => `<span>dense rank ${y.dense_rank}</span><span>bm25 rank ${y.bm25_rank ?? "-"}</span>` : null,
    })).join("")}</div>`;
  }).join("");
});

// ---------------- 4. Re-ranking ----------------
$("rr-run").onclick = () => run($("rr-run"), $("rr-before"), async () => {
  const d = await api("/rerank", { ...cfg(), query: $("rr-query").value, retriever: $("rr-ret").value,
    reranker: $("rr-method").value, top_n: +$("rr-n").value, top_k: +$("rr-k").value });
  const kept = new Set(d.after.map((x) => x.index));
  $("rr-before").innerHTML = d.before.map((x, i) => card(x, { rank: i + 1, scoreKey: "score", cls: kept.has(x.index) ? "kept" : "dropped" })).join("");
  $("rr-after").innerHTML = d.after.map((x, i) => {
    const from = d.before.findIndex((b) => b.index === x.index) + 1;
    return card(x, { rank: i + 1, scoreKey: "rerank_score", cls: "kept", extra: (y) => {
      let s = `<span>was #${from}</span>`;
      if (y.relevance !== undefined) s += `<span>relevance ${y.relevance} · redundancy ${y.redundancy}</span>`;
      if (y.cluster !== undefined) s += `<span>best of cluster ${y.cluster} (${y.cluster_size} chunks)</span>`;
      return s;
    } });
  }).join("");
});

// ---------------- 5. Ask ----------------
$("ask-run").onclick = () => run($("ask-run"), $("ask-out"), async () => {
  const d = await api("/ask", { ...cfg(), query: $("ask-query").value, retriever: $("ask-ret").value,
    reranker: $("ask-rr").value, top_n: 8, top_k: 3 });
  $("ask-out").innerHTML = renderPipeline(d);
});

// Shared by the Ask tab and the Your-document tab: answer + every pipeline step
function renderPipeline(d) {
  const g = d.generation, t = d.timings;
  return `
    <div class="step"><h4>Answer ${g.llm ? `<small>(${esc(g.llm)})</small>` : ""}</h4>
      ${g.answer ? `<div class="answer">${esc(g.answer)}</div>` : ""}
      ${g.error ? `<p class="error">${esc(g.error)}</p>` : ""}</div>
    <div class="step"><h4>1-2. Index: ${d.index.chunks} chunks, ${d.index.dim}-dim vectors <small>(${t.index_ms} ms, cached after first build)</small></h4></div>
    <div class="step"><h4>3. Retrieved top-${d.retrieved.results.length} with ${esc(d.retrieved.method)} <small>(${t.retrieve_ms} ms)</small></h4>
      ${d.retrieved.results.map((x, i) => card(x, { rank: i + 1, scoreKey: "score" })).join("")}</div>
    <div class="step"><h4>4. Re-ranked to top-${d.reranked.length} <small>(${t.rerank_ms} ms)</small></h4>
      ${d.reranked.map((x, i) => card(x, { rank: i + 1, scoreKey: "rerank_score", cls: "kept" })).join("")}</div>
    <div class="step"><h4>5. Prompt sent to the LLM <small>(${t.generate_ms} ms)</small></h4>
      <pre>SYSTEM: ${esc(g.system_prompt)}\n\nUSER:\n${esc(g.prompt)}</pre></div>`;
}

// ---------------- 6. Your document ----------------
const upCfg = () => ({ doc_id: $("up-doc").value, strategy: $("up-strategy").value,
  size: +$("up-size").value, overlap: +$("up-overlap").value, embedder: $("up-emb").value });

function syncUploadControls() {
  document.querySelectorAll(".up-fixed").forEach((e) => e.style.display = $("up-strategy").value === "fixed_size" ? "" : "none");
}
$("up-strategy").onchange = syncUploadControls;

function showDocInfo(doc) {
  $("up-info").innerHTML = doc ? `<div class="card"><div class="meta"><b>${esc(doc.name)}</b>
    <span>${doc.pages} page${doc.pages > 1 ? "s" : ""} · ${doc.chars.toLocaleString()} characters</span></div>
    <small>${esc(doc.preview)}${doc.chars > doc.preview.length ? " ..." : ""}</small></div>` : "";
}

async function refreshUploads(selectId) {
  const docs = await api("/upload");
  $("up-doc").innerHTML = docs.length
    ? docs.map((d) => `<option value="${d.doc_id}">${esc(d.name)}</option>`).join("")
    : `<option value="">(upload a file first)</option>`;
  if (selectId) $("up-doc").value = selectId;
  showDocInfo(docs.find((d) => d.doc_id === $("up-doc").value));
  return docs;
}
$("up-doc").onchange = () => refreshUploads($("up-doc").value);

$("up-run").onclick = () => run($("up-run"), $("up-info"), async () => {
  const f = $("up-file").files[0];
  if (!f) throw new Error("Choose a PDF, Word or text file first.");
  const form = new FormData();
  form.append("file", f);
  const res = await fetch("/api/upload", { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Upload failed");
  await refreshUploads(data.doc_id);
  $("up-out").innerHTML = "";
});

$("up-chunks").onclick = () => run($("up-chunks"), $("up-out"), async () => {
  if (!$("up-doc").value) throw new Error("Upload a document first.");
  const d = await api("/upload/chunks", upCfg());
  $("up-out").innerHTML = `<p class="stats">${d.count} chunks · avg ${d.avg_chars} chars</p><div class="cards">${
    d.chunks.map((c) => card(c, { extra: () => `<span>page ${c.metadata.page}</span><span>${c.text.length} chars</span>` })).join("")}</div>`;
});

$("up-ask").onclick = () => run($("up-ask"), $("up-out"), async () => {
  if (!$("up-doc").value) throw new Error("Upload a document first.");
  if (!$("up-query").value.trim()) throw new Error("Type a question first.");
  const d = await api("/upload/ask", { ...upCfg(), query: $("up-query").value,
    retriever: $("up-ret").value, reranker: $("up-rr").value, top_n: 8, top_k: 4 });
  $("up-out").innerHTML = renderPipeline(d);
});
$("up-query").onkeydown = (e) => { if (e.key === "Enter") $("up-ask").click(); };

syncUploadControls();
refreshUploads().catch(() => {});
syncChunkControls();
loadData().catch((e) => $("data-docs").innerHTML = `<p class="error">${esc(e.message)}</p>`);
