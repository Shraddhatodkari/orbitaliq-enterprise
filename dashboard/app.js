"use strict";

/* OrbitalIQ Enterprise Dashboard — a thin presentation layer over the
 * existing FastAPI backend. This file never recomputes anything: every
 * number, label, and status shown here is read verbatim from an API
 * response. Where this file composes prose (the Executive Strategy View),
 * it only templates already-computed, already-labeled fields into hedged
 * language — it does not infer, score, or fabricate anything new. */

const API_BASE = "/api/v1";

// ---------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------

function $(id) { return document.getElementById(id); }

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
  }
  (children || []).forEach((c) => {
    if (c === null || c === undefined) return;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  });
  return node;
}

function escapeHtml(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function fmtNum(v, digits) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toLocaleString(undefined, { maximumFractionDigits: digits ?? 2, minimumFractionDigits: 0 });
}

function fmtPercent(fraction, digits) {
  if (fraction === null || fraction === undefined) return "—";
  return (fraction * 100).toFixed(digits ?? 1) + "%";
}

function fmtUsd(v) {
  if (v === null || v === undefined) return "—";
  const abs = Math.abs(v);
  if (abs >= 1e9) return "$" + (v / 1e9).toFixed(2) + "B";
  if (abs >= 1e6) return "$" + (v / 1e6).toFixed(2) + "M";
  if (abs >= 1e3) return "$" + (v / 1e3).toFixed(1) + "K";
  return "$" + fmtNum(v, 0);
}

function fmtMetricValue(metric) {
  if (metric.status !== "AVAILABLE" || metric.current_value === null || metric.current_value === undefined) return "—";
  if (metric.unit === "USD") return fmtUsd(metric.current_value);
  if (metric.unit === "percent") return fmtPercent(metric.current_value, 1);
  return fmtNum(metric.current_value, 3); // ratio
}

function fmtDate(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch (e) {
    return iso;
  }
}

function badgeClassForLevel(level) {
  switch (level) {
    case "AGGRESSIVE_EXPANSION": return "badge-bad";
    case "STRONG": return "badge-warn";
    case "EMERGING": return "badge-accent";
    case "STABLE": return "badge-neutral";
    case "HIGH": return "badge-bad";
    case "MODERATE": return "badge-warn";
    case "LOW": return "badge-neutral";
    case "INSUFFICIENT": return "badge-neutral";
    case "INSUFFICIENT_DATA": return "badge-neutral";
    default: return "badge-neutral";
  }
}

// A momentum_level of INSUFFICIENT_DATA means the backend could not
// legitimately calculate a score for this assessment (every underlying
// financial/satellite signal was a live-mode fallback or unavailable --
// see core/intelligence_engine.py) -- the numeric momentum_score on that
// record is a fixed 0.0 placeholder, never a real score, so it must never
// be displayed as if it were one.
function isInsufficientData(a) {
  return a && a.momentum_level === "INSUFFICIENT_DATA";
}

function fmtMomentumScore(a) {
  return isInsufficientData(a) ? "Insufficient Data" : fmtNum(a.momentum_score, 1) + " / 100";
}

// --- Output philosophy (task #73): the qualitative Competitive Expansion
// Signal level is the headline every view leads with; the 0-100 composite
// index is real, disclosed supporting detail -- never hidden, but never
// the primary framing. See streamlit_app/components.py for the same
// rationale on the Streamlit side. ---------------------------------------
const SATELLITE_LIMITATION_CAVEAT =
  "Satellite evidence is directional only — large-scale land/construction change detected from public " +
  "VIIRS/MODIS imagery — and does not confirm the specific use, ownership, or purpose of any change.";

function fmtCompositeIndex(a) {
  return isInsufficientData(a) ? "no composite index — insufficient live data" : `composite index ${fmtNum(a.momentum_score, 1)}/100`;
}

function badgeClassForOutcome(outcome) {
  switch (outcome) {
    case "SIGNAL_CONVERGENCE": return "badge-bad";
    case "PARTIAL_SIGNAL_CONVERGENCE": return "badge-warn";
    case "SIGNAL_CONFLICT": return "badge-warn";
    case "INSUFFICIENT_EVIDENCE": return "badge-neutral";
    default: return "badge-neutral";
  }
}

function badgeClassForNarrativeStatus(status) {
  switch (status) {
    case "AI_GENERATED_GROUNDED": return "badge-accent";
    case "AI_GENERATED_BLOCKED_REPLACED_WITH_DETERMINISTIC": return "badge-warn";
    case "DETERMINISTIC_GROUNDED_TEMPLATE": return "badge-neutral";
    default: return "badge-neutral";
  }
}

function badgeClassForWatchlistTier(tier) {
  switch (tier) {
    case "HIGH_SIGNAL": return "badge-bad";
    case "REVIEW": return "badge-warn";
    case "WATCH": return "badge-accent";
    default: return "badge-neutral";
  }
}

// Mirrors the canonical 5-state contract in
// core/data_quality_state.py::classify_state (and
// streamlit_app/components.py::source_status, which documents the full
// rule): *_live / derived:* are real values ("ok"); cached_real:* is a
// real but explicitly stale value, and synthetic_fallback:*/
// offline_demo_mode are disclosed non-production labels (both "warn");
// error:*/NOT_AVAILABLE/insufficient_data* are a genuine absence of a
// real value ("bad").
function sourceBadge(source) {
  if (!source) return el("span", { class: "badge badge-neutral" }, ["NOT_AVAILABLE"]);
  const s = String(source);
  if (s === "offline_demo_mode" || s.startsWith("synthetic_fallback")) return el("span", { class: "badge badge-warn" }, [s.toUpperCase()]);
  if (s.startsWith("cached_real")) return el("span", { class: "badge badge-warn" }, [s.toUpperCase()]);
  if (s.endsWith("_live") || s.startsWith("derived")) return el("span", { class: "badge badge-ok" }, [s.toUpperCase()]);
  if (s.startsWith("error") || s === "NOT_AVAILABLE" || s.includes("insufficient")) return el("span", { class: "badge badge-bad" }, [s.toUpperCase()]);
  return el("span", { class: "badge badge-neutral" }, [s.toUpperCase()]);
}

// Renders any shape of FastAPI's `detail` field into a single readable
// line. Three shapes are expected in practice:
//   1. A plain string (404s, simple validation messages).
//   2. Our own structured pipeline/persistence/response-serialization
//      error object — {stage, message, source, ticker, request_id} — see
//      api/routes_assessment.py::_error_detail. This is what turns a bare
//      "500: Internal Server Error" into something you can actually act
//      on: which stage failed, the real exception text, and a request id
//      to find it in the server log.
//   3. FastAPI/Pydantic's automatic 422 validation-error array
//      ([{loc, msg, type}, ...]) — previously rendered as the useless
//      literal "[object Object]" because it was never specifically
//      handled.
function formatApiErrorDetail(detail, fallback) {
  if (detail == null) return fallback;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((e) => {
        if (e && typeof e === "object") {
          const loc = Array.isArray(e.loc) ? e.loc.filter((p) => p !== "body").join(".") : "";
          return loc ? `${loc}: ${e.msg}` : e.msg || JSON.stringify(e);
        }
        return String(e);
      })
      .join("; ");
  }
  if (typeof detail === "object" && ("stage" in detail || "message" in detail)) {
    const parts = [];
    if (detail.stage) parts.push(`stage=${detail.stage}`);
    if (detail.message) parts.push(detail.message);
    if (detail.source) parts.push(`source=${detail.source}`);
    if (detail.request_id) parts.push(`request_id=${detail.request_id}`);
    return parts.join(" | ") || fallback;
  }
  try {
    return JSON.stringify(detail);
  } catch (e) {
    return fallback;
  }
}

async function apiFetch(path, opts) {
  const resp = await fetch(API_BASE + path, Object.assign({ headers: { "Content-Type": "application/json" } }, opts || {}));
  let body = null;
  try { body = await resp.json(); } catch (e) { /* no body — a bare 500 with no JSON, e.g. an unhandled crash outside our own error handling */ }
  if (!resp.ok) {
    const detail = formatApiErrorDetail(body && body.detail, resp.statusText);
    const err = new Error(`${resp.status}: ${detail}`);
    err.status = resp.status;
    err.detail = body && body.detail;
    throw err;
  }
  return body;
}

// ---------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------

const state = {
  currentAssessment: null,
  assessments: [],
  selectedForCompare: new Set(),
};

function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
  $("view-" + name).classList.add("active");
  document.querySelector(`.nav-item[data-view="${name}"]`).classList.add("active");
  if (name === "assessments") loadAssessments();
  if (name === "watchlist") loadWatchlist();
}

document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", () => showView(item.getAttribute("data-view")));
});

// ---------------------------------------------------------------------
// New Assessment
// ---------------------------------------------------------------------

$("assessment-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const btn = $("submit-btn");
  const hint = $("submit-hint");
  const payload = {
    ticker: form.ticker.value.trim(),
    company_name: form.company_name.value.trim(),
    facility_name: form.facility_name.value.trim(),
    industry: form.industry.value.trim() || "general",
    latitude: parseFloat(form.latitude.value),
    longitude: parseFloat(form.longitude.value),
  };
  if (form.market_cap_usd.value) payload.market_cap_usd = parseFloat(form.market_cap_usd.value);

  btn.disabled = true;
  hint.textContent = "Running ingestion → vision → scoring → report → watchlist…";
  try {
    const result = await apiFetch("/assessments", { method: "POST", body: JSON.stringify(payload) });
    hint.textContent = "Done.";
    state.currentAssessment = result;
    renderDashboard(result);
    showView("dashboard");
    form.reset();
  } catch (err) {
    hint.textContent = "";
    alert("Assessment failed: " + err.message);
  } finally {
    btn.disabled = false;
  }
});

// ---------------------------------------------------------------------
// Executive Dashboard rendering
// ---------------------------------------------------------------------

function renderDashboard(a) {
  $("dashboard-empty").style.display = "none";
  const root = $("dashboard-content");
  root.style.display = "block";
  root.innerHTML = "";

  root.appendChild(buildHeaderPanel(a));
  root.appendChild(buildKpiRow(a));
  root.appendChild(buildNarrativePanel(a));
  root.appendChild(buildScoreBreakdownPanel(a));
  root.appendChild(buildFinancialPanel(a));
  root.appendChild(buildSatellitePanel(a));
  root.appendChild(buildConvergencePanel(a));
  root.appendChild(buildDataQualityPanel(a));
  root.appendChild(buildEvidencePanel(a));
  root.appendChild(buildStrategyPanel(a));
  root.appendChild(buildHistoricalPanel(a));
  root.appendChild(buildAgentTracePanel(a));
}

function buildHeaderPanel(a) {
  const wrap = el("div", { class: "page-header" }, [
    el("div", {}, [
      el("h1", { class: "page-title" }, [`${a.company_name} (${a.ticker})`]),
      el("div", { class: "page-subtitle" }, [`${a.facility_name} · ${a.industry} · assessed ${fmtDate(a.created_at)}`]),
    ]),
    el("div", { class: "tag-list" }, [
      el("span", { class: "badge " + badgeClassForLevel(a.momentum_level) }, [a.momentum_level]),
      a.watchlist_tier !== "NONE" ? el("span", { class: "badge " + badgeClassForWatchlistTier(a.watchlist_tier) }, ["WATCHLIST: " + a.watchlist_tier]) : null,
    ]),
  ]);
  return wrap;
}

function buildKpiRow(a) {
  const dq = a.data_quality || {};
  const row = el("div", {}, [
    el("div", { class: "kpi-row" }, [
      el("div", { class: "kpi-card" }, [
        el("div", { class: "kpi-label" }, ["Competitive Expansion Signal"]),
        el("div", { class: "kpi-value" }, [a.momentum_level]),
        el("div", { class: "kpi-sub" }, [fmtCompositeIndex(a)]),
      ]),
      el("div", { class: "kpi-card" }, [
        el("div", { class: "kpi-label" }, ["Data Completeness"]),
        el("div", { class: "kpi-value" }, [fmtNum(dq.completeness_pct, 0) + "%"]),
        el("div", { class: "kpi-sub" }, [`${dq.live_signals ?? "—"}/${dq.total_signals ?? "—"} signals live`]),
      ]),
      el("div", { class: "kpi-card" }, [
        el("div", { class: "kpi-label" }, ["Live Source Count"]),
        el("div", { class: "kpi-value" }, [String(dq.source_count ?? "—")]),
        el("div", { class: "kpi-sub" }, [dq.strict_mode ? "Strict real-data mode" : "Standard mode"]),
      ]),
      el("div", { class: "kpi-card" }, [
        el("div", { class: "kpi-label" }, ["Narrative"]),
        el("div", {}, [el("span", { class: "badge " + badgeClassForNarrativeStatus(a.narrative_status) }, [a.narrative_status.replace(/_/g, " ")])]),
        el("div", { class: "kpi-sub" }, [a.llm_generated ? "NVIDIA NIM" : "Deterministic template"]),
      ]),
    ]),
    el("div", { class: "panel-note", style: "margin-top:6px;" }, [SATELLITE_LIMITATION_CAVEAT]),
  ]);
  return row;
}

function buildNarrativePanel(a) {
  const children = [
    el("div", { class: "panel-title" }, ["Executive Summary"]),
    el("p", {}, [a.narrative_report]),
  ];
  if (a.narrative_grounding) {
    const g = a.narrative_grounding;
    children.push(
      el("div", { class: "panel-note" }, [
        `Grounding critic: ${g.status} — ${g.checked_numbers} numeric claim(s) checked` +
          (g.flagged_values && g.flagged_values.length ? `, unsupported: ${g.flagged_values.join(", ")}` : "."),
      ])
    );
  }
  if (a.recommended_actions && a.recommended_actions.length) {
    children.push(el("hr", { class: "section-divider" }));
    children.push(el("div", { class: "panel-title" }, ["Recommended Next Steps"]));
    children.push(el("ul", {}, a.recommended_actions.map((act) => el("li", {}, [act]))));
  }
  return el("div", { class: "panel" }, children);
}

function buildScoreBreakdownPanel(a) {
  const sb = a.signal_breakdown || {};
  const rows = Object.entries(sb).map(([k, v]) =>
    el("tr", {}, [
      el("td", {}, [k.replace(/_/g, " ")]),
      el("td", { class: "num" }, [fmtNum(v, 1) + " / 100"]),
    ])
  );
  const children = [
    el("div", { class: "panel-title" }, ["Score Breakdown — Why This Signal?"]),
    el("table", {}, [el("tbody", {}, rows)]),
  ];

  const brk = a.score_breakdown;
  if (brk) {
    const w = brk.weights || {};
    children.push(
      el("div", { class: "convergence-grid" }, [
        el("div", { class: "convergence-tile" }, [
          el("div", { class: "convergence-label" }, ["Financial Contribution"]),
          el("div", { class: "convergence-value" }, [
            brk.financial_contribution === null || brk.financial_contribution === undefined
              ? "—" : fmtNum(brk.financial_contribution, 1),
          ]),
          el("div", { class: "muted" }, [`weight ${w.financial_weight ?? "—"}`]),
        ]),
        el("div", { class: "convergence-tile" }, [
          el("div", { class: "convergence-label" }, ["Satellite Contribution"]),
          el("div", { class: "convergence-value" }, [
            brk.satellite_contribution === null || brk.satellite_contribution === undefined
              ? "—" : fmtNum(brk.satellite_contribution, 1),
          ]),
          el("div", { class: "muted" }, [`weight ${w.satellite_weight ?? "—"}`]),
        ]),
        el("div", { class: "convergence-tile" }, [
          el("div", { class: "convergence-label" }, ["Convergence Bonus"]),
          el("div", { class: "convergence-value" }, [`+${fmtNum(brk.convergence_bonus, 1)}`]),
          el("div", { class: "muted" }, [`max ${w.convergence_bonus_max ?? "—"}`]),
        ]),
        el("div", { class: "convergence-tile" }, [
          el("div", { class: "convergence-label" }, ["Evidence Coverage"]),
          el("div", { class: "convergence-value" }, [`${fmtNum(brk.data_confidence_pct, 0)}%`]),
          el("div", { class: "muted" }, ["share of tracked financial/satellite signals LIVE/CACHED_REAL/DERIVED — not overall public-evidence completeness"]),
        ]),
      ])
    );
    if (brk.contradictory_signals) {
      children.push(
        el("div", { class: "panel-note", style: "color:var(--bad,#c0392b);" }, [
          `⚠ Contradictory signals (${brk.convergence_outcome}): ${brk.convergence_explanation || ""}`,
        ])
      );
    }
    const explanations = brk.signal_explanations || [];
    if (explanations.length) {
      children.push(el("div", { class: "panel-title" }, ["Per-signal drill-down"]));
      children.push(
        el("table", {}, [
          el("thead", {}, [
            el("tr", {}, ["Signal", "Trusted", "Value", "Source", "Reason", "What it measures"].map((h) => el("th", {}, [h]))),
          ]),
          el(
            "tbody",
            {},
            explanations.map((e) =>
              el("tr", {}, [
                el("td", {}, [String(e.signal).replace(/_/g, " ")]),
                el("td", {}, [e.trusted ? el("span", { class: "badge badge-ok" }, ["✓"]) : el("span", { class: "badge badge-neutral" }, ["excluded"])]),
                el("td", { class: "num" }, [e.value_0_100 === null || e.value_0_100 === undefined ? "—" : fmtNum(e.value_0_100, 1)]),
                el("td", {}, [e.source || "—"]),
                el("td", { class: "muted italic" }, [e.reason || ""]),
                el("td", { class: "muted" }, [e.description || ""]),
              ])
            )
          ),
        ])
      );
    }
  }

  children.push(
    el("div", { class: "panel-note" }, [
      "Composite = 0.6 × financial composite (avg of revenue/R&D/capex growth signals) + 0.4 × satellite " +
      "composite (avg of construction-expansion/vegetation-clearing signals), plus up to +8 convergence " +
      "bonus points when both composites independently clear 50/100 — the existing, unmodified deterministic " +
      "formula in core/intelligence_engine.py. Not re-weighted for this dashboard.",
    ])
  );
  return el("div", { class: "panel" }, children);
}

function metricRow(m) {
  const statusBadge = m.status === "AVAILABLE"
    ? el("span", { class: "badge badge-ok" }, ["AVAILABLE"])
    : el("span", { class: "badge badge-neutral" }, ["INSUFFICIENT DATA"]);
  return el("tr", {}, [
    el("td", {}, [m.label]),
    el("td", { class: "num" }, [fmtMetricValue(m)]),
    el("td", { class: "num" }, [m.yoy_change_pct !== null && m.yoy_change_pct !== undefined ? fmtPercent(m.yoy_change_pct, 1) : "—"]),
    el("td", {}, [m.fiscal_period || "—"]),
    el("td", {}, [m.xbrl_concept ? el("span", { class: "muted" }, [m.xbrl_concept]) : "—"]),
    el("td", {}, [statusBadge]),
    el("td", { class: "muted italic" }, [m.status === "AVAILABLE" ? "" : (m.note || "")]),
  ]);
}

function buildFinancialPanel(a) {
  const fp = a.financial_profile;
  if (!fp) {
    return el("div", { class: "panel" }, [
      el("div", { class: "panel-title" }, ["Real Financial Intelligence (SEC EDGAR XBRL)"]),
      el("div", { class: "unavailable-box" }, [
        "Not available for this assessment — either live data mode is disabled for this deployment, or the " +
        "ticker could not be resolved against SEC EDGAR's company directory. No figures are fabricated to fill this gap.",
      ]),
    ]);
  }
  const categories = ["growth", "profitability", "balance_sheet", "investment_intensity"];
  const categoryLabels = {
    growth: "Growth", profitability: "Profitability", balance_sheet: "Balance Sheet", investment_intensity: "Investment Intensity",
  };
  const byCategory = {};
  Object.values(fp.metrics).forEach((m) => {
    byCategory[m.category] = byCategory[m.category] || [];
    byCategory[m.category].push(m);
  });

  const sections = categories
    .filter((c) => byCategory[c] && byCategory[c].length)
    .map((c) =>
      el("div", {}, [
        el("div", { style: "font-size:11.5px;font-weight:600;color:var(--text-muted);margin:12px 0 6px 0;" }, [categoryLabels[c]]),
        el("table", {}, [
          el("thead", {}, [el("tr", {}, ["Metric", "Value", "YoY", "Fiscal Period", "XBRL Concept", "Status", "Note"].map((h) => el("th", {}, [h])))]),
          el("tbody", {}, byCategory[c].map(metricRow)),
        ]),
      ])
    );

  return el("div", { class: "panel" }, [
    el("div", { class: "panel-title" }, [
      "Real Financial Intelligence (SEC EDGAR XBRL)",
      el("span", { class: "hint" }, [fp.filing_source_url ? el("a", { href: fp.filing_source_url, target: "_blank", rel: "noopener" }, ["Filing source ↗"]) : ""]),
    ]),
    ...sections,
    el("div", { class: "panel-note" }, [
      `CIK ${fp.cik || "—"} · retrieved ${fmtDate(fp.retrieved_at)}. Every metric above is either a genuine ` +
      "value disclosed in a real SEC filing (with its exact XBRL concept and fiscal period) or explicitly " +
      "Insufficient Data — never a synthetic substitute.",
    ]),
  ]);
}

function buildSatellitePanel(a) {
  const sv = a.satellite_visual;
  if (!sv) {
    return el("div", { class: "panel" }, [
      el("div", { class: "panel-title" }, ["Satellite Change Detection View (NASA GIBS)"]),
      el("div", { class: "unavailable-box" }, [
        "Not available for this assessment — live data mode is disabled, no live imagery was returned, or " +
        "strict real-data mode suppressed a non-live fallback image rather than displaying it as if it were real.",
      ]),
    ]);
  }
  return el("div", { class: "panel" }, [
    el("div", { class: "panel-title" }, ["Satellite Change Detection View (NASA GIBS)"]),
    el("div", { class: "satellite-grid" }, [
      el("div", { class: "satellite-tile" }, [
        el("img", { src: sv.current_image_png, alt: "current" }),
        el("div", { class: "caption" }, [`Current — ${sv.current_observation_date || "—"}`]),
      ]),
      el("div", { class: "satellite-tile" }, [
        el("img", { src: sv.prior_image_png, alt: "prior" }),
        el("div", { class: "caption" }, [`Prior (−${sv.lookback_days}d) — ${sv.prior_observation_date || "—"}`]),
      ]),
      el("div", { class: "satellite-tile" }, [
        el("img", { src: sv.diff_image_png, alt: "change heatmap" }),
        el("div", { class: "caption" }, ["Change heatmap"]),
      ]),
    ]),
    el("div", { class: "panel-note" }, [
      `${sv.source_name} · ${sv.satellite_product || "—"} · ${fmtNum(sv.resolution_m_per_pixel, 0)} m/pixel effective ` +
      `resolution · (${fmtNum(sv.coordinates.latitude, 4)}, ${fmtNum(sv.coordinates.longitude, 4)}). ${sv.change_score_note}`,
    ]),
  ]);
}

function buildConvergencePanel(a) {
  const c = a.convergence;
  if (!c) return el("div", {});
  return el("div", { class: "panel" }, [
    el("div", { class: "panel-title" }, ["Strategic Signal Convergence"]),
    el("div", { class: "convergence-grid" }, [
      el("div", { class: "convergence-tile" }, [
        el("div", { class: "label" }, ["Financial"]),
        el("span", { class: "badge " + badgeClassForLevel(c.financial_level) }, [c.financial_level]),
      ]),
      el("div", { class: "convergence-tile" }, [
        el("div", { class: "label" }, ["Physical (Satellite)"]),
        el("span", { class: "badge " + badgeClassForLevel(c.physical_level) }, [c.physical_level]),
      ]),
      el("div", { class: "convergence-tile" }, [
        el("div", { class: "label" }, ["External / Public Disclosure"]),
        el("span", { class: "badge " + badgeClassForLevel(c.external_level) }, [c.external_level]),
      ]),
      el("div", { class: "convergence-tile" }, [
        el("div", { class: "label" }, ["Outcome"]),
        el("span", { class: "badge " + badgeClassForOutcome(c.outcome) }, [c.outcome.replace(/_/g, " ")]),
      ]),
    ]),
    el("p", { class: "hint" }, [c.explanation]),
    el("div", { class: "panel-note" }, [
      "No independent external/public-disclosure evidence source is wired into the deterministic pipeline yet " +
      "(see the Multi-Source Competitive Intelligence roadmap item) — honestly reported as INSUFFICIENT rather " +
      "than inferred from the other two buckets.",
    ]),
  ]);
}

const CATEGORY_LABELS = { financial: "Financial", satellite: "Satellite", public_evidence: "Public Evidence" };

function buildCategoryRollup(categories) {
  if (!categories || !Object.keys(categories).length) return null;
  const cards = Object.entries(categories).map(([key, cat]) => {
    const label = CATEGORY_LABELS[key] || key;
    if (!cat.total_signals) {
      return el("div", { class: "kpi-card" }, [
        el("div", { class: "kpi-label" }, [label]),
        el("div", { class: "kpi-value" }, ["N/A"]),
        el("div", { class: "kpi-sub" }, [cat.note || "No signals defined for this category."]),
      ]);
    }
    return el("div", { class: "kpi-card" }, [
      el("div", { class: "kpi-label" }, [label]),
      el("div", { class: "kpi-value" }, [fmtNum(cat.completeness_pct, 0) + "%"]),
      el("div", { class: "kpi-sub" }, [`${cat.live_signals}/${cat.total_signals} live`]),
    ]);
  });
  return el("div", { style: "margin-top:6px;" }, [
    el("div", { class: "hint", style: "margin-bottom:4px;" }, ["Per-category rollup:"]),
    el("div", { class: "kpi-row" }, cards),
  ]);
}

function buildDataQualityPanel(a) {
  const dq = a.data_quality;
  if (!dq) return el("div", {});
  const fieldTag = (f, cls) => el("span", { class: "badge " + cls }, [f]);
  const freshnessDays = dq.freshness_days || {};
  const daysSuffix = (key) => (freshnessDays[key] !== undefined && freshnessDays[key] !== null ? ` (${freshnessDays[key]}d ago)` : "");
  const categoryRollup = buildCategoryRollup(dq.categories);
  return el("div", { class: "panel" }, [
    el("div", { class: "panel-title" }, ["Data Quality Center"]),
    el("div", { class: "kpi-row" }, [
      el("div", { class: "kpi-card" }, [
        el("div", { class: "kpi-label" }, ["Completeness"]),
        el("div", { class: "kpi-value" }, [fmtNum(dq.completeness_pct, 0) + "%"]),
        el("div", { class: "kpi-sub" }, ["tracked financial/satellite signals only — public evidence isn't wired in"]),
      ]),
      el("div", { class: "kpi-card" }, [
        el("div", { class: "kpi-label" }, ["Evidence Coverage"]),
        el("div", { class: "kpi-value" }, [fmtNum(dq.evidence_coverage_pct, 0) + "%"]),
        el("div", { class: "kpi-sub" }, ["live + disclosed fallback, tracked signals only"]),
      ]),
      el("div", { class: "kpi-card" }, [
        el("div", { class: "kpi-label" }, ["Overall Rating"]),
        el("div", {}, [el("span", { class: "badge " + badgeClassForLevel(dq.overall_rating) }, [dq.overall_rating || "—"])]),
        el("div", { class: "kpi-sub" }, ["rates only the tracked signals above, not the full public-evidence picture"]),
      ]),
      el("div", { class: "kpi-card" }, [el("div", { class: "kpi-label" }, ["Live Signals"]), el("div", { class: "kpi-value" }, [`${dq.live_signals ?? "—"}/${dq.total_signals ?? "—"}`])]),
    ]),
    categoryRollup,
    el("div", { style: "margin-top:6px;" }, [
      el("div", { class: "hint", style: "margin-bottom:4px;" }, ["Fallback fields:"]),
      el("div", { class: "tag-list" }, (dq.fallback_fields || []).length ? dq.fallback_fields.map((f) => fieldTag(f, "badge-warn")) : [el("span", { class: "muted" }, ["none"])]),
    ]),
    el("div", { style: "margin-top:8px;" }, [
      el("div", { class: "hint", style: "margin-bottom:4px;" }, ["Unavailable fields:"]),
      el("div", { class: "tag-list" }, (dq.unavailable_fields || []).length ? dq.unavailable_fields.map((f) => fieldTag(f, "badge-bad")) : [el("span", { class: "muted" }, ["none"])]),
    ]),
    el("div", { class: "panel-note" }, [
      `Fiscal period: ${dq.freshness && dq.freshness.most_recent_fiscal_period || "—"} · Satellite current: ` +
      `${dq.freshness && dq.freshness.satellite_current_observation_date || "—"}${daysSuffix("satellite_current_days_ago")} · Satellite prior: ` +
      `${dq.freshness && dq.freshness.satellite_prior_observation_date || "—"}${daysSuffix("satellite_prior_days_ago")} · Assessed at: ` +
      `${dq.freshness && dq.freshness.assessed_at || "—"}${daysSuffix("assessed_at_days_ago")}.` +
      (dq.strict_mode ? " Strict real-data mode is ON: no synthetic value is ever substituted for a missing metric." : ""),
    ]),
  ]);
}

function buildEvidencePanel(a) {
  const items = a.evidence || [];
  if (!items.length) {
    return el("div", { class: "panel" }, [
      el("div", { class: "panel-title" }, ["Evidence / Audit Trail"]),
      el("div", { class: "empty-state" }, ["No evidence items recorded for this assessment."]),
    ]);
  }
  const rows = items.map((it) =>
    el("tr", {}, [
      el("td", {}, [it.claim]),
      el("td", {}, [it.source_url ? el("a", { href: it.source_url, target: "_blank", rel: "noopener" }, [it.source_name]) : it.source_name]),
      el("td", { class: "muted" }, [it.document || "—"]),
      el("td", { class: "muted" }, [it.retrieved_at ? fmtDate(it.retrieved_at) : "—"]),
      el("td", { class: "num" }, [it.value]),
      el("td", { class: "muted" }, [it.calculation]),
      el("td", { class: "muted" }, [it.dashboard_signal]),
      el("td", { class: "muted" }, [it.agent || "—"]),
    ])
  );
  return el("div", { class: "panel" }, [
    el("div", { class: "panel-title" }, [`Evidence / Audit Trail (${items.length} claims)`]),
    el("div", { class: "table-scroll" }, [
      el("table", {}, [
        el("thead", {}, [el("tr", {}, ["Claim", "Source", "Document", "Retrieved", "Value", "Calculation", "Dashboard Signal", "Agent"].map((h) => el("th", {}, [h])))]),
        el("tbody", {}, rows),
      ]),
    ]),
    el("div", { class: "panel-note" }, [
      "Every claim also carries a unique evidence_id, the assessment_id of this record, and the correlation_id " +
      "of the API request that produced it, for full audit traceability (see the raw /evidence endpoint).",
    ]),
  ]);
}

function buildStrategyPanel(a) {
  const dq = a.data_quality || {};
  const conv = a.convergence;
  const fp = a.financial_profile;

  const situation = isInsufficientData(a)
    ? `${a.company_name} (${a.ticker}): Competitive Expansion Signal could not be legitimately calculated for ` +
      `${a.facility_name} — insufficient live data (see Data Quality).`
    : `${a.company_name} (${a.ticker})'s Competitive Expansion Signal is ${a.momentum_level.replace(/_/g, " ")} ` +
      `(composite index ${fmtNum(a.momentum_score, 1)}/100) at ${a.facility_name}.`;

  const evidenceBullets = [];
  const revenueMetric = fp && fp.metrics && fp.metrics.revenue_growth;
  if (revenueMetric && revenueMetric.status === "AVAILABLE" && revenueMetric.yoy_change_pct !== null) {
    evidenceBullets.push(`SEC filings indicate revenue changed ${fmtPercent(revenueMetric.yoy_change_pct, 1)} year-over-year (${revenueMetric.fiscal_period}).`);
  }
  const capexMetric = fp && fp.metrics && fp.metrics.capex_growth;
  if (capexMetric && capexMetric.status === "AVAILABLE" && capexMetric.yoy_change_pct !== null) {
    evidenceBullets.push(`Disclosed capital expenditure changed ${fmtPercent(capexMetric.yoy_change_pct, 1)} year-over-year, directionally relevant to a capacity build-out thesis.`);
  }
  if (a.satellite_visual) {
    evidenceBullets.push(
      `Satellite imagery of the facility indicates an overall change score of ${fmtNum(a.vision_findings.overall_change_score, 2)} ` +
      `(large-scale land-cover change only — not confirmation of any specific business event).`
    );
  }
  if (!evidenceBullets.length) evidenceBullets.push("No independently-sourced quantitative evidence was available for this assessment; the score above rests on deterministic signal composition alone.");

  let implication = "Insufficient corroborating evidence to state an implication with confidence.";
  if (conv) {
    if (conv.outcome === "SIGNAL_CONVERGENCE") implication = "Financial and satellite evidence independently point the same direction, which is a materially stronger basis for a follow-up than either signal alone — though still not a confirmed business event.";
    else if (conv.outcome === "PARTIAL_SIGNAL_CONVERGENCE") implication = "Financial and/or satellite evidence suggest expansion, but external corroboration is limited — treat this as a lead worth investigating, not a conclusion.";
    else if (conv.outcome === "SIGNAL_CONFLICT") implication = "Financial and satellite evidence point in different directions; the two should not be cited together as confirming the same narrative until reconciled.";
    else implication = "At least one primary evidence group lacks enough live data to support a convergence judgment.";
  }

  const uncertaintyBits = [];
  if (dq.completeness_pct !== undefined) uncertaintyBits.push(`${fmtNum(dq.completeness_pct, 0)}% of tracked signals are live for this assessment.`);
  if (dq.fallback_fields && dq.fallback_fields.length) uncertaintyBits.push(`Fallback/estimated fields: ${dq.fallback_fields.join(", ")}.`);
  if (dq.unavailable_fields && dq.unavailable_fields.length) uncertaintyBits.push(`Unavailable fields: ${dq.unavailable_fields.join(", ")}.`);
  uncertaintyBits.push(`Narrative provenance: ${a.narrative_status.replace(/_/g, " ")}.`);
  if (a.narrative_grounding && a.narrative_grounding.flagged_values && a.narrative_grounding.flagged_values.length) {
    uncertaintyBits.push(`Grounding critic flagged unsupported figures in the original AI draft: ${a.narrative_grounding.flagged_values.join(", ")} (draft was replaced with the deterministic template).`);
  }

  return el("div", { class: "panel" }, [
    el("div", { class: "panel-title" }, ["Executive Strategy View"]),
    el("div", { class: "strategy-block" }, [el("h4", {}, ["Situation"]), el("p", {}, [situation])]),
    el("div", { class: "strategy-block" }, [el("h4", {}, ["Evidence"]), el("ul", {}, evidenceBullets.map((b) => el("li", {}, [b])))]),
    el("div", { class: "strategy-block" }, [el("h4", {}, ["Implication"]), el("p", {}, [implication])]),
    el("div", { class: "strategy-block" }, [el("h4", {}, ["Uncertainty"]), el("ul", {}, uncertaintyBits.map((b) => el("li", {}, [b])))]),
    el("div", { class: "strategy-block" }, [
      el("h4", {}, ["Areas to Investigate"]),
      el("ul", {}, (a.recommended_actions || []).map((act) => el("li", {}, [act]))),
    ]),
    el("div", { class: "panel-note" }, [
      "This view is templated client-side from the already-computed fields above (financial metrics, satellite " +
      "findings, convergence, data quality, recommended actions) — it introduces no new analysis or scoring.",
    ]),
  ]);
}

function drawLineChart(canvas, series, opts) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 46, r: 12, t: 12, b: 26 };
  const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;

  const points = series.filter((p) => p.y !== null && p.y !== undefined);
  if (!points.length) {
    ctx.fillStyle = "#8a93a1";
    ctx.font = "12px sans-serif";
    ctx.fillText("No disclosed values for this series", pad.l, h / 2);
    return;
  }
  const xs = series.map((p) => p.x);
  const ys = points.map((p) => p.y);
  let yMin = Math.min(0, ...ys), yMax = Math.max(...ys);
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const xStep = series.length > 1 ? plotW / (series.length - 1) : 0;

  ctx.strokeStyle = "#dfe3e8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, pad.t + plotH);
  ctx.lineTo(pad.l + plotW, pad.t + plotH);
  ctx.stroke();

  ctx.fillStyle = "#5b6472";
  ctx.font = "10px sans-serif";
  ctx.fillText(opts.yFmt(yMax), 2, pad.t + 8);
  ctx.fillText(opts.yFmt(yMin), 2, pad.t + plotH);

  ctx.strokeStyle = opts.color || "#2454a6";
  ctx.lineWidth = 2;
  ctx.beginPath();
  let started = false;
  series.forEach((p, i) => {
    const x = pad.l + i * xStep;
    ctx.fillStyle = "#5b6472";
    ctx.font = "10px sans-serif";
    const label = String(p.x);
    ctx.fillText(label, x - 12, h - 6);
    if (p.y === null || p.y === undefined) { started = false; return; }
    const y = pad.t + plotH - ((p.y - yMin) / (yMax - yMin)) * plotH;
    if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
  });
  ctx.stroke();

  ctx.fillStyle = opts.color || "#2454a6";
  series.forEach((p, i) => {
    if (p.y === null || p.y === undefined) return;
    const x = pad.l + i * xStep;
    const y = pad.t + plotH - ((p.y - yMin) / (yMax - yMin)) * plotH;
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function buildHistoricalPanel(a) {
  const container = el("div", { class: "panel" }, [
    el("div", { class: "flex-between" }, [
      el("div", { class: "panel-title" }, ["Historical Analysis"]),
      el("button", { class: "secondary", onclick: () => loadHistorical(a.ticker, container) }, ["Load fiscal-year history"]),
    ]),
    el("div", { class: "panel-note" }, ["Real, disclosed-only fiscal-year series from SEC EDGAR. Missing years or line items are left absent — never interpolated."]),
  ]);
  container.dataset.loaded = "false";
  return container;
}

async function loadHistorical(ticker, container) {
  const noteEl = container.querySelector(".panel-note");
  noteEl.textContent = "Loading…";
  try {
    const hist = await apiFetch(`/historical/${encodeURIComponent(ticker)}`);
    const years = hist.fiscal_years || [];
    // Remove any previously-rendered chart content.
    container.querySelectorAll(".chart-wrap, table").forEach((n) => n.remove());
    if (!years.length) {
      noteEl.textContent = "No disclosed multi-year fiscal series available for this ticker (live data mode may be disabled, or the ticker could not be resolved).";
      return;
    }
    const revSeries = years.map((y) => ({ x: y.fiscal_year, y: y.revenue }));
    const wrap = el("div", { class: "chart-wrap" }, [el("canvas", { class: "chart", width: "640", height: "180" })]);
    container.insertBefore(wrap, noteEl);
    drawLineChart(wrap.querySelector("canvas"), revSeries, { yFmt: (v) => fmtUsd(v), color: "#2454a6" });

    const rows = years.map((y) =>
      el("tr", {}, [
        el("td", {}, [String(y.fiscal_year)]),
        el("td", { class: "num" }, [y.revenue !== null && y.revenue !== undefined ? fmtUsd(y.revenue) : "—"]),
        el("td", { class: "num" }, [y.rd !== null && y.rd !== undefined ? fmtUsd(y.rd) : "—"]),
        el("td", { class: "num" }, [y.capex !== null && y.capex !== undefined ? fmtUsd(y.capex) : "—"]),
        el("td", { class: "num" }, [y.gross_margin !== null && y.gross_margin !== undefined ? fmtPercent(y.gross_margin, 1) : "—"]),
        el("td", { class: "num" }, [y.operating_margin !== null && y.operating_margin !== undefined ? fmtPercent(y.operating_margin, 1) : "—"]),
        el("td", { class: "num" }, [y.net_margin !== null && y.net_margin !== undefined ? fmtPercent(y.net_margin, 1) : "—"]),
      ])
    );
    const table = el("table", {}, [
      el("thead", {}, [el("tr", {}, ["Fiscal Year", "Revenue", "R&D", "CapEx", "Gross Margin", "Operating Margin", "Net Margin"].map((h) => el("th", {}, [h])))]),
      el("tbody", {}, rows),
    ]);
    container.insertBefore(table, noteEl);
    noteEl.textContent = `Retrieved ${fmtDate(hist.retrieved_at)} · CIK ${hist.cik || "—"}. Blank cells mean the filer did not disclose that line item for that fiscal year.`;
  } catch (err) {
    noteEl.textContent = "Failed to load: " + err.message;
  }
}

function buildAgentTracePanel(a) {
  const rows = (a.agent_trace || []).map((t) =>
    el("tr", {}, [
      el("td", {}, [t.agent]),
      el("td", {}, [el("span", { class: "badge " + (t.status === "ok" ? "badge-ok" : "badge-bad") }, [t.status])]),
      el("td", { class: "num" }, [fmtNum(t.duration_ms, 1) + " ms"]),
      el("td", { class: "muted" }, [t.summary]),
    ])
  );
  return el("div", { class: "panel" }, [
    el("div", { class: "panel-title" }, ["Agent Trace"]),
    el("table", {}, [
      el("thead", {}, [el("tr", {}, ["Agent", "Status", "Duration", "Summary"].map((h) => el("th", {}, [h])))]),
      el("tbody", {}, rows),
    ]),
  ]);
}

// ---------------------------------------------------------------------
// Assessment History
// ---------------------------------------------------------------------

async function loadAssessments() {
  const listEl = $("assessments-list");
  listEl.innerHTML = "";
  listEl.appendChild(el("div", { class: "loading" }, ["Loading…"]));
  try {
    const data = await apiFetch("/assessments?limit=100");
    state.assessments = data.results || [];
    $("assessments-count").textContent = `${data.total_matching} total`;
    listEl.innerHTML = "";
    if (!state.assessments.length) {
      listEl.appendChild(el("div", { class: "empty-state" }, ["No assessments yet — run one from New Assessment."]));
      return;
    }
    state.assessments.forEach((a) => {
      const checkbox = el("input", { type: "checkbox" });
      checkbox.checked = state.selectedForCompare.has(a.id);
      checkbox.addEventListener("change", (e) => {
        e.stopPropagation();
        if (checkbox.checked) state.selectedForCompare.add(a.id); else state.selectedForCompare.delete(a.id);
        updateCompareButton();
      });
      const row = el("div", { class: "list-row" }, [
        el("div", { class: "checkbox-inline", onclick: (e) => e.stopPropagation() }, [checkbox]),
        el("div", { style: "flex:1;margin-left:10px;", onclick: () => { state.currentAssessment = a; renderDashboard(a); showView("dashboard"); } }, [
          el("div", { class: "main-line" }, [`${a.company_name} (${a.ticker}) — ${a.facility_name}`]),
          el("div", { class: "sub-line" }, [`${a.momentum_level} · ${fmtCompositeIndex(a)} · ${fmtDate(a.created_at)}`]),
        ]),
        el("span", { class: "badge " + badgeClassForLevel(a.momentum_level) }, [a.momentum_level]),
      ]);
      listEl.appendChild(row);
    });
  } catch (err) {
    listEl.innerHTML = "";
    listEl.appendChild(el("div", { class: "error-box" }, ["Failed to load assessments: " + err.message]));
  }
}

function updateCompareButton() {
  const btn = $("compare-selected-btn");
  btn.disabled = state.selectedForCompare.size < 2;
  btn.textContent = `Compare selected (${state.selectedForCompare.size}) →`;
}

$("refresh-assessments-btn").addEventListener("click", loadAssessments);

$("compare-selected-btn").addEventListener("click", async () => {
  const ids = Array.from(state.selectedForCompare);
  if (ids.length < 2) return;
  showView("compare");
  await runCompare(ids);
});

// ---------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------

async function runCompare(ids) {
  $("compare-empty").style.display = "none";
  const root = $("compare-content");
  root.style.display = "block";
  root.innerHTML = "";
  root.appendChild(el("div", { class: "loading" }, ["Loading…"]));
  try {
    const data = await apiFetch("/compare", { method: "POST", body: JSON.stringify({ assessment_ids: ids }) });
    root.innerHTML = "";
    if (data.missing_ids && data.missing_ids.length) {
      root.appendChild(el("div", { class: "error-box" }, [`Not found: ${data.missing_ids.join(", ")}`]));
    }
    const allSignalKeys = new Set();
    data.results.forEach((r) => Object.keys(r.signal_breakdown || {}).forEach((k) => allSignalKeys.add(k)));

    const headerRow = el("tr", {}, ["Company", "Momentum", "Level", "Completeness", "Narrative", ...Array.from(allSignalKeys).map((k) => k.replace(/_/g, " "))].map((h) => el("th", {}, [h])));
    const rows = data.results.map((r) =>
      el("tr", {}, [
        el("td", {}, [el("strong", {}, [`${r.ticker}`]), el("div", { class: "muted" }, [r.company_name])]),
        el("td", { class: "num" }, [isInsufficientData(r) ? "Insufficient Data" : fmtNum(r.momentum_score, 1)]),
        el("td", {}, [el("span", { class: "badge " + badgeClassForLevel(r.momentum_level) }, [r.momentum_level])]),
        el("td", { class: "num" }, [r.data_quality ? fmtNum(r.data_quality.completeness_pct, 0) + "%" : "—"]),
        el("td", {}, [el("span", { class: "badge " + badgeClassForNarrativeStatus(r.narrative_status) }, [r.narrative_status.replace(/_/g, " ")])]),
        ...Array.from(allSignalKeys).map((k) => el("td", { class: "num" }, [r.signal_breakdown && r.signal_breakdown[k] !== undefined ? fmtNum(r.signal_breakdown[k], 1) : "—"])),
      ])
    );
    root.appendChild(el("div", { class: "panel" }, [
      el("div", { class: "panel-title" }, [`Sorted by ${data.sorted_by}`]),
      el("div", { class: "table-scroll" }, [el("table", {}, [el("thead", {}, [headerRow]), el("tbody", {}, rows)])]),
    ]));
  } catch (err) {
    root.innerHTML = "";
    root.appendChild(el("div", { class: "error-box" }, ["Failed to compare: " + err.message]));
  }
}

// ---------------------------------------------------------------------
// Watchlist
// ---------------------------------------------------------------------

async function loadWatchlist() {
  const root = $("watchlist-content");
  root.innerHTML = "";
  root.appendChild(el("div", { class: "loading" }, ["Loading…"]));
  const tier = $("watchlist-tier-filter").value;
  try {
    const qs = tier ? `?tier=${encodeURIComponent(tier)}` : "";
    const data = await apiFetch("/watchlist" + qs);
    root.innerHTML = "";
    if (!data.results.length) {
      root.appendChild(el("div", { class: "empty-state" }, ["No entries for this filter."]));
      return;
    }
    const rows = data.results.map((wEntry) =>
      el("tr", {}, [
        el("td", {}, [el("strong", {}, [wEntry.ticker]), el("div", { class: "muted" }, [wEntry.company_name])]),
        el("td", {}, [wEntry.facility_name]),
        el("td", {}, [el("span", { class: "badge " + badgeClassForWatchlistTier(wEntry.tier) }, [wEntry.tier])]),
        el("td", {}, [wEntry.channel || "—"]),
        el("td", { class: "num" }, [isInsufficientData(wEntry) ? "Insufficient Data" : fmtNum(wEntry.momentum_score, 1)]),
        el("td", {}, [el("span", { class: "badge " + (wEntry.webhook_status === "DISPATCHED" ? "badge-ok" : wEntry.webhook_status === "DISPATCH_FAILED" ? "badge-bad" : "badge-neutral") }, [wEntry.webhook_status.replace(/_/g, " ")])]),
        el("td", { class: "muted" }, [fmtDate(wEntry.updated_at)]),
      ])
    );
    root.appendChild(el("table", {}, [
      el("thead", {}, [el("tr", {}, ["Company", "Facility", "Tier", "Channel", "Momentum", "Webhook", "Updated"].map((h) => el("th", {}, [h])))]),
      el("tbody", {}, rows),
    ]));
  } catch (err) {
    root.innerHTML = "";
    root.appendChild(el("div", { class: "error-box" }, ["Failed to load watchlist: " + err.message]));
  }
}

$("refresh-watchlist-btn").addEventListener("click", loadWatchlist);
$("watchlist-tier-filter").addEventListener("change", loadWatchlist);
