# Phase 2 Build Brief — Beyond Mean-Variance Optimiser

*Paste this as the first message of a new chat to start the build phase with full context.*

---

## 1. What this is

I'm extending my existing Streamlit app, the **Beyond Mean-Variance Portfolio Optimiser**, into a richer "investing flight-simulator" — a tool where users can profile their risk, explore instruments, optimise, and track a **simulated** portfolio over time.

**Goal of this phase:** the app is a **credibility piece for my job search** (program/transformation/quant roles in banking & capital markets). It is *not* a commercial product right now, so prioritise impressiveness, correctness, and end-to-end product polish over monetisation.

**Hard rule — no investment advice.** Under French/EU law, a personalised recommendation on financial instruments = *conseil en investissement* (regulated, requires CIF/AMF/ORIAS). So the app stays strictly **educational / analytical / user-driven**: it shows information, computes, and explains — it never ranks-and-recommends specific securities or tells the user what to buy. A visible disclaimer ("Ceci ne constitue pas un conseil en investissement") appears wherever instrument data is shown.

## 2. Stack & where things live

- Single-file **Streamlit** app: `app.py` (~6,200 lines).
- **Phase 2 working folder (source of truth for this phase):** `C:\PortfolioApp_Phase2\` — a clean copy, isolated from the deployed app. Work here. It already contains all fixes through 13 June 2026 (glossary scroll + Barrier-M diagram).
- The **deployed/live** app lives separately at `C:\Users\borjs\Projects\PythonPortfolio\` — leave it untouched while experimenting; merge back only when a phase is stable.
- **Tool note:** read `app.py` with the Read/Edit tools, not bash — the bash file-mount can show a stale cached copy. Use bash only for running things.
- Repo: `SamiJeddou/behavioral-portfolio-optimizer`. **Commit hygiene:** only commit `app.py` + the images it references. Never commit working/junk files (`app.py.bak_*`, `*_v*.pptx`, `guide_icons/`, `.~lock.*`, `lu*.tmp`, etc.).
- Runs locally (`streamlit run app.py`, localhost:8501) and deploys on Streamlit Cloud (watch for cloud timeouts on heavy runs).
- Engine modules alongside `app.py`: `behavioral_portfolio_optimizer.py`, `mc_cvar_optimizer.py`, `turbo_optimizer.py`, `es_rigorous.py`, `stress_test_turbo.py`. Market data: Yahoo Finance via `yfinance`.

## 3. Existing functionality (already built)

- Three engines: **grid** (mean-variance vs behavioural efficient frontier), **scalable** (Monte-Carlo + CVaR, Rockafellar–Uryasev LP), **backtest** (out-of-sample).
- Constraints: VaR / ES / rigorous-ES, threshold **H** and shortfall prob **α**, implied risk-aversion **λ**. Derivatives/structured products supported in the grid engine.
- **AI Glossary** module: click a term → AI-generated explanation, with payoff diagrams via `_GLOSSARY_DERIV`. Reuse this AI-explanation pattern for Phase 4 ratio explanations.
- About / User Guide / Worked Examples / Run-Locally guide all done.

## 4. Conventions & known Streamlit gotchas

- **Palette:** gold `#E3C77E` / `#C9A24B` / `#f5b942`; blue `#4a9eff`; green `#26a641` / `#10b981`; slate `#9aa7bd`; page bg `#0d1117`; panels `#1b2330` / `#141a23`; border `#30363d`.
- Streamlit's sanitizer **strips `id`** from raw HTML → use `st.subheader(title, anchor=...)` for in-page anchors.
- `:material/...:` icon shortcodes work in markdown but **not** reliably inline-with-links → use inline SVGs there.
- A table's first row becomes `<th>` and loses cell colour → use `st.html` with styled `<td>` headers when colour is needed.
- `st.components.v1.html` is **cached** unless the HTML changes → add a per-render nonce when injecting JS that must re-run.
- Plotly charts: pass `config={'responsive': True}` to avoid mobile rotation/resize bugs.

## 5. Build order (4 phases, each independently demoable)

**Phase 1 — Per-asset weight constraints.** Replace the single global "max weight" input with per-security min/max bounds, wired into the existing optimiser's box constraints. Small, builds on existing machinery. First win.

**Phase 2 — Risk-profiling module.** Short questionnaire → mapped to λ (or the H/α parameters). Self-contained, no external data. Use a citable methodology (e.g. a Grable-Lytton-style scale or a documented custom mapping) so it's interview-defensible. Output framed as "this sets your simulation's risk parameters," not a suitability profile.

**Phase 3 — Paper-portfolio tracking (the centerpiece).**
- 3a (foundation): a persistence layer to save profile / portfolio / history, and a decided price-data source.
- 3b: confirm or manually adjust weights; live stats (mean, variance, risk level vs. appetite, portfolio value, per-asset & portfolio alpha/beta, correlations); re-run optimiser to compare a suggested reallocation; historical time-series of valuation and risk level. Framed throughout as "your simulated portfolio."

**Phase 4 — Ticker analytics + AI ratio explanations (reframed Item 2).** User enters a ticker → app shows info, ratios (CFA-style: valuation, profitability, leverage, risk), and **AI-powered plain-language explanations** of each ratio (reuse the glossary AI pattern). Show characteristics and a neutral factor/risk breakdown — **never** a score-and-recommend or "best picks." Start with a handful of ratios, expand. Disclaimer always visible.

## 6. Let's start with Phase 1

Please read `app.py` (with the Read tool) from `C:\PortfolioApp_Phase2\`, find the current global max-weight input and the optimiser's constraint setup, and propose how to add per-security min/max weight bounds. Don't change code until we've agreed the approach.
