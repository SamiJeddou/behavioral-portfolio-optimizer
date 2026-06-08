# Making the Optimiser Pluggable — `core/` Boundary & Refactor Blueprint

**Goal:** separate the compute engines from the Streamlit UI so the same logic can be
called by a REST API, an MCP server, a notebook, or Excel — without dragging `streamlit`
along. The maths is **not** rewritten; functions are relocated and given one stable,
typed entry point each.

**Guiding rule:** anything in `core/` and `reporting/` must contain **zero `st.` calls**
and do **no I/O side-effects** except what it's explicitly for (network only in
`markets.py`). The UI imports the core; the core never imports the UI.

---

## 1. Where each existing function goes

| Current location (`app.py` unless noted) | New home | Notes |
|---|---|---|
| `build_state_space`, `assign_probabilities`, `optimize_portfolio` *(behavioral_portfolio_optimizer)* | `core/grid.py` (re-export) | already a module — keep, re-export |
| `optimize_portfolio_turbo` *(turbo_optimizer)* | `core/grid.py` (re-export) | keep |
| `optimize_portfolio_es_rigorous` *(es_rigorous)* | `core/grid.py` (re-export) | keep |
| `run_opt`, `build_frontier`, `compute_mv_frontier`, `mv_frontier_at_return`, `implied_lambda` | `core/grid.py` | orchestrators; wrap with typed API below |
| `mc_generate_scenarios`, `_mc_psd_cholesky`, `mc_build_matrix` | `core/scenario.py` | scenario generation |
| `mc_max_return_cvar`, `mc_min_cvar`, `mc_frontier`, `_mc_cvar_rows`, `mc_gmv_weights` | `core/scenario.py` | the CVaR LP |
| `mc_realised_es`, `mc_analytical_es` | `core/scenario.py` | tail measures |
| `bs_call`, `bs_put`, `compute_structured_payoff` *(behavioral_portfolio_optimizer)* | `core/pricing.py` (re-export) | keep |
| `build_der_config`, `preset_components` | `core/pricing.py` | derivative config |
| `mc_der_returns`, `_mc_leg_intrinsic`, `_mc_leg_value_vec` | `core/pricing.py` | scenario payoffs |
| `_bt_legs`, `_leg_value`, `mtm_gross_path`, `_bt_portfolio_path` | `core/pricing.py` | mark-to-market |
| `_bt_metrics`, `_capm_alpha_beta`, `stats_from_prices` | `core/backtest.py` | realised stats, alpha/beta |
| `fetch_tickers`, `fetch_close_prices`, `parse_csv`, `clean_returns`, `corr_to_cov` | `core/markets.py` | data layer (only network code) |
| `_styled_pdf`, `_pdf_safe`, `_md_bold`, `generate_pdf_report`, `generate_mc_pdf_report`, `generate_backtest_pdf_report`, `make_donut_svg` | `reporting/` | st-free already; produce bytes/SVG |
| `plot_frontier_plotly`, `plot_payoff`, `plot_named_payoff`, `plot_backtest_paths_plotly`, `plot_mc_frontier`, … | `viz/` (optional) | return figures; UI-adjacent but importable |
| `get_explanation`, `get_ai_chat_response` | `core/ai.py` | inject the API key, don't hardcode |
| `_render_home`, `show_portfolio_data`, `_mc_head`, `_bt_head`, `_bt_param_inputs`, `_mc_isblank`, `_go_home` | **stay in `app.py`** | pure UI |

---

## 2. The typed boundary (`core/types.py`)

This is the real design work — one input type and one result type per engine, so the
public surface is stable even if internals change.

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class AssetUniverse:
    names: list[str]
    means: np.ndarray            # annualised expected returns, shape (n,)
    sigmas: np.ndarray           # annualised vols, shape (n,)
    corr: np.ndarray             # correlation matrix, shape (n, n)

@dataclass
class DerivativeSpec:
    type: str                    # 'protective_put' | 'collar' | 'cgn' | 'straddle' | ...
    underlying_idx: int
    params: dict                 # {'strike': 0.95} etc. (matches build_der_config)
    T: float = 1.0
    r: float = 0.03
    label: str = ""
    vol_override: float | None = None

@dataclass
class Constraint:
    kind: str = "var"            # 'var' | 'es_thesis' | 'es_rigorous'
    H: float = -0.15             # loss threshold
    alpha: float = 0.05          # shortfall probability (VaR) / CVaR tail
    L: float | None = None       # ES / CVaR floor (for es_* and scenario)

@dataclass
class PortfolioResult:
    weights: np.ndarray
    labels: list[str]
    expected_return: float
    std_dev: float
    shortfall_stat: float        # P(r<H) or E[r|r<H], depending on kind
    feasible: bool
    meta: dict = field(default_factory=dict)   # skew, scenarios, copula, solver, ...

@dataclass
class FrontierPoint:
    x: float                     # risk axis (std dev %, or CVaR floor %)
    y: float                     # max expected return %
    label: str
    feasible: bool = True
```

> The existing solvers already return dicts with `weights`, `std_dev`,
> `expected_return`, `shortfall_stat` — so mapping a dict to `PortfolioResult` is a
> one-liner. No solver changes.

---

## 3. Public engine API (the stable front door)

### `core/grid.py`
```python
def optimise_grid(universe: AssetUniverse,
                  constraint: Constraint,
                  derivatives: list[DerivativeSpec] | None = None,
                  resolution: str = "standard") -> PortfolioResult:
    """Exact grid engine. resolution in {fast, standard, high, turbo}.
    Internally: corr->cov, build der_config, call run_opt(...)."""

def grid_frontier(universe: AssetUniverse,
                  constraint: Constraint,
                  derivatives: list[DerivativeSpec] | None = None,
                  resolution: str = "standard") -> list[FrontierPoint]:
    """Wraps build_frontier(...)."""
```

### `core/scenario.py`
```python
def optimise_scenario(universe: AssetUniverse,
                      constraint: Constraint,
                      derivatives: list[DerivativeSpec] | None = None,
                      scenarios: int = 10_000,
                      copula: str = "gaussian",   # 'gaussian' | 'student_t'
                      dof: int = 5,
                      w_max: float | None = None,
                      seed: int = 0) -> PortfolioResult:
    """MC + CVaR LP. Internally: mc_generate_scenarios -> mc_build_matrix ->
    mc_max_return_cvar(R, alpha, L, w_max)."""

def scenario_frontier(universe: AssetUniverse,
                      constraint: Constraint,
                      derivatives=None,
                      floors: list[float] | None = None,
                      scenarios: int = 10_000,
                      copula: str = "gaussian",
                      w_max: float | None = None,
                      seed: int = 0) -> list[FrontierPoint]:
    """Wraps mc_frontier(...) with common random numbers (fixed seed)."""
```

### `core/backtest.py`
```python
@dataclass
class BacktestResult:
    p1: PortfolioResult          # no-derivative
    p2: PortfolioResult          # with-derivative
    realised: dict               # ann/vol/cum/breach for P1 & P2
    alpha_beta: dict             # per-asset and per-portfolio beta/alpha/r2
    paths: dict                  # {'dates': [...], 'pv1': [...], 'pv2': [...]}
    verdict: list[str]

def run_backtest(tickers: list[str],
                 construction: tuple[str, str],     # (start, end) ISO dates
                 evaluation: tuple[str, str],
                 constraint: Constraint,
                 derivative: DerivativeSpec | None = None,
                 benchmark: str | None = None,
                 freq: str = "Daily",
                 rf: float = 0.0,
                 source: "DataSource | None" = None) -> BacktestResult:
    """Construction-window optimise -> hold -> mark-to-market -> realised stats."""
```

### `core/markets.py`  (pluggable data — the other meaning of "external tools")
```python
from typing import Protocol
import pandas as pd

class DataSource(Protocol):
    def prices(self, tickers: list[str], start: str, end: str,
               freq: str) -> pd.DataFrame: ...

class YFinanceSource:    # wraps existing fetch_close_prices / fetch_tickers
    ...
class CSVSource:         # wraps parse_csv
    ...
# A vendor feed (Bloomberg/Refinitiv) becomes one more class implementing prices().

def universe_from_prices(prices: pd.DataFrame, freq: str) -> AssetUniverse:
    """clean_returns -> stats_from_prices -> corr -> AssetUniverse."""
```

---

## 4. Thin adapters (built once the core exists)

### `api/main.py` — FastAPI
```python
from fastapi import FastAPI
from core.grid import optimise_grid
from core.scenario import optimise_scenario
from core.types import AssetUniverse, Constraint, DerivativeSpec

app = FastAPI(title="Beyond Mean-Variance Optimiser API")

@app.post("/optimise/scenario")
def _scenario(universe: AssetUniverse, constraint: Constraint,
              derivatives: list[DerivativeSpec] = []):
    return optimise_scenario(universe, constraint, derivatives or None)
# + /optimise/grid, /frontier, /backtest
```
(Use Pydantic models mirroring the dataclasses for request/response validation.)

### `mcp/server.py` — MCP (highest appeal-per-effort)
Expose the same four functions as MCP tools (`optimise_grid`, `optimise_scenario`,
`trace_frontier`, `run_backtest`). Each tool description = a sentence; each handler =
build the dataclasses from JSON args, call the core function, return JSON. ~150–250 lines.

---

## 5. Migration order (low-risk, each step independently testable)

1. **Create `core/types.py`** — dataclasses only. No behaviour change.
2. **Move `core/pricing.py`** (most self-contained). Import back into `app.py`; run AppTest — views unchanged.
3. **Move `core/scenario.py`**, then `core/grid.py`, then `core/backtest.py`, then `core/markets.py`. After each move, `app.py` just `from core.X import ...`; AppTest all views each time.
4. **Add the typed wrappers** (`optimise_grid`, `optimise_scenario`, …) in each core module. Re-point the Streamlit views to call the wrappers instead of the raw functions — confirms the boundary is real.
5. **Move `reporting/`** (PDF + SVG). Re-import.
6. **Add `api/main.py`** — validate against a couple of known cases (compare API output to the app's numbers).
7. **Add `mcp/server.py`.**

At every step `app.py` keeps working because it just imports from the new modules.
The thesis-validation numbers are your regression test: if `optimise_grid` reproduces
the 18 thesis tables after the move, the refactor is correct.

---

## 6. Honest caveats

- **Grid engine** still has the `m^N` blow-up — expose `optimise_scenario` as the
  "scale" endpoint; keep `optimise_grid` for small/exact and as the reference.
- **Latency** — the CVaR LP is ~10s/solve. Fine for batch, MCP, and notebooks; for a
  synchronous REST call, document it or make `/optimise/scenario` async (job + poll).
- **Data licensing** — a real vendor `DataSource` carries redistribution terms; the
  `Protocol` keeps that concern at the edge, out of the engines.
- **AI layer** — `core/ai.py` must take the API key as an argument/env var, never
  hardcoded, so the package is safe to share.

---

*Companion to the Beyond Mean-Variance Portfolio Optimiser. Relocation preserves all
existing validation; no solver maths is changed.*
