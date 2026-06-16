# © 2026 Sami Jeddou. All rights reserved.
# Published publicly for demonstration and evaluation only — no license is granted.
# Copying, modification, redistribution, or reuse (in whole or in part) without the
# author's prior written permission is prohibited.
"""Naive / classical benchmark portfolios — UI-free.

Well-defined reference portfolios for *any* asset universe, used to put the
behavioural / CVaR optima in context:

  * Equal-weight (1/N)          — naive diversification
  * Minimum-variance (long-only) — lowest-variance fully-invested portfolio
  * Max-Sharpe / tangency (long-only) — the classical mean-variance optimum

Plus two small stat helpers so the same risk/return numbers can be produced from
either Gaussian moments (the grid engine's world) or a Monte-Carlo scenario matrix
(the scalable engine's world). No Streamlit, no I/O.

These are deliberately long-only and fully invested (sum w = 1, w >= 0) so they are
directly comparable to the app's long-only optimisers. They are references, not
recommendations.
"""
import numpy as np
from scipy.stats import norm as _norm

try:                                   # optional; we fall back gracefully if absent
    from scipy.optimize import minimize as _minimize
except Exception:                      # pragma: no cover
    _minimize = None

__all__ = [
    "RF_ANNUAL", "equal_weight", "min_variance_weights", "max_sharpe_weights",
    "benchmark_set", "stats_from_moments", "stats_from_scenarios",
]

RF_ANNUAL = 0.03                       # default annual risk-free rate (matches backtest)


# ── weight constructors ──────────────────────────────────────────────────────

def equal_weight(n):
    """1/N weights."""
    n = int(n)
    return np.ones(n) / n if n > 0 else np.zeros(0)


def _project_simplex(w):
    """Project a weight vector onto {w >= 0, sum w = 1} (Euclidean). Fallback only."""
    w = np.asarray(w, float)
    if w.sum() <= 0:
        return equal_weight(len(w))
    u = np.sort(w)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, len(w) + 1)
    cond = u - css / ind > 0
    rho = ind[cond][-1] if cond.any() else 1
    theta = css[rho - 1] / rho
    return np.maximum(w - theta, 0.0)


def min_variance_weights(cov, long_only=True):
    """Long-only minimum-variance weights: min wᵀΣw s.t. sum w = 1, w >= 0.

    Uses SLSQP when SciPy's optimiser is available, otherwise falls back to the
    analytical global-minimum-variance solution projected onto the simplex."""
    cov = np.asarray(cov, float)
    n = cov.shape[0]
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    if _minimize is not None and long_only:
        try:
            res = _minimize(
                lambda w: float(w @ cov @ w),
                equal_weight(n),
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n,
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                options={"maxiter": 500, "ftol": 1e-12},
            )
            if res.success and np.isfinite(res.x).all():
                w = np.clip(res.x, 0.0, None)
                s = w.sum()
                if s > 0:
                    return w / s
        except Exception:
            pass
    # analytical GMV (shorts allowed) → project to long-only simplex
    try:
        inv = np.linalg.pinv(cov)
        one = np.ones(n)
        w = inv @ one
        w = w / (one @ w)
        return _project_simplex(w) if long_only else w
    except Exception:
        return equal_weight(n)


def max_sharpe_weights(means, cov, rf=RF_ANNUAL, long_only=True):
    """Long-only maximum-Sharpe (tangency) weights: max (wᵀμ − rf)/√(wᵀΣw)
    s.t. sum w = 1, w >= 0. SLSQP with an analytical fallback."""
    means = np.asarray(means, float)
    cov = np.asarray(cov, float)
    n = len(means)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    if _minimize is not None and long_only:
        def _neg_sharpe(w):
            mu = float(w @ means)
            sd = float(np.sqrt(max(w @ cov @ w, 1e-18)))
            return -(mu - rf) / sd
        try:
            res = _minimize(
                _neg_sharpe,
                equal_weight(n),
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n,
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                options={"maxiter": 500, "ftol": 1e-12},
            )
            if res.success and np.isfinite(res.x).all():
                w = np.clip(res.x, 0.0, None)
                s = w.sum()
                if s > 0:
                    return w / s
        except Exception:
            pass
    # analytical tangency (shorts allowed) → project to long-only simplex
    try:
        inv = np.linalg.pinv(cov)
        excess = means - rf
        w = inv @ excess
        s = w.sum()
        if abs(s) < 1e-12:
            return equal_weight(n)
        w = w / s
        return _project_simplex(w) if long_only else w
    except Exception:
        return equal_weight(n)


def return_matched_mv_weights(means, cov, target_return, long_only=True):
    """Long-only minimum-variance portfolio achieving a target expected return:
    min wᵀΣw s.t. sum w = 1, wᵀμ = target, w >= 0.

    This is the classical Markowitz frontier point at a chosen return (the same
    construction the grid engine shows as "Portfolio (0)"). If `target_return` lies
    outside the long-only achievable range [min(μ), max(μ)] it is clipped into range
    (e.g. an optimum lifted above the securities' reach by derivatives maps to the
    max-return securities portfolio). Returns None if the solver is unavailable or fails.
    """
    means = np.asarray(means, float)
    cov = np.asarray(cov, float)
    n = len(means)
    if n == 0:
        return None
    if n == 1:
        return np.ones(1)
    _tgt = float(np.clip(target_return, means.min(), means.max()))
    if _minimize is not None and long_only:
        try:
            res = _minimize(
                lambda w: float(w @ cov @ w),
                equal_weight(n),
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n,
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                             {"type": "eq", "fun": lambda w, _t=_tgt: float(w @ means) - _t}],
                options={"maxiter": 700, "ftol": 1e-12},
            )
            if res.success and np.isfinite(res.x).all():
                w = np.clip(res.x, 0.0, None)
                s = w.sum()
                if s > 0:
                    return w / s
        except Exception:
            pass
    return None


def mv_frontier_weights(means, cov, n_points=16, long_only=True):
    """Weights along the long-only Markowitz efficient frontier, from the global-
    minimum-variance return up to the maximum achievable (single-asset) return.

    Returns a list of (expected_return, weights) pairs ordered by return — feed each
    to `stats_from_*` to place the frontier in whatever risk space the caller plots."""
    means = np.asarray(means, float)
    cov = np.asarray(cov, float)
    n = len(means)
    if n < 2:
        return []
    _gmv = min_variance_weights(cov, long_only=long_only)
    r_lo = float(_gmv @ means)
    r_hi = float(means.max())
    if r_hi <= r_lo + 1e-9:
        return [(r_lo, _gmv)]
    out = []
    for _t in np.linspace(r_lo, r_hi, int(n_points)):
        _w = return_matched_mv_weights(means, cov, _t, long_only=long_only)
        if _w is not None:
            out.append((float(_w @ means), _w))
    return out


def benchmark_set(means, cov, rf=RF_ANNUAL, long_only=True):
    """Ordered list of (label, weights) for the three standard benchmarks.

    Returns weights over the same N securities described by `means`/`cov`."""
    means = np.asarray(means, float)
    cov = np.asarray(cov, float)
    n = len(means)
    return [
        ("Equal-weight (1/N)", equal_weight(n)),
        ("Minimum-variance", min_variance_weights(cov, long_only=long_only)),
        ("Max-Sharpe (tangency)", max_sharpe_weights(means, cov, rf=rf, long_only=long_only)),
    ]


# ── stat helpers (return signed decimals, e.g. -0.28 for a −28% tail) ─────────

def stats_from_moments(w, means, cov, alpha=0.05, rf=RF_ANNUAL):
    """Gaussian closed-form stats for weights `w` against moments (means, cov).

    er  — expected return            (wᵀμ)
    vol — volatility                 (√(wᵀΣw))
    sharpe — (er − rf)/vol
    var — α-quantile return          (signed; Normal)
    es  — α-CVaR / tail-average      (signed; Normal, = mc_analytical_es)
    """
    w = np.asarray(w, float)
    means = np.asarray(means, float)
    cov = np.asarray(cov, float)
    er = float(w @ means)
    var_p = float(w @ cov @ w)
    vol = float(np.sqrt(max(var_p, 0.0)))
    z = float(_norm.ppf(alpha))
    var = er + vol * z
    es = er - vol * float(_norm.pdf(z)) / alpha if alpha > 0 else er
    sharpe = (er - rf) / vol if vol > 1e-12 else float("nan")
    return {"er": er, "vol": vol, "sharpe": sharpe, "var": var, "es": es}


def stats_from_scenarios(R, w, alpha=0.05, rf=RF_ANNUAL):
    """Realised stats for weights `w` against an S×N scenario-return matrix R.

    Uses the same definitions the scalable engine reports: α-CVaR is the mean of the
    worst-α scenario returns. Returns signed decimals."""
    R = np.asarray(R, float)
    w = np.asarray(w, float)
    port = R @ w
    er = float(port.mean())
    vol = float(port.std())
    p = np.sort(port)
    k = max(1, int(np.floor(alpha * len(p))))
    es = float(p[:k].mean())
    var = float(np.quantile(port, alpha))
    sharpe = (er - rf) / vol if vol > 1e-12 else float("nan")
    return {"er": er, "vol": vol, "sharpe": sharpe, "var": var, "es": es}
