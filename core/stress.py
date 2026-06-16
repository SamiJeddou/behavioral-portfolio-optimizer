# © 2026 Sami Jeddou. All rights reserved.
# Published publicly for demonstration and evaluation only — no license is granted.
# Copying, modification, redistribution, or reuse (in whole or in part) without the
# author's prior written permission is prohibited.
"""Stress testing for the Live Portfolio — UI-free.

Three complementary views of forward / hypothetical risk on a *current* set of weights:

  * historical scenario replay — apply each holding's realised returns over a named crisis
    window to the current weights (renormalising over the holdings that have price history
    back then) and report the hypothetical cumulative loss and max drawdown,
  * instantaneous shocks — a market move propagated through the portfolio's CAPM beta, and an
    explicit per-asset shock vector,
  * a parametric stress — scale volatilities and push correlations toward 1 (the "diversification
    breaks down in a crash" effect), then re-derive the portfolio volatility and parametric VaR.

No Streamlit, no I/O — the caller supplies price DataFrames (e.g. from core.markets) and the
annualised covariance (e.g. from the Live Portfolio's stored moments).
"""
import numpy as np
import pandas as pd

__all__ = ["SCENARIOS", "replay", "market_shock", "asset_shock", "parametric", "z_for"]

# Named historical stress windows (peak→trough-ish draw-down periods).
SCENARIOS = [
    {"key": "gfc2008", "name": "Global Financial Crisis (2008–09)",
     "start": "2008-09-02", "end": "2009-03-09",
     "blurb": "Lehman collapse to the March-2009 equity bottom."},
    {"key": "euro2011", "name": "Euro sovereign-debt crisis (2011)",
     "start": "2011-07-01", "end": "2011-10-03",
     "blurb": "Sovereign-debt sell-off and US downgrade."},
    {"key": "china2015", "name": "China shock / 2015–16",
     "start": "2015-08-10", "end": "2016-02-11",
     "blurb": "Yuan devaluation and global growth scare."},
    {"key": "q4_2018", "name": "Q4-2018 selloff",
     "start": "2018-10-01", "end": "2018-12-24",
     "blurb": "Rate-hike and growth fears."},
    {"key": "covid2020", "name": "COVID crash (2020)",
     "start": "2020-02-19", "end": "2020-03-23",
     "blurb": "Fastest -34% S&P drop on record."},
    {"key": "infl2022", "name": "2022 rate-shock selloff",
     "start": "2022-01-03", "end": "2022-10-12",
     "blurb": "Inflation and aggressive hiking — stocks and bonds fell together."},
]


def replay(prices, weights, min_obs=5):
    """Replay a scenario window on the current weights.

    `prices`: DataFrame (index=dates in the window, columns=tickers).
    `weights`: {ticker: fraction}. Renormalised over the tickers that have data in the window.
    Returns a dict {cum, mdd, coverage, used, missing} or None if no holding has data.
      cum      — hypothetical portfolio cumulative return over the window (negative = loss)
      mdd      — worst peak-to-trough drawdown within the window (negative)
      coverage — share of the original weight that had price history (0..1)
    """
    if prices is None or getattr(prices, "empty", True):
        return None
    cols = [t for t in weights if t in prices.columns]
    good = [t for t in cols if prices[t].dropna().shape[0] >= min_obs]
    missing = [t for t in weights if t not in good]
    if not good:
        return None
    raw = np.array([max(float(weights[t]), 0.0) for t in good], float)
    coverage = float(raw.sum())
    if coverage <= 0:
        return None
    w = raw / raw.sum()
    rets = prices[good].pct_change().dropna(how="all").fillna(0.0)
    if rets.empty:
        return None
    port = rets.values @ w
    eq = np.cumprod(1.0 + port)
    cum = float(eq[-1] - 1.0)
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    return {"cum": cum, "mdd": mdd, "coverage": coverage, "used": good, "missing": missing,
            "equity": pd.Series(eq, index=rets.index)}


def market_shock(beta, shock_pct):
    """Portfolio return under an instantaneous market move, via CAPM beta.
    Ignores alpha and idiosyncratic risk — a first-order estimate. Returns a fraction."""
    if beta is None or (isinstance(beta, float) and beta != beta):
        return None
    return float(beta) * float(shock_pct) / 100.0


def asset_shock(weights, shocks_pct):
    """Portfolio return from an explicit per-asset shock vector: Σ wᵢ·shockᵢ. Returns a fraction."""
    total = 0.0
    for t, w in weights.items():
        total += float(w) * float(shocks_pct.get(t, 0.0)) / 100.0
    return float(total)


def z_for(conf_pct):
    """One-sided normal z for a confidence level (e.g. 95 → 1.645)."""
    from scipy.stats import norm
    return float(norm.ppf(min(max(conf_pct, 50.0), 99.9) / 100.0))


def parametric(weights_vec, cov, vol_mult=1.0, corr_lambda=0.0, conf_pct=95.0):
    """Base vs stressed portfolio volatility and parametric VaR.

    `cov`: annualised covariance (k×k) aligned to `weights_vec`.
    `vol_mult`: multiply each asset's volatility by this (≥1 stresses up).
    `corr_lambda`: blend the correlation matrix toward all-ones in [0,1] (1 = perfect correlation).
    Returns base/stressed annualised sigma and VaR (= z·sigma, a positive loss fraction)."""
    w = np.asarray(weights_vec, float)
    cov = np.asarray(cov, float)
    d = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
    R = cov / np.outer(d, d)
    R = np.clip(R, -1.0, 1.0)
    base_sigma = float(np.sqrt(max(w @ cov @ w, 0.0)))
    d2 = d * float(vol_mult)
    J = np.ones_like(R)
    Rs = (1.0 - corr_lambda) * R + corr_lambda * J
    np.fill_diagonal(Rs, 1.0)
    cov_s = Rs * np.outer(d2, d2)
    str_sigma = float(np.sqrt(max(w @ cov_s @ w, 0.0)))
    z = z_for(conf_pct)
    return {"base_sigma": base_sigma, "str_sigma": str_sigma,
            "base_var": z * base_sigma, "str_var": z * str_sigma, "z": z}
