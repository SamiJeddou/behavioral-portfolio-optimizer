# © 2026 Sami Jeddou. All rights reserved.
# Published publicly for demonstration and evaluation only — no license is granted.
# Copying, modification, redistribution, or reuse (in whole or in part) without the
# author's prior written permission is prohibited.
"""Live-portfolio analytics — UI-free.

Turns a saved portfolio (a dated *rebalance log* + a benchmark) plus daily close prices
into a performance track and the headline statistics the Live Portfolio view shows:

  * the daily portfolio return series, stitched segment-by-segment from the rebalance log
    (each segment is rebalanced daily to that segment's target weights),
  * an equity curve (growth of 1) for the portfolio and the benchmark,
  * risk/return metrics — cumulative & annualised return, volatility, Sharpe, max drawdown,
    and daily VaR / CVaR at level α,
  * a CAPM regression of the portfolio's excess return on the benchmark's — realised
    annualised alpha (Jensen's α), beta and R².

No Streamlit, no I/O — feed it a price DataFrame (e.g. from core.markets.fetch_close_prices).
"""
import numpy as np
import pandas as pd

__all__ = [
    "RF_ANNUAL", "PERIODS_PER_YEAR",
    "holdings_in_log", "weights_frame", "portfolio_returns",
    "equity_curve", "metrics", "capm", "analyze",
]

RF_ANNUAL = 0.03            # default annual risk-free rate (matches the rest of the app)
PERIODS_PER_YEAR = 252      # trading days


def holdings_in_log(rebalance_log):
    """Ordered list of every security that appears in any rebalance entry's weights."""
    seen = []
    for e in rebalance_log:
        for t in (e.get("weights") or {}):
            if t not in seen:
                seen.append(t)
    return seen


def weights_frame(dates, rebalance_log, tickers):
    """A weights DataFrame (index=`dates`, columns=`tickers`) where each date carries the
    weights of the most recent rebalance entry on or before it. Weights are normalised
    (long-only, sum to 1) over the supplied tickers; dates before the first entry are 0."""
    dates = pd.DatetimeIndex(dates)
    W = pd.DataFrame(0.0, index=dates, columns=list(tickers))
    log = sorted(rebalance_log, key=lambda e: pd.Timestamp(e["date"]))
    for e in log:
        d = pd.Timestamp(e["date"])
        raw = e.get("weights") or {}
        row = pd.Series({t: max(float(raw.get(t, 0.0)), 0.0) for t in tickers})
        s = float(row.sum())
        if s > 0:
            row = row / s
        W.loc[dates >= d, :] = row.values
    return W


def portfolio_returns(prices, rebalance_log):
    """Daily portfolio return series from `prices` (DataFrame: index=dates, columns=tickers)
    and a dated rebalance log. The series starts at the first rebalance (inception) date.

    Positions are **bought and held between rebalance dates** — weights are reset to the target
    on each rebalance date and then drift with each holding's own performance until the next one.
    This is the correct model for a tracker (you don't trade daily), and it is essential once
    derivatives are held: a daily-rebalanced weight on a position whose value collapses toward
    zero near expiry would book ruinous, unrealistic losses, whereas buy-and-hold caps a long
    option's loss at its premium (its weight)."""
    if not rebalance_log:
        return pd.Series(dtype=float)
    tickers = [t for t in holdings_in_log(rebalance_log) if t in prices.columns]
    if not tickers:
        return pd.Series(dtype=float)
    log = sorted(rebalance_log, key=lambda e: pd.Timestamp(e["date"]))
    start = pd.Timestamp(log[0]["date"])
    rets = prices[tickers].pct_change().dropna(how="all")
    rets = rets[rets.index >= start]
    if rets.empty:
        return pd.Series(dtype=float)
    rets = rets.fillna(0.0)
    bounds = [pd.Timestamp(e["date"]) for e in log] + [pd.Timestamp.max]
    port = pd.Series(np.nan, index=rets.index)
    for i, e in enumerate(log):
        d0, d1 = bounds[i], bounds[i + 1]
        seg = rets.loc[(rets.index >= d0) & (rets.index < d1)]
        if seg.empty:
            continue
        raw = pd.Series({t: max(float((e.get("weights") or {}).get(t, 0.0)), 0.0) for t in tickers})
        s = float(raw.sum())
        if s <= 0:
            continue
        w0 = (raw / s).values
        growth = (1.0 + seg).cumprod()                  # buy-and-hold growth within the segment
        V = (growth.values * w0).sum(axis=1)            # portfolio value (base 1 at segment start)
        prev = np.concatenate([[1.0], V[:-1]])
        port.loc[seg.index] = V / prev - 1.0            # daily returns; first day vs base 1
    port = port.dropna()
    port.name = "portfolio"
    return port


def equity_curve(returns):
    """Growth of 1 over the return series (compounded)."""
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return pd.Series(dtype=float)
    return (1.0 + returns).cumprod()


def metrics(returns, rf=RF_ANNUAL, ppy=PERIODS_PER_YEAR, alpha=0.05):
    """Headline risk/return stats for a daily return series. Returns signed decimals.

    cumulative — total compounded return over the window
    annualised — geometric annualised return
    vol        — annualised volatility
    sharpe     — (annualised − rf) / vol
    max_drawdown — worst peak-to-trough decline (negative)
    var / cvar — daily α-quantile return and its tail average (negative)
    n / years  — sample size and span
    """
    r = pd.Series(returns).dropna()
    n = int(len(r))
    if n == 0:
        return {"cumulative": 0.0, "annualised": 0.0, "vol": 0.0, "sharpe": float("nan"),
                "max_drawdown": 0.0, "var": 0.0, "cvar": 0.0, "n": 0, "years": 0.0}
    eq = (1.0 + r).cumprod()
    cumulative = float(eq.iloc[-1] - 1.0)
    years = n / float(ppy)
    annualised = float((eq.iloc[-1]) ** (1.0 / years) - 1.0) if years > 0 and eq.iloc[-1] > 0 else 0.0
    vol = float(r.std(ddof=1) * np.sqrt(ppy)) if n > 1 else 0.0
    sharpe = (annualised - rf) / vol if vol > 1e-12 else float("nan")
    peak = eq.cummax()
    max_dd = float((eq / peak - 1.0).min())
    var = float(np.quantile(r.values, alpha))
    tail = r.values[r.values <= var]
    cvar = float(tail.mean()) if tail.size else var
    return {"cumulative": cumulative, "annualised": annualised, "vol": vol, "sharpe": sharpe,
            "max_drawdown": max_dd, "var": var, "cvar": cvar, "n": n, "years": years}


def capm(port_returns, bench_returns, rf=RF_ANNUAL, ppy=PERIODS_PER_YEAR):
    """CAPM regression of portfolio excess return on benchmark excess return.

    Returns annualised Jensen's alpha, beta, R² and the overlap sample size. Alpha is the
    regression intercept (daily) annualised by the period count."""
    p = pd.Series(port_returns).dropna()
    b = pd.Series(bench_returns).dropna()
    df = pd.concat([p.rename("p"), b.rename("b")], axis=1, join="inner").dropna()
    n = int(len(df))
    if n < 3:
        return {"alpha": float("nan"), "beta": float("nan"), "r2": float("nan"), "n": n}
    rf_d = rf / float(ppy)
    ex_p = df["p"].values - rf_d
    ex_b = df["b"].values - rf_d
    beta, alpha_d = np.polyfit(ex_b, ex_p, 1)
    # R² from the correlation between the regressor and regressand
    corr = np.corrcoef(ex_b, ex_p)[0, 1]
    r2 = float(corr ** 2)
    alpha_annual = float(alpha_d * ppy)
    return {"alpha": alpha_annual, "beta": float(beta), "r2": r2, "n": n}


def analyze(prices, rebalance_log, benchmark=None, rf=RF_ANNUAL, ppy=PERIODS_PER_YEAR, alpha=0.05):
    """One-shot analysis for the view. `prices` should contain every holding (and the
    benchmark, if given) as columns. Returns a dict with the portfolio/benchmark return
    series, equity curves, metric dicts and the CAPM result."""
    port = portfolio_returns(prices, rebalance_log)
    out = {
        "port_returns": port,
        "port_equity": equity_curve(port),
        "metrics": metrics(port, rf=rf, ppy=ppy, alpha=alpha),
        "bench_returns": None, "bench_equity": None, "bench_metrics": None, "capm": None,
        "inception": (min(pd.Timestamp(e["date"]) for e in rebalance_log) if rebalance_log else None),
        "as_of": (port.index[-1] if len(port) else None),
    }
    if benchmark and benchmark in prices.columns and len(port):
        bench = prices[benchmark].pct_change().dropna()
        bench = bench[bench.index >= port.index[0]]
        out["bench_returns"] = bench
        out["bench_equity"] = equity_curve(bench)
        out["bench_metrics"] = metrics(bench, rf=rf, ppy=ppy, alpha=alpha)
        out["capm"] = capm(port, bench, rf=rf, ppy=ppy)
    return out
