"""Out-of-sample backtest performance math — UI-free.

Buy-and-hold portfolio value paths, realised window/annualised return & volatility with
loss-threshold breach flag, and realised CAPM alpha/beta/R-squared against a benchmark.
Sits on top of core.pricing (mark-to-market) and core.grid (weights). No Streamlit.
"""
import numpy as np

__all__ = ["_bt_portfolio_path", "_bt_metrics", "_capm_alpha_beta"]


def _bt_portfolio_path(sec_gross, w_sec, der_gross=None, w_der=0.0):
    """Buy-and-hold portfolio value path, normalised to start at 1.0."""
    pv = (np.asarray(sec_gross, dtype=float) * np.asarray(w_sec, dtype=float)).sum(axis=1)
    if der_gross is not None:
        pv = pv + float(w_der) * np.asarray(der_gross, dtype=float)
    pv = pv / pv[0]
    return pv

def _bt_metrics(pv, factor, H, T_years):
    """Realised window return, annualised return, annualised vol, breach flag."""
    rets = pv[1:] / pv[:-1] - 1.0
    cum = float(pv[-1] / pv[0] - 1.0)
    ann_ret = float((1.0 + cum) ** (1.0 / max(T_years, 1e-6)) - 1.0)
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(factor)) if len(rets) > 1 else float('nan')
    return cum, ann_ret, ann_vol, bool(cum < H)

def _capm_alpha_beta(r_asset, r_bench, rf_per_period, factor):
    """Realised CAPM regression of one return series on a benchmark over a window.

    r_asset, r_bench : aligned 1-D arrays of per-period returns.
    rf_per_period    : per-period risk-free rate (annual / factor).
    factor           : periods per year, used to annualise the alpha intercept.
    Returns (beta, alpha_annual, r2); any component is NaN where undefined.
    """
    a = np.asarray(r_asset, dtype=float)
    m = np.asarray(r_bench, dtype=float)
    n = min(a.size, m.size)
    a, m = a[:n], m[:n]
    mask = np.isfinite(a) & np.isfinite(m)
    a, m = a[mask], m[mask]
    if a.size < 3:
        return float('nan'), float('nan'), float('nan')
    ae = a - rf_per_period
    me = m - rf_per_period
    var_m = float(np.var(me))            # population variance (ddof=0)
    if not np.isfinite(var_m) or var_m < 1e-12:
        return float('nan'), float('nan'), float('nan')
    beta = float(np.cov(ae, me, ddof=0)[0, 1] / var_m)
    alpha_annual = float((ae.mean() - beta * me.mean()) * factor)
    r2 = float(np.corrcoef(ae, me)[0, 1] ** 2) if ae.std() > 0 and me.std() > 0 else float('nan')
    return beta, alpha_annual, r2
