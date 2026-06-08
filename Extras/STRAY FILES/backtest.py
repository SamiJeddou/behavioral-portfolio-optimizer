"""Out-of-sample backtest performance math — UI-free.

Buy-and-hold portfolio value paths, realised window/annualised return & volatility with
loss-threshold breach flag, and realised CAPM alpha/beta/R-squared against a benchmark.
Sits on top of core.pricing (mark-to-market) and core.grid (weights). No Streamlit.
"""
import numpy as np

__all__ = ["_bt_portfolio_path", "_bt_metrics", "_capm_alpha_beta", "run_backtest"]


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


def run_backtest(tickers, construction, evaluation, constraint, derivative,
                 benchmark=None, freq="Daily", rf=0.0, resolution="standard",
                 source=None):
    """Out-of-sample backtest: build the optimum on a construction window, hold it through
    an evaluation window (buy-and-hold; the derivative is marked to market), and compare
    expected vs realised performance for a no-derivative (P1) and with-derivative (P2) book.

    tickers: list of symbols. construction/evaluation: (start, end) ISO dates or date objects.
    constraint: core.types.Constraint. derivative: core.types.DerivativeSpec (required;
    underlying_idx may be -1 to auto-pick the highest-volatility security).
    benchmark: optional symbol for realised alpha/beta. freq: 'Daily' or 'Monthly'.
    rf: annual risk-free rate. resolution: 'fast' | 'standard' | 'high'.
    source: a core.markets.DataSource (defaults to YFinanceSource).
    Returns a core.types.BacktestResult.
    """
    import datetime as _dt
    import numpy as _np
    from core.types import BacktestResult, PortfolioResult
    from core.grid import run_opt
    from core.pricing import build_der_config, _bt_legs, mtm_gross_path
    from core.markets import stats_from_prices, corr_to_cov, YFinanceSource

    if derivative is None:
        raise ValueError("run_backtest requires a derivative (it compares P1 no-derivative "
                         "vs P2 with-derivative).")
    if source is None:
        source = YFinanceSource()

    _RES = {"fast": (21, 15), "standard": (35, 50), "high": (51, 99)}
    _KIND = {"var": "var", "es_thesis": "es", "es_rigorous": "es_rigorous"}
    m_bt, mp_bt = _RES.get(resolution, _RES["standard"])
    ct = _KIND.get(constraint.kind, "var")

    con_start, con_end = construction
    ev_start, ev_end = evaluation

    def _d(x):
        return x if hasattr(x, "toordinal") else _dt.date.fromisoformat(str(x))
    T_years = (_d(ev_end) - _d(ev_start)).days / 365.25
    factor = 252 if freq == "Daily" else 12

    # ── construction window: stats + optimise ─────────────────────────────────
    con_px = source.prices(tickers, con_start, con_end)
    means, sigs, corr, names, _ = stats_from_prices(con_px, freq)
    if len(names) < 2:
        raise RuntimeError("Fewer than two usable securities after cleaning.")
    cov = corr_to_cov(sigs, corr)
    sigs = _np.asarray(sigs, float)

    u_idx = derivative.underlying_idx
    if u_idx is None or u_idx < 0 or u_idx >= len(names):
        u_idx = int(_np.argmax(sigs))
    vol_u = float(derivative.vol_override) if derivative.vol_override is not None else float(sigs[u_idx])
    params = dict(derivative.params)
    params["vol"] = vol_u; params["r"] = derivative.r; params["T"] = T_years
    der_cfg = build_der_config(derivative.type, params, sigs, u_idx)

    nd_res, _ = run_opt(means, sigs, cov, None,    constraint.H, constraint.alpha,
                        m_bt, mp_bt, ct, L=constraint.L)
    dr_res, _ = run_opt(means, sigs, cov, der_cfg, constraint.H, constraint.alpha,
                        m_bt, mp_bt, ct, L=constraint.L)

    # ── evaluation window: hold fixed weights, mark derivative to market ──────
    ev_px = source.prices(tickers, ev_start, ev_end)
    missing = [nm for nm in names if nm not in ev_px.columns]
    if missing:
        raise RuntimeError(f"No evaluation-period data for: {missing}")
    ev_px = ev_px[names].ffill().dropna()
    if freq == "Monthly":
        ev_px = ev_px.resample("ME").last().dropna()
    if len(ev_px) < 3:
        raise RuntimeError("Insufficient evaluation-period observations.")

    sec_gross = ev_px.values / ev_px.values[0]
    spot_path = ev_px[names[u_idx]].values
    legs, norm_mode, prem = _bt_legs(derivative.type, params)
    g_path = mtm_gross_path(legs, norm_mode, prem, spot_path, T_years, vol_u, derivative.r)

    w1 = _np.asarray(nd_res["weights"], float)
    w2 = _np.asarray(dr_res["weights"], float)
    w2_sec, w2_der = w2[:-1], float(w2[-1])
    pv1 = _bt_portfolio_path(sec_gross, w1)
    pv2 = _bt_portfolio_path(sec_gross, w2_sec, der_gross=g_path, w_der=w2_der)

    cum1, ann1, vol1, br1 = _bt_metrics(pv1, factor, constraint.H, T_years)
    cum2, ann2, vol2, br2 = _bt_metrics(pv2, factor, constraint.H, T_years)

    # ── plain-language verdict ────────────────────────────────────────────────
    H = constraint.H
    verdict = []
    d_ret = ann2 - ann1
    d_vol = vol2 - vol1
    if d_ret > 0.005:
        verdict.append(f"The derivative added {d_ret:.2%} of realised annual return versus "
                       f"the no-derivative portfolio.")
    elif d_ret < -0.005:
        verdict.append(f"The derivative cost {-d_ret:.2%} of realised annual return this window.")
    else:
        verdict.append("Realised returns of the two portfolios were broadly similar.")
    if not _np.isnan(d_vol):
        if d_vol < -0.005:
            verdict.append(f"It also reduced realised volatility by {-d_vol:.2%}.")
        elif d_vol > 0.005:
            verdict.append(f"It raised realised volatility by {d_vol:.2%}.")
    if br1 and not br2:
        verdict.append(f"P1 breached the {H:.0%} loss threshold while P2 did not - "
                       f"the protection held this window.")
    elif br2 and not br1:
        verdict.append(f"P2 breached the {H:.0%} threshold while P1 did not.")
    elif br1 and br2:
        verdict.append(f"Both portfolios finished below the {H:.0%} threshold.")
    else:
        verdict.append(f"Neither portfolio finished below the {H:.0%} threshold.")

    # ── optional realised alpha / beta vs a benchmark ─────────────────────────
    alpha_beta = None
    if benchmark:
        try:
            bpx = source.prices([benchmark], ev_start, ev_end)
            bser = bpx.iloc[:, 0].reindex(ev_px.index).ffill().bfill()
            bench_ret = bser.pct_change().values[1:]
            rf_per = rf / factor
            r1 = pv1[1:] / pv1[:-1] - 1.0
            r2 = pv2[1:] / pv2[:-1] - 1.0

            def _f(x):
                return None if (x != x) else float(x)

            def _ab(series):
                b, a, r2_ = _capm_alpha_beta(series, bench_ret, rf_per, factor)
                return {"beta": _f(b), "alpha": _f(a), "r2": _f(r2_)}

            alpha_beta = {"benchmark": benchmark, "P1": _ab(r1), "P2": _ab(r2), "securities": {}}
            sec_ret = ev_px[names].pct_change().values[1:]
            for j, nm in enumerate(names):
                alpha_beta["securities"][nm] = _ab(sec_ret[:, j])
        except Exception:
            alpha_beta = None

    p1 = PortfolioResult.from_grid(nd_res, list(names), feasible=True)
    p2 = PortfolioResult.from_grid(dr_res, list(names) + [derivative.label or derivative.type],
                                   feasible=True)
    realised = {
        "P1": {"expected_return": float(nd_res["expected_return"]), "realised_return": float(ann1),
               "expected_vol": float(nd_res["std_dev"]), "realised_vol": float(vol1),
               "window_return": float(cum1), "breached_H": bool(br1)},
        "P2": {"expected_return": float(dr_res["expected_return"]), "realised_return": float(ann2),
               "expected_vol": float(dr_res["std_dev"]), "realised_vol": float(vol2),
               "window_return": float(cum2), "breached_H": bool(br2)},
        "H": float(H), "T_years": float(T_years),
    }
    paths = {"dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in ev_px.index],
             "pv1": [float(x) for x in pv1], "pv2": [float(x) for x in pv2]}
    return BacktestResult(p1=p1, p2=p2, underlying=names[u_idx], derivative_weight=w2_der,
                          realised=realised, paths=paths, verdict=verdict, alpha_beta=alpha_beta)
