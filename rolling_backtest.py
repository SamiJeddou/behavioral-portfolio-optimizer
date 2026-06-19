#!/usr/bin/env python3
"""Rolling, multi-window out-of-sample backtest of the goal-based optimiser.

Two engines:
  * grid     -- exact thesis grid (practical to ~4 assets); VaR floor P(r<H)<=alpha.
  * scenario -- scalable Monte-Carlo + CVaR LP (large universes); CVaR_alpha >= L.
  * auto     -- grid if <=4 tickers, else scenario.

Across many overlapping windows it reports the breach frequency of the downside floor with a
Wilson 95% CI, realised alpha vs a benchmark with a one-sample t-test, and the return distribution.

DATA: live (yfinance, run locally) or offline from a CSV (date index, one column per ticker incl.
the benchmark):
    python rolling_backtest.py --tickers AAPL MSFT JPM --benchmark SPY --start 2005-01-01
    python rolling_backtest.py --engine scenario --wmax 0.20 \
        --tickers AAPL MSFT JPM XOM JNJ PG KO WMT HD V MA UNH ... --benchmark SPY
"""
import argparse, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from core.markets import stats_from_prices, corr_to_cov
from core.grid import run_opt
from core.pricing import build_der_config, _bt_legs, mtm_gross_path, live_derivative_series
from core.backtest import _bt_portfolio_path, _bt_metrics, _capm_alpha_beta
from core.optimise import optimise_scenario
from core.scenario import mc_max_return_cvar
from core.types import AssetUniverse, Constraint

_RES = {"fast": (21, 15), "standard": (35, 50), "high": (51, 99)}

def get_prices(tickers, bench, start, end, csv=None):
    cols = list(dict.fromkeys(list(tickers) + [bench]))
    if csv:
        df = pd.read_csv(csv, index_col=0, parse_dates=True).sort_index()
        miss = [c for c in cols if c not in df.columns]
        if miss: raise SystemExit(f"CSV missing columns: {miss}")
        return df[cols].dropna(how="all")
    import yfinance as yf
    raw = yf.download(cols, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return px[cols].dropna(how="all")

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0, c-h), min(1, c+h))


def _empirical_scenarios(Rd, S, h_days, block, rng):
    """Block-bootstrap S horizon-return scenarios from conditioning daily returns Rd (Td x N)."""
    Td, N = Rd.shape
    nblk = int(np.ceil(h_days / block))
    starts = rng.integers(0, Td, size=(S, nblk))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % Td
    idx = idx.reshape(S, nblk * block)[:, :h_days]
    acc = np.ones((S, N))
    for d in range(h_days):
        acc *= 1.0 + Rd[idx[:, d]]
    return acc - 1.0

def run(df, tickers, bench, H, alpha, con_m, ev_m, step_m, rf, resolution, put_strike,
        engine, L, wmax, scenarios, hedge="none", h_s1=0.90, h_s2=1.10, h_frac=0.05,
        tc_bps=0.0, h_vol_add=0.0, copula="gaussian", dof=5,
        hedge_vol_source="realized", vix=None):
    m, mp = _RES[resolution]; factor = 252
    use_scn = engine == "scenario" or (engine == "auto" and len(tickers) > 4)
    do_hedge = hedge not in ("none", None)
    idx = df.index; rows = []; infeasible = 0; s = idx[0]
    tc = tc_bps / 1e4; w_prev = {}
    while True:
        c0 = s; c1 = c0 + pd.DateOffset(months=con_m); e1 = c1 + pd.DateOffset(months=ev_m)
        if e1 > idx[-1]: break
        Pc = df.loc[c0:c1, tickers].dropna(); Pe = df.loc[c1:e1].dropna()
        nxt = s + pd.DateOffset(months=step_m)
        if len(Pc) < 60 or len(Pe) < 20 or Pe[tickers].shape[1] < len(tickers): s = nxt; continue
        means, sigs, corr, names, _ = stats_from_prices(Pc, "Daily")
        if len(names) < len(tickers): s = nxt; continue
        sigs = np.asarray(sigs, float); T = (Pe.index[-1] - Pe.index[0]).days / 365.25
        secg = Pe[names].values / Pe[names].values[0]
        try:
            if use_scn:
                if copula == "empirical":
                    Rd = Pc[names].pct_change().dropna().values
                    h_days = max(20, int(round(252 * ev_m / 12)))
                    R_emp = _empirical_scenarios(Rd, scenarios, h_days, 10, np.random.default_rng(0))
                    w1, _, _, _ = mc_max_return_cvar(R_emp, alpha, L, w_max=wmax)
                    if w1 is None:
                        infeasible += 1; s = nxt; continue
                    w1 = np.asarray(w1, float)
                else:
                    uni = AssetUniverse(names=names, means=np.array(means), sigmas=sigs, corr=np.array(corr))
                    r = optimise_scenario(uni, Constraint(kind="es_rigorous", H=H, alpha=alpha, L=L),
                                          scenarios=scenarios, copula=copula, dof=dof, w_max=wmax, seed=0)
                    if (not r.feasible) or np.asarray(r.weights).size == 0:
                        infeasible += 1; s = nxt; continue
                    w1 = np.asarray(r.weights, float)
            else:
                nd, _ = run_opt(means, sigs, corr_to_cov(sigs, corr), None, H, alpha, m, mp, "var")
                w1 = np.asarray(nd["weights"], float)
        except Exception:
            infeasible += 1; s = nxt; continue
        pv1 = _bt_portfolio_path(secg, w1)
        cum1, ann1, vol1, br1 = _bt_metrics(pv1, factor, H, T)
        # transaction-cost haircut on equity-book turnover vs the previously held weights
        wprev_vec = np.array([w_prev.get(nm, 0.0) for nm in names])
        turn = float(np.sum(np.abs(w1 - wprev_vec))); w_prev = {nm: float(wi) for nm, wi in zip(names, w1)}
        term_net = pv1[-1] * (1 - tc * turn)
        cum1n = float(term_net - 1.0); br1n = int(term_net < 1.0 + H)
        # optional protective-put overlay (grid engine only)
        cum2, br2, w2d = cum1, br1, 0.0
        if not use_scn:
            try:
                u = int(np.argmax(sigs)); params = {"strike": put_strike, "vol": float(sigs[u]), "r": rf, "T": T}
                dr, _ = run_opt(means, sigs, corr_to_cov(sigs, corr), build_der_config("put", params, sigs, u),
                                H, alpha, m, mp, "var")
                w2 = np.asarray(dr["weights"], float); w2s, w2d = w2[:-1], float(w2[-1])
                legs, norm, prem = _bt_legs("put", params)
                g = mtm_gross_path(legs, norm, prem, Pe[names[u]].values, T, float(sigs[u]), rf)
                pv2 = _bt_portfolio_path(secg, w2s, der_gross=g, w_der=w2d)
                cum2, _, _, br2 = _bt_metrics(pv2, factor, H, T)
            except Exception:
                cum2, br2, w2d = cum1, br1, 0.0
        # engine-agnostic INDEX-OVERLAY sleeve: allocate h_frac of capital to a standing
        # index put/collar on the benchmark, priced at the CONDITIONING-window volatility
        # (faithful entry conditions), the rest to the optimised equity book. Premium drag is
        # automatic (the sleeve decays when the hedge expires out-of-the-money).
        cumH, brH = cum1, br1
        if do_hedge:
            try:
                base = None
                if hedge_vol_source == "vix" and vix is not None:
                    try:
                        base = float(vix.asof(c1)) / 100.0
                    except Exception:
                        base = None
                if not (base and base > 0):
                    bc = df.loc[c0:c1, bench].dropna().pct_change().dropna()
                    base = float(bc.std() * np.sqrt(factor)) if len(bc) > 5 else None
                bvol = (base + h_vol_add) if base is not None else None
                ov = live_derivative_series(Pe[bench], hedge, h_s1, h_s2, T, r=rf, vol=bvol)
                if ov is not None and len(ov) >= 2:
                    ov = (ov / float(ov.iloc[0])).reindex(Pe.index).ffill().values
                    n = min(len(pv1), len(ov)); ph = (1 - h_frac) * pv1[:n] + h_frac * ov[:n]
                    cumH = float(ph[-1] - 1.0); brH = int(ph[-1] < 1.0 + H)
            except Exception:
                cumH, brH = cum1, br1
        bret = Pe[bench].pct_change().values[1:]
        b1, a1, _ = _capm_alpha_beta(pv1[1:]/pv1[:-1]-1, bret, rf/factor, factor)
        rows.append(dict(start=str(c1.date()), cum1=cum1, ann1=ann1, br1=br1,
                         cum2=cum2, br2=br2, wput=w2d, alpha1=a1, beta1=b1,
                         cumH=cumH, brH=brH, cum1n=cum1n, br1n=br1n, turnover=turn))
        s = nxt
    print(f"(engine={'scenario' if use_scn else 'grid'}; skipped {infeasible} infeasible windows)")
    return pd.DataFrame(rows)

def report(R, H):
    n = len(R)
    if n == 0: print("No valid windows."); return
    from scipy import stats as st
    bk = int(R.br1.sum()); lo, hi = wilson(bk, n); a = R.alpha1.dropna().values
    tt = st.ttest_1samp(a, 0.0) if len(a) > 2 else None
    q = R.cum1.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"\n=== Rolling out-of-sample backtest: {n} windows ===")
    print(f"Mean realised beta: {R.beta1.mean():.2f}")
    print(f"Floor breaches (realised return < {H:.0%}): {bk}/{n} = {bk/n:.1%}  [Wilson 95% CI {lo:.1%}, {hi:.1%}]")
    print(f"Realised annual alpha: mean {np.mean(a):+.2%}, median {np.median(a):+.2%}"
          + (f", t={tt.statistic:.2f}, p={tt.pvalue:.3f}" if tt else ""))
    print(f"Realised window return: mean {R.cum1.mean():.1%} | p05 {q[0.05]:.1%} | median {q[0.5]:.1%} | p95 {q[0.95]:.1%} | worst {R.cum1.min():.1%}")
    if "br1n" in R.columns and (R.cum1n != R.cum1).any():
        nk = int(R.br1n.sum()); nlo, nhi = wilson(nk, n)
        print(f"After transaction costs: breach {nk}/{n} = {nk/n:.1%}  [Wilson 95% CI {nlo:.1%}, {nhi:.1%}]"
              f"  (gross {bk/n:.1%}) | mean turnover {R.turnover.mean():.2f} | net mean ret {R.cum1n.mean():.1%}")
    if "brH" in R.columns and (R.cumH != R.cum1).any():
        hk = int(R.brH.sum()); hlo, hhi = wilson(hk, n)
        drag = R.cum1.mean() - R.cumH.mean()
        print("\n--- with index-overlay hedge ---")
        print(f"Floor breaches (hedged): {hk}/{n} = {hk/n:.1%}  [Wilson 95% CI {hlo:.1%}, {hhi:.1%}]   (unhedged {bk/n:.1%})")
        print(f"Mean return cost of the hedge: {drag:+.2%} per window (hedged mean {R.cumH.mean():.1%} | worst {R.cumH.min():.1%})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["AAPL","MSFT","JPM"])
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--start", default="2005-01-01"); p.add_argument("--end", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--engine", default="auto", choices=["auto","grid","scenario"])
    p.add_argument("--H", type=float, default=-0.10); p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--L", type=float, default=-0.20, help="CVaR floor for the scenario engine")
    p.add_argument("--wmax", type=float, default=None, help="per-asset weight cap (recommend ~0.20 for large universes)")
    p.add_argument("--scenarios", type=int, default=8000)
    p.add_argument("--con", type=int, default=12); p.add_argument("--ev", type=int, default=12)
    p.add_argument("--step", type=int, default=6); p.add_argument("--rf", type=float, default=0.03)
    p.add_argument("--resolution", default="standard"); p.add_argument("--put", type=float, default=0.90)
    p.add_argument("--hedge", default="none", choices=["none","put","safety_collar","bear_put_spread"],
                   help="index-overlay sleeve on the benchmark (priced at conditioning-window vol)")
    p.add_argument("--hedge-strike", type=float, default=0.90, help="put / lower strike (frac of spot)")
    p.add_argument("--hedge-strike2", type=float, default=1.10, help="call / upper strike (collar, spread)")
    p.add_argument("--hedge-frac", type=float, default=0.05, help="capital fraction in the hedge sleeve")
    p.add_argument("--hedge-vol-add", type=float, default=0.0, help="vol points added to hedge pricing (skew / IV premium stress, e.g. 0.05)")
    p.add_argument("--tc-bps", type=float, default=0.0, help="one-way transaction cost in bps applied to rebalancing turnover")
    p.add_argument("--copula", default="gaussian", choices=["gaussian","t","empirical"], help="scenario generator (scenario engine)")
    p.add_argument("--hedge-vol-source", default="realized", choices=["realized","vix"], help="hedge pricing vol: trailing realized, or market-implied (VIX at entry)")
    p.add_argument("--dof", type=int, default=5, help="Student-t copula degrees of freedom")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    df = get_prices(a.tickers, a.benchmark, a.start, a.end, a.csv)
    vix = None
    if a.hedge_vol_source == "vix":
        if a.csv:
            try:
                _v = pd.read_csv(a.csv, index_col=0, parse_dates=True).sort_index()
                if "^VIX" in _v.columns: vix = _v["^VIX"].dropna()
            except Exception: vix = None
        else:
            import yfinance as yf
            _vx = yf.download("^VIX", start=a.start, end=a.end, auto_adjust=False, progress=False)
            _c = _vx["Close"] if (hasattr(_vx, "columns") and "Close" in getattr(_vx, "columns", [])) else _vx
            try: vix = pd.Series(np.asarray(_c).ravel(), index=pd.to_datetime(_vx.index)).dropna()
            except Exception: vix = None
        print("VIX loaded:" , None if vix is None else f"{len(vix)} obs")
    print(f"Loaded {len(df)} rows, {df.index[0].date()}..{df.index[-1].date()}, {len(a.tickers)} assets + {a.benchmark}")
    R = run(df, a.tickers, a.benchmark, a.H, a.alpha, a.con, a.ev, a.step, a.rf, a.resolution, a.put,
            a.engine, a.L, a.wmax, a.scenarios,
            hedge=a.hedge, h_s1=a.hedge_strike, h_s2=a.hedge_strike2, h_frac=a.hedge_frac,
            tc_bps=a.tc_bps, h_vol_add=a.hedge_vol_add, copula=a.copula, dof=a.dof,
            hedge_vol_source=a.hedge_vol_source, vix=vix)
    report(R, a.H)
    out = a.out or ("rb_" + "_".join(a.tickers[:6]) + (f"_plus{len(a.tickers)-6}" if len(a.tickers) > 6 else "") + ".csv")
    R.to_csv(out, index=False); print(f"\nPer-window results -> {out}")
