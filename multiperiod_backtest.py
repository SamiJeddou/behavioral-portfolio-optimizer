#!/usr/bin/env python3
"""Single-period vs multi-period (periodic re-optimisation) robustness.

The main results hold each window's portfolio buy-and-hold to the horizon (single-period). This
checks whether re-optimising WITHIN the window changes the out-of-sample tail: for each outer
window it compares
  BUY-HOLD : optimise once on the conditioning window, hold to the horizon (the paper's design)
  RE-OPT   : re-optimise every --reopt-months on a trailing conditioning window, compounding
If the breach rate barely moves, the single-period assumption is shown to be representative.
Gross of costs (re-optimisation trades more, so its realistic net would be slightly lower).

    python multiperiod_backtest.py --benchmark SPY --start 2005-01-01 --wmax 0.20 --reopt-months 3
"""
import argparse
import numpy as np, pandas as pd
from rolling_backtest import get_prices, wilson
from core.markets import stats_from_prices
from core.optimise import optimise_scenario
from core.types import AssetUniverse, Constraint

BAL = ("AAPL MSFT NVDA GOOGL AMZN JPM XOM JNJ PG KO WMT HD UNH CVX PFE MRK "
       "INTC CSCO ORCL PEP MCD IBM DIS BAC TLT IEF LQD GLD").split()


def opt_w(Pcond, uni, H, alpha, L, wmax, scenarios):
    try:
        means, sigs, corr, names, _ = stats_from_prices(Pcond[uni], "Daily")
    except Exception:
        return None, None
    if len(names) < len(uni):
        return None, None
    u = AssetUniverse(names=names, means=np.array(means), sigmas=np.asarray(sigs, float), corr=np.array(corr))
    r = optimise_scenario(u, Constraint(kind="es_rigorous", H=H, alpha=alpha, L=L),
                          scenarios=scenarios, w_max=wmax, seed=0)
    if (not r.feasible) or np.asarray(r.weights).size == 0:
        return None, names
    return np.asarray(r.weights, float), names


def run(df, tickers, bench, H, alpha, L, wmax, scenarios, con_m, ev_m, step_m, reopt_m):
    off = lambda k: pd.DateOffset(months=k)
    idx = df.index; rows = []; s = idx[0]; skip = 0
    while True:
        c0 = s; c1 = c0 + off(con_m); e1 = c1 + off(ev_m)
        if e1 > idx[-1]:
            break
        nxt = s + off(step_m)
        Pe = df.loc[c1:e1].dropna()
        if len(Pe) < 20 or any(t not in Pe.columns for t in tickers) or any(Pe[t].isna().any() for t in tickers):
            s = nxt; continue
        wbh, names = opt_w(df.loc[c0:c1], tickers, H, alpha, L, wmax, scenarios)
        if wbh is None:
            skip += 1; s = nxt; continue
        # BUY-HOLD terminal return
        g = Pe[names].values / Pe[names].values[0]
        ret_bh = float(g[-1] @ wbh) - 1.0
        # RE-OPT: compound sub-period returns, re-optimising each sub-period on a trailing window
        V = 1.0; t = c1
        while t < e1:
            t1 = min(t + off(reopt_m), e1)
            cond = df.loc[t - off(con_m):t, tickers].dropna()
            w = wbh
            if len(cond) >= 60:
                ww, _ = opt_w(cond, tickers, H, alpha, L, wmax, scenarios)
                if ww is not None:
                    w = ww
            sub = df.loc[t:t1, names].dropna()
            if len(sub) < 2:
                break
            V *= float((sub.values[-1] / sub.values[0]) @ w)
            t = t1
        ret_ro = V - 1.0
        rows.append(dict(start=str(c1.date()), bh=ret_bh, br_bh=int(ret_bh < H),
                         ro=ret_ro, br_ro=int(ret_ro < H)))
        s = nxt
    print(f"(skipped {skip} infeasible/short windows; re-opt every {reopt_m} months)")
    return pd.DataFrame(rows)


def report(R, H):
    n = len(R)
    if n == 0:
        print("No windows."); return
    for tag, k in (("SINGLE-PERIOD (buy-hold)", "bh"), ("MULTI-PERIOD (re-opt)", "ro")):
        bk = int(R[f"br_{k}"].sum()); lo, hi = wilson(bk, n)
        print(f"[{tag}] breach {bk}/{n}={bk/n:.1%} [{lo:.1%},{hi:.1%}] | mean {R[k].mean():.1%} | worst {R[k].min():.1%}")
    print(f"\nDelta (re-opt - buy-hold): breach {R.br_ro.mean()-R.br_bh.mean():+.1%} | "
          f"mean ret {R.ro.mean()-R.bh.mean():+.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=BAL)
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--start", default="2005-01-01"); p.add_argument("--end", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--H", type=float, default=-0.10); p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--L", type=float, default=-0.20); p.add_argument("--wmax", type=float, default=0.20)
    p.add_argument("--scenarios", type=int, default=8000)
    p.add_argument("--con", type=int, default=12); p.add_argument("--ev", type=int, default=12)
    p.add_argument("--step", type=int, default=6); p.add_argument("--reopt-months", type=int, default=3)
    p.add_argument("--out", default="multiperiod_backtest.csv")
    a = p.parse_args()
    df = get_prices(a.tickers, a.benchmark, a.start, a.end, a.csv)
    print(f"Loaded {len(df)} rows, {df.index[0].date()}..{df.index[-1].date()}; {len(a.tickers)} instruments + {a.benchmark}")
    R = run(df, a.tickers, a.benchmark, a.H, a.alpha, a.L, a.wmax, a.scenarios,
            a.con, a.ev, a.step, a.reopt_months)
    report(R, a.H)
    R.to_csv(a.out, index=False); print(f"-> {a.out}")
