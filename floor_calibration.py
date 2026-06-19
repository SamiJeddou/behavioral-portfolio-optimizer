#!/usr/bin/env python3
"""Calibrate the in-sample CVaR floor L to a TARGET out-of-sample breach rate.

The goal-based promise ("<=alpha chance of finishing below H") holds in-sample by construction
but breaches out-of-sample. This sweeps the in-sample CVaR floor L while holding the breach
reference H fixed (default -10%), and reports the realised OOS breach rate for each L -- so you
can read off the floor (and the BUFFER, L - H) needed to actually deliver, say, 5% breaches out
of sample on a crisis-spanning book. One data download, many floors.

    python floor_calibration.py --benchmark SPY --start 2005-01-01 --wmax 0.20
"""
import argparse
import numpy as np, pandas as pd
from rolling_backtest import get_prices, run, wilson

# default = the balanced multi-asset reference book (24 equities + rates/credit + gold)
BAL = ("AAPL MSFT NVDA GOOGL AMZN JPM XOM JNJ PG KO WMT HD UNH CVX PFE MRK "
       "INTC CSCO ORCL PEP MCD IBM DIS BAC TLT IEF LQD GLD").split()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=BAL)
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--start", default="2005-01-01"); p.add_argument("--end", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--H", type=float, default=-0.10, help="breach reference (the stated goal)")
    p.add_argument("--alpha", type=float, default=0.05, help="target/tail probability")
    p.add_argument("--wmax", type=float, default=0.20); p.add_argument("--scenarios", type=int, default=8000)
    p.add_argument("--con", type=int, default=12); p.add_argument("--ev", type=int, default=12)
    p.add_argument("--step", type=int, default=6); p.add_argument("--rf", type=float, default=0.03)
    p.add_argument("--tc-bps", type=float, default=10.0)
    p.add_argument("--Lgrid", nargs="+", type=float,
                   default=[-0.10, -0.125, -0.15, -0.175, -0.20, -0.25, -0.30])
    p.add_argument("--out", default="floor_calibration.csv")
    a = p.parse_args()

    df = get_prices(a.tickers, a.benchmark, a.start, a.end, a.csv)
    print(f"Loaded {len(df)} rows, {df.index[0].date()}..{df.index[-1].date()}; "
          f"{len(a.tickers)} instruments + {a.benchmark}")
    rows = []
    for L in a.Lgrid:
        R = run(df, a.tickers, a.benchmark, a.H, a.alpha, a.con, a.ev, a.step, a.rf,
                "standard", 0.90, "scenario", L, a.wmax, a.scenarios, tc_bps=a.tc_bps)
        n = len(R)
        if n == 0:
            print(f"  L={L:+.3f}: no feasible windows"); continue
        bk = int(R.br1n.sum() if "br1n" in R.columns else R.br1.sum())
        lo, hi = wilson(bk, n)
        meanret = float(R.cum1n.mean() if "cum1n" in R.columns else R.cum1.mean())
        rows.append(dict(L=L, buffer=L - a.H, n=n, breach=bk / n, blo=lo, bhi=hi,
                         meanret=meanret, worst=float(R.cum1.min()), alpha=float(R.alpha1.mean())))
    T = pd.DataFrame(rows).sort_values("L", ascending=False).reset_index(drop=True)
    print(f"\n=== Floor calibration: OOS breach of {a.H:.0%} vs in-sample CVaR floor L "
          f"(book of {len(a.tickers)}, {a.tc_bps:.0f}bps costs) ===")
    print("   L (floor)  buffer(L-H)   OOS breach [95% CI]      mean ret   worst    alpha")
    for _, r in T.iterrows():
        print(f"   {r.L:+.3f}     {r.buffer:+.3f}       {r.breach:5.1%} [{r.blo:4.1%},{r.bhi:5.1%}]"
              f"     {r.meanret:6.1%}   {r.worst:6.1%}   {r.alpha:+.1%}")
    # crude crossing estimate for the target
    tgt = a.alpha
    below = T[T.breach <= tgt]
    if len(below):
        Lstar = below.iloc[0].L
        print(f"\n  -> first floor achieving OOS breach <= {tgt:.0%}: L* = {Lstar:+.3f} "
              f"(buffer {Lstar - a.H:+.3f} vs the {a.H:.0%} goal)")
    else:
        print(f"\n  -> no swept floor reached {tgt:.0%}; tighten the L grid (less negative).")
    T.to_csv(a.out, index=False); print(f"-> {a.out}")
