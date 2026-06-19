#!/usr/bin/env python3
"""Scenario-generation methodology comparison (a data-driven robustness study).

Does the WAY scenarios are generated change out-of-sample tail control? Runs the same rolling
backtest under four generators on identical windows, floor and data:
  gaussian   : Gaussian copula, Normal marginals (the thesis assumption)
  t5 / t3    : Student-t copula (dof 5, 3) -- progressively fatter JOINT tails
  empirical  : non-parametric block-bootstrap of conditioning-window returns (no parametric model)
Reports OOS breach of the -10% floor, mean return, worst window, alpha and beta for each.

    python scengen_experiment.py --benchmark SPY --start 2005-01-01 --wmax 0.20
"""
import argparse
import numpy as np, pandas as pd
from rolling_backtest import get_prices, run, wilson

BAL = ("AAPL MSFT NVDA GOOGL AMZN JPM XOM JNJ PG KO WMT HD UNH CVX PFE MRK "
       "INTC CSCO ORCL PEP MCD IBM DIS BAC TLT IEF LQD GLD").split()

GENS = [("gaussian", "gaussian", 5), ("t5", "t", 5), ("t3", "t", 3), ("empirical", "empirical", 5)]

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
    p.add_argument("--step", type=int, default=6); p.add_argument("--rf", type=float, default=0.03)
    p.add_argument("--tc-bps", type=float, default=10.0)
    p.add_argument("--out", default="scengen_experiment.csv")
    a = p.parse_args()
    df = get_prices(a.tickers, a.benchmark, a.start, a.end, a.csv)
    print(f"Loaded {len(df)} rows, {df.index[0].date()}..{df.index[-1].date()}; "
          f"{len(a.tickers)} instruments + {a.benchmark}")
    rows = []
    for label, cop, dof in GENS:
        R = run(df, a.tickers, a.benchmark, a.H, a.alpha, a.con, a.ev, a.step, a.rf,
                "standard", 0.90, "scenario", a.L, a.wmax, a.scenarios,
                tc_bps=a.tc_bps, copula=cop, dof=dof)
        n = len(R)
        if n == 0:
            print(f"  {label}: no feasible windows"); continue
        bk = int(R.br1n.sum() if "br1n" in R.columns else R.br1.sum()); lo, hi = wilson(bk, n)
        mr = float(R.cum1n.mean() if "cum1n" in R.columns else R.cum1.mean())
        rows.append(dict(gen=label, n=n, breach=bk / n, blo=lo, bhi=hi, mean=mr,
                         worst=float(R.cum1.min()), alpha=float(R.alpha1.mean()), beta=float(R.beta1.mean())))
    T = pd.DataFrame(rows)
    print(f"\n=== Scenario-generation comparison (balanced book, ref {a.H:.0%}, {a.tc_bps:.0f}bps) ===")
    print("  generator   OOS breach [95% CI]       mean ret   worst    alpha   beta")
    for _, r in T.iterrows():
        print(f"  {r.gen:9} {r.breach:5.1%} [{r.blo:4.1%},{r.bhi:5.1%}]    {r['mean']:6.1%}   {r.worst:6.1%}   {r.alpha:+.1%}   {r.beta:.2f}")
    T.to_csv(a.out, index=False); print(f"-> {a.out}")
