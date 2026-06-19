#!/usr/bin/env python3
"""Naive 1/N baselines (DeMiguel-Garlappi-Uppal) on the SAME rolling windows, to test whether the
goal-based hierarchy actually beats naive diversification. Estimate-free, so near-instant.

  EW-ALL        : plain equal weight over every instrument (pure 1/N; name-count weighted).
  BUCKET-PARITY : equal weight ACROSS asset-class buckets, equal within (asset-class 1/N, NO grid).

Compare these to grid+equal-weight from hybrid_backtest.py (which adds the exact goal-based VaR grid
ACROSS buckets). If EW-ALL or BUCKET-PARITY matches grid+EW, the grid layer adds nothing.

    python naive_benchmarks.py --benchmark SPY --start 2005-01-01 \
      --buckets "EQ:AAPL,MSFT,...;RATES:TLT,IEF,LQD;GOLD:GLD;ALT:VNQ,XLE"
"""
import argparse
import numpy as np, pandas as pd
from rolling_backtest import get_prices, wilson
from core.backtest import _bt_portfolio_path, _bt_metrics, _capm_alpha_beta


def parse_buckets(spec):
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        name, items = part.split(":")
        out[name.strip()] = [t.strip() for t in items.split(",") if t.strip()]
    return out


def run(df, buckets, bench, H, con_m, ev_m, step_m, rf):
    factor = 252
    all_t = [t for b in buckets.values() for t in b]
    idx = df.index; rows = []; s = idx[0]
    while True:
        c1 = s + pd.DateOffset(months=con_m); e1 = c1 + pd.DateOffset(months=ev_m)
        if e1 > idx[-1]:
            break
        nxt = s + pd.DateOffset(months=step_m)
        Pe = df.loc[c1:e1].dropna()
        if len(Pe) < 20 or any(t not in Pe.columns for t in all_t) or any(Pe[t].isna().any() for t in all_t):
            s = nxt; continue
        T = (Pe.index[-1] - Pe.index[0]).days / 365.25
        bret = Pe[bench].pct_change().values[1:]
        # EW-ALL: plain 1/N over all instruments
        g_all = Pe[all_t].values / Pe[all_t].values[0]
        pv_e = _bt_portfolio_path(g_all, np.ones(len(all_t)) / len(all_t))
        # BUCKET-PARITY: 1/K across buckets, equal within
        K = len(buckets); bp = np.zeros(len(g_all))
        for items in buckets.values():
            gb = Pe[items].values / Pe[items].values[0]
            bp = bp + (1.0 / K) * _bt_portfolio_path(gb, np.ones(len(items)) / len(items))
        rec = dict(start=str(c1.date()))
        for tag, pv in (("e", pv_e), ("b", bp)):
            cum, _, _, br = _bt_metrics(pv, factor, H, T)
            be, al, _ = _capm_alpha_beta(pv[1:] / pv[:-1] - 1, bret[:len(pv) - 1], rf / factor, factor)
            rec[f"cum_{tag}"] = cum; rec[f"br_{tag}"] = br; rec[f"al_{tag}"] = al; rec[f"be_{tag}"] = be
        rows.append(rec); s = nxt
    return pd.DataFrame(rows)


def report(R, H):
    from scipy import stats as st
    n = len(R)
    if n == 0:
        print("No windows."); return
    print(f"\n=== Naive 1/N baselines: {n} windows, floor {H:.0%} ===")
    for tag, key in (("EW-ALL  (pure 1/N)", "e"), ("BUCKET-PARITY (1/K, no grid)", "b")):
        bk = int(R[f"br_{key}"].sum()); lo, hi = wilson(bk, n); a = R[f"al_{key}"].dropna().values
        tt = st.ttest_1samp(a, 0.0)
        print(f"[{tag}] beta {R[f'be_{key}'].mean():.2f} | breach {bk}/{n}={bk/n:.1%} [{lo:.1%},{hi:.1%}] | "
              f"mean {R[f'cum_{key}'].mean():.1%} | worst {R[f'cum_{key}'].min():.1%} | alpha {a.mean():+.1%} p={tt.pvalue:.3f}")
    print("\nCompare to grid+EW (hybrid_threeway.csv: method 3). If these match it, the grid adds nothing.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--buckets", required=True)
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--start", default="2005-01-01"); p.add_argument("--end", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--H", type=float, default=-0.10)
    p.add_argument("--con", type=int, default=12); p.add_argument("--ev", type=int, default=12)
    p.add_argument("--step", type=int, default=6); p.add_argument("--rf", type=float, default=0.03)
    p.add_argument("--out", default="naive_benchmarks.csv")
    a = p.parse_args()
    buckets = parse_buckets(a.buckets)
    all_t = [t for b in buckets.values() for t in b]
    df = get_prices(all_t, a.benchmark, a.start, a.end, a.csv)
    print(f"Loaded {len(df)} rows, {df.index[0].date()}..{df.index[-1].date()}; {len(all_t)} instruments + {a.benchmark}")
    R = run(df, buckets, a.benchmark, a.H, a.con, a.ev, a.step, a.rf)
    report(R, a.H)
    R.to_csv(a.out, index=False); print(f"-> {a.out}")
