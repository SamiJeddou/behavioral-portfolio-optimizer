#!/usr/bin/env python3
"""Hierarchical HYBRID optimiser -- three-way rolling backtest / value decomposition.

On identical windows, floors and data it compares THREE constructions:
  (1) PURE SCALABLE   -- the scalable Monte-Carlo + CVaR LP on the full union universe (flat).
  (2) HYBRID          -- exact grid ACROSS asset-class buckets x scalable CVaR fill WITHIN each.
  (3) GRID + EQ-WEIGHT-- exact grid ACROSS the same buckets x naive equal-weight fill within each
                         (the "4-proxy grid": same structure, no scalable selection).

Decomposition:
  (2) - (1)  = value of the hierarchical STRUCTURE (exact VaR across asset classes vs flat CVaR)
  (2) - (3)  = value of the scalable FILL inside each bucket (vs equal weight)
  (3) - (1)  = value of structure alone, with a naive fill

Two engines used for their strengths: the scalable LP scales to many names inside a bucket; the
exact grid enforces the true non-convex goal-based VaR floor P(r<H)<=alpha across the few buckets.

    python hybrid_backtest.py --benchmark SPY --start 2005-01-01 --wmax 0.20 \
      --buckets "EQ:AAPL,MSFT,JPM,XOM,JNJ,PG,KO,WMT;RATES:TLT,IEF,LQD;GOLD:GLD;ALT:VNQ,XLE"
"""
import argparse, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from core.markets import stats_from_prices, corr_to_cov
from core.grid import run_opt
from core.backtest import _bt_portfolio_path, _bt_metrics, _capm_alpha_beta
from core.optimise import optimise_scenario
from core.types import AssetUniverse, Constraint

_RES = {"fast": (21, 15), "standard": (35, 50), "high": (51, 99)}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def parse_buckets(spec):
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        name, items = part.split(":")
        out[name.strip()] = [t.strip() for t in items.split(",") if t.strip()]
    return out


def get_prices(all_tickers, bench, start, end, csv=None):
    cols = list(dict.fromkeys(list(all_tickers) + [bench]))
    if csv:
        df = pd.read_csv(csv, index_col=0, parse_dates=True).sort_index()
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise SystemExit(f"CSV missing columns: {miss}")
        return df[cols].dropna(how="all")
    import yfinance as yf
    raw = yf.download(cols, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return px[cols].dropna(how="all")


def scalable_weights(P, names_in, H, alpha, L, wmax, scenarios):
    means, sigs, corr, names, _ = stats_from_prices(P[names_in], "Daily")
    if len(names) < len(names_in):
        return None, None
    uni = AssetUniverse(names=names, means=np.array(means), sigmas=np.asarray(sigs, float),
                        corr=np.array(corr))
    r = optimise_scenario(uni, Constraint(kind="es_rigorous", H=H, alpha=alpha, L=L),
                          scenarios=scenarios, w_max=wmax, seed=0)
    if (not r.feasible) or np.asarray(r.weights).size == 0:
        return None, names
    return np.asarray(r.weights, float), names


def bucket_blocks(Pc, Pe, buckets, fill, H, alpha, L, wmax, scenarios, factor):
    """For each bucket return (annualised mean, annualised vol, conditioning daily-ret series,
    eval growth-of-1 path) using the chosen within-bucket fill ('scalable' or 'equal')."""
    bmu, bsig, bpc, bpe = [], [], [], []
    for items in buckets.values():
        if len(items) == 1:
            nb = list(items); wb = np.array([1.0])
        elif fill == "scalable":
            wb, nb = scalable_weights(Pc, items, H, alpha, max(L, -0.60), wmax, scenarios)
            if wb is None:
                nb = list(items); wb = np.ones(len(nb)) / len(nb)
        else:  # equal weight
            nb = list(items); wb = np.ones(len(nb)) / len(nb)
        rc = Pc[nb].pct_change().dropna().values @ wb
        bmu.append(rc.mean() * factor); bsig.append(rc.std() * np.sqrt(factor)); bpc.append(rc)
        spe = Pe[nb].values / Pe[nb].values[0]
        bpe.append(_bt_portfolio_path(spe, wb))
    return bmu, bsig, bpc, bpe


def grid_over_buckets(bmu, bsig, bpc, bpe, H, alpha, m, mp):
    """Exact grid across the bucket sub-portfolios; return the combined growth-of-1 path."""
    L0 = min(len(r) for r in bpc)
    B = pd.DataFrame(np.column_stack([r[:L0] for r in bpc]))
    bsig = np.asarray(bsig, float)
    nd, _ = run_opt(np.asarray(bmu, float), bsig, corr_to_cov(bsig, B.corr().values),
                    None, H, alpha, m, mp, "var")
    a = np.asarray(nd["weights"], float)
    Lp = min(len(p) for p in bpe)
    return np.sum([a[i] * bpe[i][:Lp] for i in range(len(bpe))], axis=0)


def run(df, buckets, bench, H, alpha, L, wmax, scenarios, con_m, ev_m, step_m, rf, resolution):
    m, mp = _RES[resolution]; factor = 252
    all_t = [t for b in buckets.values() for t in b]
    idx = df.index; rows = []; s = idx[0]; skip = 0
    while True:
        c0 = s; c1 = c0 + pd.DateOffset(months=con_m); e1 = c1 + pd.DateOffset(months=ev_m)
        if e1 > idx[-1]:
            break
        nxt = s + pd.DateOffset(months=step_m)
        Pc = df.loc[c0:c1].dropna(); Pe = df.loc[c1:e1].dropna()
        if len(Pc) < 60 or len(Pe) < 20 or any(t not in Pe.columns for t in all_t):
            s = nxt; continue
        if any(Pe[t].isna().any() or Pc[t].isna().any() for t in all_t):
            s = nxt; continue
        T = (Pe.index[-1] - Pe.index[0]).days / 365.25
        try:
            # (1) flat scalable on the full union
            w_scn, names_all = scalable_weights(Pc, all_t, H, alpha, L, wmax, scenarios)
            if w_scn is None:
                skip += 1; s = nxt; continue
            pv_s = _bt_portfolio_path(Pe[names_all].values / Pe[names_all].values[0], w_scn)
            # (2) hybrid: scalable fill + grid across buckets
            bm, bs, bp, be = bucket_blocks(Pc, Pe, buckets, "scalable", H, alpha, L, wmax, scenarios, factor)
            pv_h = grid_over_buckets(bm, bs, bp, be, H, alpha, m, mp)
            # (3) grid across buckets + equal-weight fill (the 4-proxy grid)
            bm2, bs2, bp2, be2 = bucket_blocks(Pc, Pe, buckets, "equal", H, alpha, L, wmax, scenarios, factor)
            pv_g = grid_over_buckets(bm2, bs2, bp2, be2, H, alpha, m, mp)
        except Exception:
            skip += 1; s = nxt; continue

        bret = Pe[bench].pct_change().values[1:]
        rec = dict(start=str(c1.date()))
        for tag, pv in (("s", pv_s), ("h", pv_h), ("g", pv_g)):
            cum, _, _, br = _bt_metrics(pv, factor, H, T)
            be_, al_, _ = _capm_alpha_beta(pv[1:] / pv[:-1] - 1, bret[:len(pv) - 1], rf / factor, factor)
            rec[f"cum_{tag}"] = cum; rec[f"br_{tag}"] = br; rec[f"alpha_{tag}"] = al_; rec[f"beta_{tag}"] = be_
        rows.append(rec); s = nxt
    print(f"(skipped {skip} infeasible/short windows)")
    return pd.DataFrame(rows)


def report(R, H):
    n = len(R)
    if n == 0:
        print("No valid windows."); return
    from scipy import stats as st
    def block(tag, key):
        bk = int(R[f"br_{key}"].sum()); lo, hi = wilson(bk, n); a = R[f"alpha_{key}"].dropna().values
        tt = st.ttest_1samp(a, 0.0) if len(a) > 2 else None
        print(f"\n[{tag}] beta {R[f'beta_{key}'].mean():.2f}")
        print(f"  breach {bk}/{n} = {bk/n:.1%}  [Wilson 95% CI {lo:.1%}, {hi:.1%}]")
        print(f"  return: mean {R[f'cum_{key}'].mean():.1%} | median {R[f'cum_{key}'].median():.1%} | worst {R[f'cum_{key}'].min():.1%}")
        print(f"  alpha: mean {np.mean(a):+.2%}" + (f", t={tt.statistic:.2f}, p={tt.pvalue:.3f}" if tt else ""))
    print(f"\n=== THREE-WAY: {n} windows, floor {H:.0%} ===")
    block("1. PURE SCALABLE (flat)", "s")
    block("2. HIERARCHICAL HYBRID", "h")
    block("3. GRID + EQUAL-WEIGHT BUCKETS", "g")
    bs, bh, bg = R.br_s.mean(), R.br_h.mean(), R.br_g.mean()
    print("\n--- value decomposition (breach-rate deltas) ---")
    print(f"  STRUCTURE  (hybrid - scalable)      : {bh-bs:+.1%}")
    print(f"  FILL       (hybrid - grid+EW)       : {bh-bg:+.1%}")
    print(f"  STRUCTURE-ONLY (grid+EW - scalable) : {bg-bs:+.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--buckets", required=True, help="'EQ:AAPL,MSFT;RATES:TLT,IEF;GOLD:GLD'")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--start", default="2005-01-01"); p.add_argument("--end", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--H", type=float, default=-0.10); p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--L", type=float, default=-0.20); p.add_argument("--wmax", type=float, default=0.20)
    p.add_argument("--scenarios", type=int, default=8000)
    p.add_argument("--con", type=int, default=12); p.add_argument("--ev", type=int, default=12)
    p.add_argument("--step", type=int, default=6); p.add_argument("--rf", type=float, default=0.03)
    p.add_argument("--resolution", default="standard"); p.add_argument("--out", default=None)
    a = p.parse_args()
    buckets = parse_buckets(a.buckets)
    all_t = [t for b in buckets.values() for t in b]
    df = get_prices(all_t, a.benchmark, a.start, a.end, a.csv)
    print(f"Loaded {len(df)} rows, {df.index[0].date()}..{df.index[-1].date()}; "
          f"{len(buckets)} buckets, {len(all_t)} instruments + {a.benchmark}")
    R = run(df, buckets, a.benchmark, a.H, a.alpha, a.L, a.wmax, a.scenarios,
            a.con, a.ev, a.step, a.rf, a.resolution)
    report(R, a.H)
    out = a.out or "hybrid_threeway.csv"
    R.to_csv(out, index=False); print(f"\nPer-window results -> {out}")
