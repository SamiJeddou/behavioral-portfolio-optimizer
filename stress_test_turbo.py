"""
Stress test: Turbo (coarse-to-fine) vs the exhaustive grid solver.

Purpose
-------
Turbo replaces the exhaustive weight-grid search with a coarse-to-fine search.
That is a heuristic: it is exact only if a coarse seed lands in the basin of the
global optimum. This script quantifies how often, and by how much, Turbo's
result departs from the exhaustive grid optimum on the SAME state space, so the
claim becomes "validated to within X pp over N random instances" rather than
"matched the 18 thesis tables".

Method
------
For each random instance we build one state space U (resolution `m`) and run:
  * baseline : optimize_portfolio(..., method='auto')      # exhaustive grid
  * turbo    : optimize_portfolio_turbo(...)                # coarse-to-fine
both at the same fine weight resolution `m_prime`, then compare expected return,
weights, and shortfall feasibility. Results are grouped by the weight-simplex
dimension (n_total - 1) and by derivative presence, because reliability of the
coarse grid scales with the weight-search dimension, while cost scales with the
number of PRIMARY securities (the state space is m**n_primaries).

Scope tested = Turbo's actual scope: VaR, n_total <= 4.

Run the full-resolution battery offline:
    python stress_test_turbo.py --full      # m=51, m_prime=99, 200 samples
Quick demo (default) keeps each optimisation small so it finishes in minutes.
"""
import argparse, time
import numpy as np

from behavioral_portfolio_optimizer import (
    build_state_space, assign_probabilities, optimize_portfolio)
from turbo_optimizer import optimize_portfolio_turbo

TOL_PP = 0.10          # disagreement threshold in percentage points of E[r]
FEAS_TOL = 1e-6


def random_corr(n, rng, lo=-0.25, hi=0.6):
    """Random symmetric correlation matrix, resampled until PSD."""
    for _ in range(200):
        C = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                C[i, j] = C[j, i] = rng.uniform(lo, hi)
        if np.all(np.linalg.eigvalsh(C) > 1e-8):
            return C
    return np.eye(n)


def random_instance(n_prime, has_deriv, rng):
    """A feasible-ish instance. Security 0 is anchored low-risk to keep many
    instances feasible; the rest are random."""
    means = np.empty(n_prime); sigs = np.empty(n_prime)
    means[0], sigs[0] = rng.uniform(0.03, 0.06), rng.uniform(0.04, 0.08)
    for i in range(1, n_prime):
        means[i] = rng.uniform(0.06, 0.30)
        sigs[i]  = rng.uniform(0.12, 0.55)
    corr = random_corr(n_prime, rng)
    cov = np.outer(sigs, sigs) * corr
    H = rng.uniform(-0.15, -0.08)
    alpha = rng.uniform(0.08, 0.25)
    der_cfg = None
    if has_deriv:
        u = rng.integers(0, n_prime)
        kind = rng.choice(['put', 'call'])
        strike = float(rng.uniform(0.8, 1.4))
        der_cfg = {'type': kind, 'underlying_index': int(u),
                   'vol': float(sigs[u]), 'strike': strike,
                   'r': 0.03, 'T': 1.0, 'S0': 1.0}
    return means, sigs, cov, H, alpha, der_cfg


def feasible_var(res, alpha):
    return res['shortfall_stat'] <= alpha + FEAS_TOL


def run_config(label, n_prime, has_deriv, m, m_prime, n_samples, rng):
    gaps, l1s, base_t, turbo_t = [], [], [], []
    n_feasible = n_turbo_infeasible = n_turbo_error = n_worse = 0
    for _ in range(n_samples):
        means, sigs, cov, H, alpha, der_cfg = random_instance(n_prime, has_deriv, rng)
        U, dr = build_state_space(means, sigs, m=m, derivative_config=der_cfg)
        U = assign_probabilities(U, means, sigs, cov, dr)
        n = U.shape[1] - 1
        try:
            t0 = time.perf_counter()
            base = optimize_portfolio(U, n, constraint_type='var', H=H,
                                      alpha=alpha, m_prime=m_prime, method='auto')
            base_t.append(time.perf_counter() - t0)
        except ValueError:
            continue                      # infeasible instance — skip both
        if not feasible_var(base, alpha):
            continue
        n_feasible += 1
        try:
            t0 = time.perf_counter()
            turbo = optimize_portfolio_turbo(U, n, constraint_type='var', H=H,
                                             alpha=alpha, m_prime=m_prime)
            turbo_t.append(time.perf_counter() - t0)
        except Exception:
            n_turbo_error += 1
            continue
        if not feasible_var(turbo, alpha):
            n_turbo_infeasible += 1
        gap = (turbo['expected_return'] - base['expected_return']) * 100  # pp
        gaps.append(gap)
        l1s.append(float(np.sum(np.abs(turbo['weights'] - base['weights']))))
        if gap < -TOL_PP:               # turbo found a materially worse optimum
            n_worse += 1
    g = np.array(gaps) if gaps else np.array([0.0])
    a = np.abs(g)
    return {
        'label': label, 'dim': n_prime + (1 if has_deriv else 0) - 1,
        'n_feasible': n_feasible, 'n_compared': len(gaps),
        'mean_abs': a.mean(), 'p95_abs': np.percentile(a, 95), 'max_abs': a.max(),
        'worst_signed': g.min(), 'max_l1': max(l1s) if l1s else 0.0,
        'n_worse': n_worse, 'n_turbo_infeasible': n_turbo_infeasible,
        'n_turbo_error': n_turbo_error,
        'disagree_rate': float((a > TOL_PP).mean()),
        'base_t': np.mean(base_t) if base_t else 0.0,
        'turbo_t': np.mean(turbo_t) if turbo_t else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='m=51, m_prime=99, 200 samples')
    ap.add_argument('--samples', type=int, default=None)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # (label, n_prime, has_deriv, m, m_prime)  — resolutions chosen so the
    # exhaustive baseline stays tractable for the demo. --full overrides to 51/99.
    configs = [
        ("2 sec, no deriv          (dim 1)", 2, False, 51, 51),
        ("2 sec + derivative       (dim 2)", 2, True,  41, 41),
        ("3 sec, no deriv          (dim 2)", 3, False, 41, 41),
        ("3 sec + derivative       (dim 3)", 3, True,  31, 31),
        ("4 sec, no deriv          (dim 3)", 4, False, 21, 25),
    ]
    n_samples = args.samples or (200 if args.full else 25)

    print(f"\nTurbo vs exhaustive grid  |  samples/config={n_samples}  "
          f"|  disagreement threshold={TOL_PP}pp  |  full={args.full}\n")
    print(f"{'config':<34}{'cmp':>4}{'meanAbs':>9}{'p95':>7}{'maxAbs':>8}"
          f"{'worst':>8}{'maxL1':>7}{'wrs':>4}{'inf':>4}{'dis%':>6}{'speedup':>9}")
    print("-" * 112)
    for label, npri, hd, m, mp in configs:
        if args.full:
            m, mp = 51, 99
        r = run_config(label, npri, hd, m, mp, n_samples, rng)
        spd = (r['base_t'] / r['turbo_t']) if r['turbo_t'] > 0 else 0.0
        print(f"{r['label']:<34}{r['n_compared']:>4}{r['mean_abs']:>9.3f}"
              f"{r['p95_abs']:>7.3f}{r['max_abs']:>8.3f}{r['worst_signed']:>8.3f}"
              f"{r['max_l1']:>7.3f}{r['n_worse']:>4}{r['n_turbo_infeasible']:>4}"
              f"{100*r['disagree_rate']:>5.0f}%{spd:>8.1f}x")
    print("\nColumns: cmp=instances compared, meanAbs/p95/maxAbs=|E[r] gap| pp, "
          "worst=most negative signed gap pp (Turbo underperformance),\n"
          "maxL1=max weight L1 distance, wrs=#cases Turbo worse by >threshold, "
          "inf=#Turbo ES/VaR-infeasible, dis%=disagreement rate.\n")


if __name__ == '__main__':
    main()
