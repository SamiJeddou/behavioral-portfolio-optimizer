"""
Rigorous Expected-Shortfall mode  (a beyond-thesis enhancement).

The thesis / R original enforces the ES constraint only when selecting the
grid seed, then runs a COBYLA refinement whose objective is the *VaR* penalty
(alpha - P(r<H))^2 — so the refined ES can drift below the stated L limit.

This module instead maximises expected return subject to
        ES = E[ r | r < CVARH ] >= L
with an *ES-aware* COBYLA penalty. The grid stage reuses Turbo's coarse-to-fine
search + state pruning for speed. Same return-dict shape as optimize_portfolio.

It is intended to be offered ALONGSIDE the thesis-faithful ES method, not to
replace it (the faithful method is kept for exact thesis reproduction).
"""
import numpy as np
from scipy.optimize import minimize
from turbo_optimizer import prune_states, _best_feasible


def _es(w, R, probs, CVARH):
    pr = R @ w
    tail = pr < CVARH
    return (float((pr[tail] @ probs[tail]) / probs[tail].sum())
            if tail.sum() > 0 else float('inf'))   # no tail -> ES vacuously satisfied


def optimize_portfolio_es_rigorous(U, n_securities, CVARH=-0.10, L=-0.15,
                                   penalty=1e12, m_prime=99,
                                   mp_coarse=25, win_steps=3, prune_keep=1.0 - 1e-4):
    """Max expected return s.t. E[r | r < CVARH] >= L. CVARH is the ES threshold
    (the thesis uses -0.10); L is the ES limit."""
    Uk = prune_states(U, prune_keep)
    probs = Uk[:, -1]
    R = Uk[:, :n_securities]
    nf = n_securities - 1

    # Stage 1 — coarse GLOBAL ES-feasible grid (locate the optimum's neighbourhood)
    sc = 1.0 / mp_coarse
    gc = np.arange(sc, 1.0, sc)
    w_c = _best_feasible(R, probs, CVARH, None, 'es', L, [gc] * nf)
    if w_c is None:
        raise ValueError("No ES-eligible portfolios found. Relax L or CVARH.")

    # Stage 2 — fine LOCAL ES-feasible grid around the coarse optimum
    sf = 1.0 / m_prime
    win = win_steps * sc
    grids = [np.arange(max(sf, w_c[i] - win), min(1.0, w_c[i] + win) + 1e-12, sf)
             for i in range(nf)]
    w_f = _best_feasible(R, probs, CVARH, None, 'es', L, grids)
    seed = w_f if w_f is not None else w_c

    # Stage 3 — ES-AWARE COBYLA (this is the correction vs the thesis refinement)
    def objective(wei):
        w = np.append(wei, 1.0 - wei.sum())
        pr = R @ w
        tail = pr < CVARH
        es = float((pr[tail] @ probs[tail]) / probs[tail].sum()) if tail.sum() > 0 else 1e9
        ret = float(pr @ probs)
        return -ret + penalty * max(0.0, L - es) ** 2

    res = minimize(objective, seed[:-1], method='COBYLA',
                   bounds=[(0.0, 1.0)] * nf,
                   options={'rhobeg': 0.01, 'maxiter': 10000, 'catol': 1e-8})
    w = np.append(res.x, 1.0 - res.x.sum())
    w = np.clip(w, 0, 1)
    w /= w.sum()

    # Feasibility guard: never return an ES-infeasible point
    if _es(w, R, probs, CVARH) < L:
        w = seed.copy()

    pr = R @ w
    mean_r = float(pr @ probs)
    variance = float(((pr - mean_r) ** 2) @ probs)
    std_dev = np.sqrt(max(variance, 0))
    skewness = (float(((pr - mean_r) ** 3) @ probs) / std_dev ** 3 if std_dev > 0 else 0.0)
    kurtosis = (float(((pr - mean_r) ** 4) @ probs) / std_dev ** 4 - 3 if std_dev > 0 else 0.0)
    tail = pr < CVARH
    es_stat = float((pr[tail] @ probs[tail]) / probs[tail].sum()) if tail.sum() > 0 else 0.0

    return {
        'weights': w,
        'expected_return': mean_r,
        'std_dev': std_dev,
        'skewness': skewness,
        'excess_kurtosis': kurtosis,
        'shortfall_stat': es_stat,   # this is ES (E[r|r<CVARH])
        'eligible_count': -1,
        'method_used': 'es_rigorous',
    }
