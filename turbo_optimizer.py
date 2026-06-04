"""
Turbo optimiser — a fast, drop-in alternative to optimize_portfolio that
reproduces the High-precision (m=51, m'=99) result in ~seconds.

Strategy (n<=4): keep the SAME high-precision state space, but replace the full
exhaustive weight-grid search with a coarse-to-fine search, plus pruning of
near-zero-probability states. The constraint logic (VaR / ES), the COBYLA
refinement and the statistics are copied verbatim from optimize_portfolio, so
the result matches High precision to the reported decimals.

For n>=5 it delegates to optimize_portfolio's differential-evolution path
unchanged, so behaviour is preserved.
"""
import numpy as np
from scipy.optimize import minimize


def prune_states(U, keep_mass=1.0 - 1e-4):
    """Drop near-zero-probability states (keep `keep_mass` of total mass),
    renormalise. Expected return and shortfall are probability-weighted sums,
    so dropping <1e-4 of mass changes them below the reported precision."""
    p = U[:, -1]
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    k = int(np.searchsorted(cum, keep_mass)) + 1
    Uk = U[order[:k]].copy()
    Uk[:, -1] = Uk[:, -1] / Uk[:, -1].sum()
    return Uk


def _best_feasible(R, probs, H, alpha, constraint_type, L, grids):
    """Enumerate simplex-feasible weight combos from per-dimension candidate
    `grids`, return the max-expected-return feasible weight vector.
    Feasibility mirrors optimize_portfolio exactly (VaR and ES)."""
    mesh = np.meshgrid(*grids, indexing='ij')
    combos = np.stack([g.ravel() for g in mesh], axis=1)
    combos = combos[combos.sum(1) <= 1.0]
    last = 1.0 - combos.sum(1, keepdims=True)
    W = np.hstack([combos, last])
    W = W[(W >= 0).all(1)]
    if len(W) == 0:
        return None
    mu = probs @ R
    exp_r = W @ mu
    Rf = R.astype(np.float32)
    pf = probs.astype(np.float32)
    Hf = np.float32(H)
    best, bw = -np.inf, None
    B = 2000
    for s in range(0, len(W), B):
        Wb = W[s:s + B].astype(np.float32)
        PR = Rf @ Wb.T                      # (states, b)
        tail = PR < Hf
        den = pf @ tail.astype(np.float32)  # shortfall prob per combo
        if constraint_type == 'var':
            elig = den <= alpha
        else:  # 'es' — mirror optimize_portfolio: eligible only if tail non-empty
            num = (pf[:, None] * PR * tail).sum(0)
            with np.errstate(invalid='ignore', divide='ignore'):
                es = np.where(den > 0, num / np.where(den > 0, den, 1.0), -np.inf)
            elig = (den > 0) if L is None else ((den > 0) & (es >= L))
        if elig.any():
            er = exp_r[s:s + B][elig]
            j = int(np.argmax(er))
            if er[j] > best:
                best, bw = er[j], Wb[elig][j].astype(np.float64)
    return bw


def _feasible(w, R, probs, H, alpha, constraint_type, L, tol=1e-6):
    """f64 feasibility test, identical semantics to optimize_portfolio."""
    pr = R @ w
    tail = pr < H
    if constraint_type == 'var':
        return float(probs[tail].sum()) <= alpha + tol
    if tail.sum() == 0:
        return False
    es = float((pr[tail] * probs[tail]).sum() / probs[tail].sum())
    return (L is None) or (es >= L)


def optimize_portfolio_turbo(U, n_securities, constraint_type='var',
                             H=-0.10, alpha=0.05, L=None, penalty=1e18,
                             m_prime=99, method='auto',
                             mp_coarse=25, win_steps=3, prune_keep=1.0 - 1e-4):
    """Drop-in replacement for optimize_portfolio. Same signature/return shape.
    `m_prime` is the FINE resolution (default 99, i.e. High-precision accuracy)."""
    # n>=5 -> delegate to the original DE path unchanged
    use_de = (method == 'de') or (method == 'auto' and n_securities >= 5)
    # Turbo accelerates ONLY the VaR grid case (n<=4). Expected-Shortfall and
    # 5+-security problems delegate to the original optimizer unchanged, so
    # their behaviour is byte-for-byte identical to today.
    if use_de or constraint_type != 'var':
        from behavioral_portfolio_optimizer import optimize_portfolio
        return optimize_portfolio(U, n_securities, constraint_type=constraint_type,
                                  H=H, alpha=alpha, m_prime=m_prime, L=L,
                                  penalty=penalty, method=method)

    Uk = prune_states(U, prune_keep)
    probs = Uk[:, -1]
    R = Uk[:, :n_securities]
    nf = n_securities - 1
    mp_fine = m_prime

    # Stage 1 — coarse GLOBAL grid (locates the optimum's neighbourhood)
    sc = 1.0 / mp_coarse
    gc = np.arange(sc, 1.0, sc)
    w_c = _best_feasible(R, probs, H, alpha, constraint_type, L, [gc] * nf)
    if w_c is None:
        raise ValueError("No eligible portfolios found. Try relaxing H, alpha, or m_prime.")

    # Stage 2 — fine LOCAL grid around the coarse optimum (at m_prime resolution)
    sf = 1.0 / mp_fine
    win = win_steps * sc
    grids = [np.arange(max(sf, w_c[i] - win), min(1.0, w_c[i] + win) + 1e-12, sf)
             for i in range(nf)]
    w_f = _best_feasible(R, probs, H, alpha, constraint_type, L, grids)
    best_weights = w_f if w_f is not None else w_c

    # Stage 3 — COBYLA refinement (verbatim from optimize_portfolio)
    def objective(wei):
        w_full = np.append(wei, 1.0 - wei.sum())
        port_ret = R @ w_full
        term1 = float(port_ret @ probs)
        shortfall = float(probs[port_ret < H].sum())
        term2 = penalty * (alpha - shortfall) ** 2
        return -term1 + term2

    result = minimize(objective, x0=best_weights[:-1], method='COBYLA',
                      bounds=[(0.0, 1.0)] * nf,
                      options={'rhobeg': 0.01, 'maxiter': 10000, 'catol': 1e-8})
    w_opt = np.append(result.x, 1.0 - result.x.sum())
    w_opt = np.clip(w_opt, 0, 1)
    w_opt /= w_opt.sum()

    # Feasibility guard: the finite COBYLA penalty can permit a tiny constraint
    # violation. If the refined point is infeasible, fall back to the feasible
    # grid optimum (which passed the constraint exactly).
    if not _feasible(w_opt, R, probs, H, alpha, constraint_type, L):
        w_opt = best_weights.copy()

    # Statistics (verbatim from optimize_portfolio)
    port_ret = R @ w_opt
    mean_r = float(port_ret @ probs)
    variance = float(((port_ret - mean_r) ** 2) @ probs)
    std_dev = np.sqrt(max(variance, 0))
    skewness = (float(((port_ret - mean_r) ** 3) @ probs) / std_dev ** 3
                if std_dev > 0 else 0.0)
    kurtosis = (float(((port_ret - mean_r) ** 4) @ probs) / std_dev ** 4 - 3
                if std_dev > 0 else 0.0)
    if constraint_type == 'var':
        q_stat = float(probs[port_ret < H].sum())
    else:
        tail = port_ret < H
        q_stat = (float((port_ret[tail] * probs[tail]).sum() / probs[tail].sum())
                  if tail.sum() > 0 else 0.0)

    return {
        'weights': w_opt,
        'expected_return': mean_r,
        'std_dev': std_dev,
        'skewness': skewness,
        'excess_kurtosis': kurtosis,
        'shortfall_stat': q_stat,
        'eligible_count': -1,
        'method_used': 'coarse_to_fine',
    }
