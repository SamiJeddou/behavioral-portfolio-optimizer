# © 2026 Sami Jeddou. All rights reserved.
# Published publicly for demonstration and evaluation only — no license is granted.
# Copying, modification, redistribution, or reuse (in whole or in part) without the
# author's prior written permission is prohibited.
"""Scalable Monte-Carlo + CVaR engine — UI-free.

Copula scenario generation (Gaussian / Student-t), derivative scenario stacking, the
Rockafellar-Uryasev CVaR linear program (HiGHS), the return/tail-risk frontier, and
closed-form ES cross-checks. No Streamlit.
"""
import numpy as np
from scipy.stats import norm as _norm, t as _student_t, chi2 as _chi2
from scipy.optimize import linprog as _linprog
from scipy.sparse import coo_matrix as _coo
from core.pricing import mc_der_returns

__all__ = [
    "mc_generate_scenarios", "mc_build_matrix", "mc_max_return_cvar", "mc_min_cvar",
    "mc_frontier", "mc_realised_es", "mc_analytical_es", "mc_gmv_weights",
    "_mc_psd_cholesky", "_mc_cvar_rows",
]


def _mc_psd_cholesky(corr):
    """Cholesky factor of a correlation matrix, repaired to nearest-PSD if needed."""
    corr = np.asarray(corr, float)
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(corr)
        w = np.clip(w, 1e-8, None)
        c2 = (V * w) @ V.T
        d = np.sqrt(np.diag(c2))
        c2 = c2 / np.outer(d, d)
        return np.linalg.cholesky(c2)

def mc_generate_scenarios(means, sigs, corr, S=10000, copula="gaussian", dof=5, seed=0):
    """Draw S joint security-return scenarios (S x N). 'gaussian' => multivariate
    Normal (the thesis assumption); 't' => Student-t copula with Normal marginals
    (tail dependence: assets fall together)."""
    rng = np.random.default_rng(seed)
    means = np.asarray(means, float); sigs = np.asarray(sigs, float)
    N = len(means); L = _mc_psd_cholesky(corr)
    if copula == "gaussian":
        Z = rng.standard_normal((S, N)) @ L.T
    else:
        Y = rng.standard_normal((S, N)) @ L.T
        g = _chi2.rvs(dof, size=S, random_state=rng) / dof
        Tv = Y / np.sqrt(g)[:, None]
        U = _student_t.cdf(Tv, dof)
        Z = _norm.ppf(np.clip(U, 1e-12, 1 - 1e-12))
    return means[None, :] + sigs[None, :] * Z

def mc_build_matrix(R_sec, der_specs, sigs, names, r=0.03, T=1.0, horizon=1.0):
    """Stack S x N security returns with one return column per derivative spec.
    der_specs: list of dicts {der_type, params, underlying_idx, [T, vol_override, r]}.
    Per-spec maturity, implied-vol override and rate fall back to the underlying's own
    sigma and the defaults. Returns (R_full, labels, errors)."""
    cols = [R_sec]; labels = list(names); errors = []
    for d in der_specs:
        ui = d["underlying_idx"]
        vol = d["vol_override"] if d.get("vol_override") is not None else sigs[ui]
        Td = float(d.get("T", T)); rd = float(d.get("r", r))
        dr = mc_der_returns(d["der_type"], d["params"], R_sec[:, ui], vol, r=rd, T=Td, horizon=horizon)
        if dr is None:
            errors.append(d.get("label", d["der_type"])); continue
        cols.append(dr[:, None]); labels.append(d.get("label", d["der_type"]))
    return np.hstack(cols), labels, errors

def mc_realised_es(port, alpha):
    p = np.sort(np.asarray(port, float))
    k = max(1, int(np.floor(alpha * len(p))))
    return float(p[:k].mean())

def mc_analytical_es(mu_p, sig_p, alpha):
    """Closed-form ES (tail-average return) for a Normal portfolio return."""
    z = _norm.ppf(alpha)
    return float(mu_p - sig_p * _norm.pdf(z) / alpha)

def mc_gmv_weights(cov):
    """Analytical global-minimum-variance weights (shorts allowed)."""
    cov = np.asarray(cov, float); n = cov.shape[0]
    inv = np.linalg.pinv(cov); one = np.ones(n)
    w = inv @ one
    return w / (one @ w)

def _mc_cvar_rows(R, alpha):
    """Common sparse blocks for the Rockafellar-Uryasev CVaR constraints.
    Scenario rows (S of them): -R[s].w - zeta - z_s <= 0. Returns the COO pieces."""
    S, n = R.shape
    sidx = np.arange(S)
    rows = list(np.repeat(sidx, n)); cols = list(np.tile(np.arange(n), S)); vals = list((-R).ravel())
    rows += list(sidx); cols += [n] * S; vals += [-1.0] * S                 # -zeta
    rows += list(sidx); cols += list(range(n + 1, n + 1 + S)); vals += [-1.0] * S  # -z_s
    return rows, cols, vals, S, n

def mc_max_return_cvar(R, alpha, L, w_max=None, long_only=True):
    """max E[r] s.t. CVaR_alpha(-r) <= -L (tail-average return >= L), sum w = 1,
    0 <= w <= w_max. Linear program in (w, zeta, z). Returns (w, E[r], realised ES, res)."""
    R = np.asarray(R, float); S, n = R.shape; nv = n + 1 + S
    mu = R.mean(axis=0)
    c = np.concatenate([-mu, [0.0], np.zeros(S)])
    rows, cols, vals, S, n = _mc_cvar_rows(R, alpha)
    rows = [r_ + 1 for r_ in rows]                       # shift scenario rows to 1..S
    # CVaR row (row 0): zeta + 1/(alpha S) sum z <= -L
    rows = [0] + [0] * S + rows
    cols = [n] + list(range(n + 1, n + 1 + S)) + cols
    vals = [1.0] + [1.0 / (alpha * S)] * S + vals
    A_ub = _coo((vals, (rows, cols)), shape=(S + 1, nv)).tocsr()
    b_ub = np.concatenate([[-L], np.zeros(S)])
    A_eq = _coo((np.ones(n), (np.zeros(n), np.arange(n))), shape=(1, nv)).tocsr()
    b_eq = np.array([1.0])
    wb = (0.0, w_max if w_max is not None else 1.0) if long_only else (None, None)
    bounds = [wb] * n + [(None, None)] + [(0.0, None)] * S
    res = _linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return None, None, None, res
    w = res.x[:n]; port = R @ w
    return w, float(port.mean()), mc_realised_es(port, alpha), res

def mc_min_cvar(R, alpha, long_only=False):
    """min CVaR_alpha(-r) s.t. sum w = 1 (shorts allowed by default). Used for the
    validation cross-check against analytical global-minimum-variance."""
    R = np.asarray(R, float); S, n = R.shape; nv = n + 1 + S
    c = np.concatenate([np.zeros(n), [1.0], np.full(S, 1.0 / (alpha * S))])
    rows, cols, vals, S, n = _mc_cvar_rows(R, alpha)
    A_ub = _coo((vals, (rows, cols)), shape=(S, nv)).tocsr()
    b_ub = np.zeros(S)
    A_eq = _coo((np.ones(n), (np.zeros(n), np.arange(n))), shape=(1, nv)).tocsr()
    b_eq = np.array([1.0])
    wb = (0.0, 1.0) if long_only else (None, None)
    bounds = [wb] * n + [(None, None)] + [(0.0, None)] * S
    res = _linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return (res.x[:n] if res.success else None), res

def mc_frontier(R, alpha, floors, w_max=None):
    out = []
    for L in floors:
        w, er, es, res = mc_max_return_cvar(R, alpha, L, w_max=w_max)
        out.append({"L": L, "ok": bool(res.success), "er": er, "es": es})
    return out
