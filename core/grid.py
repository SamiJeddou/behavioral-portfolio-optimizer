"""Exact grid engine (thesis-validated reference) — UI-free.

Orchestrators that build the discretised state space and call the solvers, the
mean-variance helpers, and the implied-risk-aversion solver. Re-exports the three
solver modules so callers have one import for everything grid-related. No Streamlit.
"""
import numpy as np
from scipy.optimize import minimize, brentq as _brentq
from scipy.stats import norm as _norm
from behavioral_portfolio_optimizer import (
    build_state_space, assign_probabilities, optimize_portfolio,
)
from turbo_optimizer import optimize_portfolio_turbo
from es_rigorous import optimize_portfolio_es_rigorous

__all__ = [
    "run_opt", "build_frontier", "mv_frontier_at_return", "implied_lambda",
    "build_state_space", "assign_probabilities", "optimize_portfolio",
    "optimize_portfolio_turbo", "optimize_portfolio_es_rigorous",
]


def mv_frontier_at_return(means, cov, target_ret, return_weights=False):
    """Minimum-variance long-only portfolio achieving expected return `target_ret`
    (decimal) — i.e. the mean-variance-efficient portfolio at that return level.
    Returns (std_dev_pct, expected_return_pct), or (std_dev_pct, expected_return_pct,
    weights) when return_weights=True, or None. This is Portfolio (1)'s
    Markowitz counterpart: it coincides with Portfolio (1) exactly when Portfolio
    (1) is mean-variance efficient (the MVT/MAT equivalence)."""
    means = np.asarray(means, dtype=float); cov = np.asarray(cov, dtype=float)
    n = len(means)
    target = min(max(float(target_ret), float(means.min())), float(means.max()))
    def var(w): return float(w @ cov @ w)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, t=target: float(w @ means) - t}]
    best = None
    for x0 in [np.ones(n)/n] + [np.eye(n)[i] for i in range(n)]:
        r = minimize(var, x0, method="SLSQP", bounds=[(0, 1)]*n, constraints=cons)
        if r.success and (best is None or r.fun < best.fun):
            best = r
    if best is None:
        return None
    w = best.x
    _std_pct = float(np.sqrt(max(w @ cov @ w, 1e-12)))*100
    _ret_pct = float(w @ means)*100
    if return_weights:
        return _std_pct, _ret_pct, w
    return _std_pct, _ret_pct

def implied_lambda(H, alpha, means, cov_matrix, lam_lo=0.01, lam_hi=500):
    """Find implied risk-aversion lambda such that VaR constraint binds at (H, alpha)."""
    def mv_w(lam):
        from scipy.optimize import minimize as _min
        n = len(means)
        def obj(w): return -(w@means-(lam/2)*(w@cov_matrix@w))
        res = _min(obj, np.ones(n)/n, method="SLSQP",
                   bounds=[(0,1)]*n,
                   constraints=[{"type":"eq","fun":lambda w: w.sum()-1}])
        return res.x
    def f(lam):
        w = mv_w(lam)
        pm = w @ means
        ps = np.sqrt(max(w @ cov_matrix @ w, 1e-12))
        return _norm.cdf((H - pm) / ps) - alpha
    try:
        f_lo = f(lam_lo)
        f_hi = f(lam_hi)
        if f_lo * f_hi > 0:
            for hi in [1000, 5000]:
                try:
                    if f_lo * f(hi) < 0:
                        return _brentq(f, lam_lo, hi)
                except Exception:
                    pass
            return None
        return _brentq(f, lam_lo, lam_hi)
    except Exception:
        return None

def run_opt(means,sigs,cov,der_config,H,alpha,m,mp,
            constraint_type='var',L=None):
    turbo=(mp=='turbo')
    rigorous=(constraint_type=='es_rigorous')
    U,dr=build_state_space(means,sigs,m=(51 if (turbo or rigorous) else m),derivative_config=der_config)
    U=assign_probabilities(U,means,sigs,cov,dr)
    n=U.shape[1]-1
    a=alpha if alpha is not None else 0.05
    if rigorous:
        res=optimize_portfolio_es_rigorous(U,n,CVARH=H,L=L)
    elif turbo:
        res=optimize_portfolio_turbo(U,n,H=H,alpha=a,m_prime=99,
                                     constraint_type=constraint_type,L=L)
    else:
        res=optimize_portfolio(U,n,H=H,alpha=a,
                               m_prime=mp,constraint_type=constraint_type,L=L)
    return res,n

def build_frontier(means,sigs,cov,der_config,alpha,m,mp,
                   constraint_type='var',L=None):
    H_vals=[-0.02,-0.05,-0.08,-0.10,-0.12,-0.15,-0.18,-0.20,-0.25,-0.30,-0.35,-0.40]
    pts=[]
    for H in H_vals:
        try:
            r,_=run_opt(means,sigs,cov,der_config,H,alpha,m,mp,
                        constraint_type=constraint_type,L=L)
            pts.append((r["std_dev"]*100, r["expected_return"]*100, f"H={H:.0%}"))
        except: pass
    # Sort by std dev ascending so line draws left to right
    pts.sort(key=lambda p: p[0])
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]
    lbls= [p[2] for p in pts]
    return xs,ys,lbls
