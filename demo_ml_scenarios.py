#!/usr/bin/env python3
"""Toy: a LEARNED scenario generator plugged into the engine's CVaR linear program.

Demonstrates the modularity claim --- the LP consumes any scenario set --- with a genuine (if small)
machine-learning generator: a Gaussian-mixture model fitted by Expectation-Maximisation (a standard
unsupervised generative density), implemented in NumPy with no extra dependency. The GMM is fitted to
historical returns, sampled to a scenario matrix, and handed to the SAME `mc_max_return_cvar` solver
the paper uses everywhere else --- the optimiser and pricing are untouched.
"""
import numpy as np
from scipy.stats import multivariate_normal as mvn
from core.scenario import mc_max_return_cvar, mc_generate_scenarios


def gmm_fit(X, K=2, iters=80, reg=1e-5, seed=0):
    """Fit a K-component Gaussian mixture by EM. Returns (weights, means, covariances)."""
    rng = np.random.default_rng(seed); n, d = X.shape
    mu = X[rng.choice(n, K, replace=False)].copy()
    cov = np.array([np.cov(X.T) + reg * np.eye(d) for _ in range(K)])
    pi = np.ones(K) / K
    for _ in range(iters):
        logr = np.column_stack([mvn.logpdf(X, mu[k], cov[k], allow_singular=True) + np.log(pi[k])
                                for k in range(K)])
        logr -= logr.max(1, keepdims=True)
        r = np.exp(logr); r /= r.sum(1, keepdims=True)
        Nk = r.sum(0) + 1e-12
        pi = Nk / n
        mu = (r.T @ X) / Nk[:, None]
        for k in range(K):
            Xc = X - mu[k]
            cov[k] = (r[:, k, None] * Xc).T @ Xc / Nk[k] + reg * np.eye(d)
    return pi, mu, cov


def gmm_sample(pi, mu, cov, S, seed=1):
    rng = np.random.default_rng(seed); K = len(pi)
    comp = rng.choice(K, size=S, p=pi); out = np.empty((S, mu.shape[1]))
    for k in range(K):
        m = comp == k
        if m.any():
            out[m] = rng.multivariate_normal(mu[k], cov[k], int(m.sum()))
    return out


def cvar(port, alpha=0.05):
    v = np.quantile(port, alpha)
    return float(port[port <= v].mean())


if __name__ == "__main__":
    # --- a small two-regime monthly history: calm most of the time, occasional crash month ---
    rng = np.random.default_rng(0); N = 5; T = 240
    mu_calm = np.array([0.012, 0.010, 0.009, 0.007, 0.005])
    cov_calm = np.diag([0.04, 0.05, 0.035, 0.03, 0.02]) ** 1
    X = rng.multivariate_normal(mu_calm, cov_calm * 0.0025, T)
    crash = rng.random(T) < 0.08                       # ~8% of months are crash months
    X[crash] = rng.multivariate_normal(-0.12 * np.ones(N), cov_calm * 0.02, crash.sum())

    alpha, L, wmax = 0.05, -0.20, 0.40

    # --- LEARNED generator: fit GMM by EM, sample scenarios, feed the SAME LP ---
    pi, m, cov = gmm_fit(X, K=2)
    S_gmm = gmm_sample(pi, m, cov, S=10000)
    w_gmm, er_g, es_g, _ = mc_max_return_cvar(S_gmm, alpha, L, w_max=wmax)

    # --- baseline parametric generator (single Gaussian) for contrast, same LP ---
    S_g = mc_generate_scenarios(X.mean(0), X.std(0), np.corrcoef(X.T), S=10000, copula="gaussian")
    w_n, er_n, es_n, _ = mc_max_return_cvar(S_g, alpha, L, w_max=wmax)

    print("Learned GMM scenarios -> same CVaR LP:")
    print(f"  fitted mixture weights {np.round(pi,2)}  (one calm, one crash component)")
    print(f"  weights {np.round(w_gmm,2)} | E[ret] {er_g:+.3f} | in-sample CVaR {es_g:+.3f}") if w_gmm is not None else print("  infeasible")
    print("Single-Gaussian scenarios -> same CVaR LP:")
    print(f"  weights {np.round(w_n,2)} | E[ret] {er_n:+.3f} | in-sample CVaR {es_n:+.3f}") if w_n is not None else print("  infeasible")
    print("\nThe learned generator is swapped in by replacing one function; the LP is unchanged.")
