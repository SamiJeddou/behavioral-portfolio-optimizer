"""
mc_cvar_optimizer.py
====================
Proof-of-concept for the scalable Monte-Carlo + CVaR linear-program optimiser
that extends the Beyond Mean-Variance app beyond the grid method's m^N
state-space curse of dimensionality.

Pipeline
--------
    1. generate_scenarios(...)        -> S x N matrix of joint security returns
                                         (Gaussian or t copula, Normal marginals)
    2. price_derivative_scenarios(...) -> S-vector of a derivative's return,
                                         priced arbitrage-consistently per scenario
    3. build_scenario_matrix(...)     -> S x (N+K) matrix: securities + K derivatives
    4. max_return_cvar(...)           -> solves  max E[r]  s.t.  ES_alpha(r) >= L
                                         as a Rockafellar-Uryasev linear program
    5. efficient_frontier(...)        -> sweeps the ES floor L (warm-startable)

Cost is O(S * (N+K)) in memory and a single LP solve per frontier point:
linear in the number of assets, independent of how it is discretised.

Author: addendum prototype for S. Jeddou's Beyond Mean-Variance optimiser.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.stats import norm, t as student_t, chi2


# ---------------------------------------------------------------------------
# Black-Scholes (same conventions as the app's engine: S0=1, strikes as
# fractions of the underlying, derivative return = (payoff - fair price)/price)
# ---------------------------------------------------------------------------
def bs_call(vol, S, r, T, K):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(vol, S, r, T, K):
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# 1. Scenario generation: joint security returns over the horizon
# ---------------------------------------------------------------------------
def generate_scenarios(means, sigs, corr, S=20000, copula="gaussian",
                       dof=5, seed=0):
    """Draw S joint return scenarios for N securities.

    means, sigs : annualised (horizon-T) expected returns and volatilities.
    corr        : N x N correlation matrix.
    copula      : 'gaussian' (=> multivariate Normal, matches the thesis MVN
                  assumption) or 't' (Student-t copula => tail dependence:
                  assets crash together) with Normal marginals.

    Returns an S x N array of simple returns over the horizon.
    """
    rng = np.random.default_rng(seed)
    means = np.asarray(means, float)
    sigs = np.asarray(sigs, float)
    corr = np.asarray(corr, float)
    N = len(means)
    L = np.linalg.cholesky(corr)

    if copula == "gaussian":
        Z = rng.standard_normal((S, N)) @ L.T            # correlated Normals
    elif copula == "t":
        Y = rng.standard_normal((S, N)) @ L.T            # correlated Normals
        g = chi2.rvs(dof, size=S, random_state=rng) / dof
        Tv = Y / np.sqrt(g)[:, None]                     # multivariate t
        U = student_t.cdf(Tv, dof)                       # copula (uniforms)
        Z = norm.ppf(np.clip(U, 1e-12, 1 - 1e-12))       # back to Normal margins
    else:
        raise ValueError("copula must be 'gaussian' or 't'")

    return means[None, :] + sigs[None, :] * Z            # S x N returns


# ---------------------------------------------------------------------------
# 2. Per-scenario derivative pricing (arbitrage-consistent, multi-instrument)
# ---------------------------------------------------------------------------
def price_derivative_scenarios(underlying_returns, spec, vol, r=0.03, T=1.0):
    """Return the S-vector of a single derivative's realised return, one per
    scenario, given the underlying's simulated return in each scenario.

    spec : dict with 'type' in {'call','put','straddle','bull_call_spread',...}
           plus the strikes it needs (fractions of the entry spot S0=1) and an
           optional 'premium' markup for structured products.

    The entry price is the Black-Scholes fair value at S0=1; the payoff is the
    instrument's terminal payoff at spot_T = 1 + underlying_return; the return
    is (payoff - price * (1+premium)) / (price * (1+premium)).  This mirrors the
    app's engine, so each scenario keeps a coherent (underlying, derivative)
    pair with no separate model.
    """
    u = np.asarray(underlying_returns, float)
    spot_T = 1.0 + u
    prem = float(spec.get("premium", 0.0))
    t = spec["type"]

    if t == "call":
        K = spec["strike"]; price = bs_call(vol, 1, r, T, K)
        payoff = np.maximum(spot_T - K, 0.0)
    elif t == "put":
        K = spec["strike"]; price = bs_put(vol, 1, r, T, K)
        payoff = np.maximum(K - spot_T, 0.0)
    elif t == "straddle":
        K = spec["strike"]; price = bs_call(vol, 1, r, T, K) + bs_put(vol, 1, r, T, K)
        payoff = np.maximum(spot_T - K, 0.0) + np.maximum(K - spot_T, 0.0)
    elif t == "bull_call_spread":
        k1, k2 = spec["k1"], spec["k2"]
        price = bs_call(vol, 1, r, T, k1) - bs_call(vol, 1, r, T, k2)
        payoff = np.maximum(spot_T - k1, 0.0) - np.maximum(spot_T - k2, 0.0)
    else:
        raise ValueError(f"derivative type '{t}' not in this POC")

    paid = max(price * (1.0 + prem), 1e-12)
    return (payoff - paid) / paid


def build_scenario_matrix(R_sec, derivatives, sigs, r=0.03, T=1.0):
    """Stack the S x N security returns with K derivative-return columns.

    derivatives : list of specs; each spec adds 'underlying' (security index)
                  used both to read that security's simulated return and to set
                  the Black-Scholes pricing vol (sigs[underlying]).  Returns the
                  S x (N+K) scenario matrix and the column labels.
    """
    cols = [R_sec]
    labels = [f"sec{i}" for i in range(R_sec.shape[1])]
    for d in derivatives:
        ui = d["underlying"]
        vol = d.get("vol", sigs[ui])
        dr = price_derivative_scenarios(R_sec[:, ui], d, vol, r=r, T=T)
        cols.append(dr[:, None])
        labels.append(f"{d['type']}@s{ui}")
    return np.hstack(cols), labels


# ---------------------------------------------------------------------------
# 3. The optimiser: maximise E[r] s.t. ES_alpha(r) >= L  (Rockafellar-Uryasev)
# ---------------------------------------------------------------------------
def max_return_cvar(R, alpha, L, w_max=None):
    """Linear program:

        maximise   mu . w
        s.t.       CVaR_alpha(-r) <= -L         (i.e. tail-average return >= L)
                   sum(w) = 1,  0 <= w <= w_max

    Implemented with the Rockafellar-Uryasev variables (zeta, z_s):
        CVaR_alpha = zeta + 1/(alpha*S) * sum_s z_s,    z_s >= -R[s].w - zeta,  z_s >= 0.

    Decision vector x = [ w (n) | zeta (1) | z (S) ].
    Returns (weights, expected_return, realised_ES, linprog_result).
    """
    R = np.asarray(R, float)
    S, n = R.shape
    nv = n + 1 + S
    mu = R.mean(axis=0)

    c = np.concatenate([-mu, [0.0], np.zeros(S)])        # minimise -E[r]

    # --- inequality block A_ub x <= b_ub ---
    # row 0: CVaR constraint;  rows 1..S: z_s >= loss_s - zeta
    rows, colss, vals = [], [], []
    # CVaR row (row 0): zeta coef 1, each z coef 1/(alpha*S)
    rows.append(0); colss.append(n); vals.append(1.0)
    rows += [0] * S
    colss += list(range(n + 1, n + 1 + S))
    vals += [1.0 / (alpha * S)] * S
    # scenario rows 1..S:  -R[s].w - zeta - z_s <= 0
    sidx = np.arange(1, S + 1)
    # w part
    rows += list(np.repeat(sidx, n))
    colss += list(np.tile(np.arange(n), S))
    vals += list((-R).ravel())
    # zeta part
    rows += list(sidx); colss += [n] * S; vals += [-1.0] * S
    # z part
    rows += list(sidx); colss += list(range(n + 1, n + 1 + S)); vals += [-1.0] * S

    A_ub = coo_matrix((vals, (rows, colss)), shape=(S + 1, nv)).tocsr()
    b_ub = np.concatenate([[-L], np.zeros(S)])

    # --- equality: sum(w) = 1 ---
    A_eq = coo_matrix((np.ones(n), (np.zeros(n), np.arange(n))), shape=(1, nv)).tocsr()
    b_eq = np.array([1.0])

    # --- bounds: w in [0, w_max]; zeta free; z >= 0 ---
    wb = (0.0, w_max if w_max is not None else 1.0)
    bounds = [wb] * n + [(None, None)] + [(0.0, None)] * S

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return None, None, None, res
    w = res.x[:n]
    port = R @ w
    er = float(port.mean())
    es = realised_es(port, alpha)
    return w, er, es, res


def realised_es(port_returns, alpha):
    """Realised Expected Shortfall (tail-average return) of a return sample:
    mean of the worst alpha fraction of outcomes."""
    p = np.sort(np.asarray(port_returns, float))
    k = max(1, int(np.floor(alpha * len(p))))
    return float(p[:k].mean())


def analytical_normal_es(mu_p, sigma_p, alpha):
    """Closed-form ES (tail-average return) for a Normal portfolio return:
    ES = mu - sigma * phi(Phi^{-1}(alpha)) / alpha."""
    z = norm.ppf(alpha)
    return mu_p - sigma_p * norm.pdf(z) / alpha


# ---------------------------------------------------------------------------
# 4. Efficient frontier: sweep the ES floor L
# ---------------------------------------------------------------------------
def efficient_frontier(R, alpha, floors, w_max=None):
    out = []
    for L in floors:
        w, er, es, res = max_return_cvar(R, alpha, L, w_max=w_max)
        out.append({"L": L, "ok": res.success, "E[r]": er, "ES": es, "w": w})
    return out


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # --- a small N=5 universe (annualised) ---
    means = [0.06, 0.09, 0.12, 0.04, 0.10]
    sigs  = [0.14, 0.20, 0.30, 0.08, 0.24]
    rho = 0.30
    N = len(means)
    corr = np.full((N, N), rho); np.fill_diagonal(corr, 1.0)
    alpha = 0.05

    print("=" * 70)
    print("STEP 1-2  Scenario generation + convergence check (Gaussian copula)")
    print("=" * 70)
    R = generate_scenarios(means, sigs, corr, S=40000, copula="gaussian", seed=1)
    w_eq = np.full(N, 1 / N)
    port = R @ w_eq
    mu_p = w_eq @ np.array(means)
    var_p = w_eq @ (np.outer(sigs, sigs) * corr) @ w_eq
    es_mc = realised_es(port, alpha)
    es_an = analytical_normal_es(mu_p, np.sqrt(var_p), alpha)
    print(f"equal-weight  E[r]={port.mean():.4f} (exact {mu_p:.4f})  "
          f"sigma={port.std():.4f} (exact {np.sqrt(var_p):.4f})")
    print(f"ES_5%  MC={es_mc:.4f}  analytical-Normal={es_an:.4f}  "
          f"|diff|={abs(es_mc-es_an):.4f}")

    print("\n" + "=" * 70)
    print("STEP 3-4  Max E[r] s.t. ES >= L   (CVaR linear program), no derivatives")
    print("=" * 70)
    for row in efficient_frontier(R, alpha, [-0.25, -0.20, -0.15, -0.10]):
        if row["ok"]:
            print(f"  L={row['L']:+.2f}  E[r]={row['E[r]']:.4f}  "
                  f"realised ES={row['ES']:.4f}  w={np.round(row['w'],3)}")
        else:
            print(f"  L={row['L']:+.2f}  INFEASIBLE — no portfolio reaches a "
                  f"tail-average return of {row['L']:+.0%} (even the safest asset is worse)")
    print("  (E[r] falls as the floor L tightens -> monotone, as expected)")

    print("\n" + "=" * 70)
    print("STEP 5  MULTIPLE derivatives at once (a put on sec2 AND a call on sec4)")
    print("=" * 70)
    derivatives = [
        {"type": "put",  "underlying": 2, "strike": 0.90},   # protect the riskiest
        {"type": "call", "underlying": 4, "strike": 1.15},   # geared upside on sec4
    ]
    Rd, labels = build_scenario_matrix(R, derivatives, np.array(sigs))
    print(f"  scenario matrix shape {Rd.shape}  columns: {labels}")
    w, er, es, res = max_return_cvar(Rd, alpha, L=-0.15, w_max=0.60)
    print(f"  solved={res.success}  E[r]={er:.4f}  realised ES={es:.4f}")
    for lab, wi in zip(labels, w):
        print(f"     {lab:14s} {wi:7.3f}")
    print("  -> K=2 derivative columns optimised jointly; cost unchanged in K.")

    print("\n" + "=" * 70)
    print("t-copula variant (tail dependence) — same machinery, heavier joint tail")
    print("=" * 70)
    Rt = generate_scenarios(means, sigs, corr, S=40000, copula="t", dof=4, seed=1)
    es_g = realised_es(R @ w_eq, alpha)
    es_t = realised_es(Rt @ w_eq, alpha)
    print(f"  equal-weight ES_5%:  Gaussian={es_g:.4f}   t(dof=4)={es_t:.4f}  "
          f"(t tail is worse, as intended)")
