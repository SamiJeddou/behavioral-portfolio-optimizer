# © 2026 Sami Jeddou. All rights reserved.
# Published publicly for demonstration and evaluation only — no license is granted.
# Copying, modification, redistribution, or reuse (in whole or in part) without the
# author's prior written permission is prohibited.
"""
Behavioral Portfolio Optimizer — Python Translation
====================================================
Based on: "Beyond Mean-Variance: Options and Structured Products in Behavioral Portfolios"
Author of original R code: Sami Ben Jeddou (Master in Finance, University of Lugano, 2012)
Supervisor: Prof. Enrico De Giorgi

Reference paper: Das, Markowitz, Scheid & Statman (2010)
"Portfolio Optimization with Mental Accounts", JFQA Vol. 45, No. 2, pp. 311-334

Algorithm:
  Step 1 — Build state space U (primary securities + derivative payoffs)
  Step 2 — Assign probabilities via Gaussian copula
  Step 3 — Grid search (<=4 assets) or differential evolution (5+ assets),
            then COBYLA gradient refinement
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize, differential_evolution
from itertools import product as cartesian_product


# =============================================================================
# BLACK-SCHOLES & EXOTIC PRICING
# =============================================================================

def bs_call(vol, S0, r, T, K):
    if vol <= 0 or T <= 0 or S0 <= 0 or K <= 0:
        return max(0.0, S0 - K * np.exp(-r * T))
    d1 = (np.log(S0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(vol, S0, r, T, K):
    if vol <= 0 or T <= 0 or S0 <= 0 or K <= 0:
        return max(0.0, K * np.exp(-r * T) - S0)
    d1 = (np.log(S0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def con_call(vol, S0, r, T, K):
    if vol <= 0 or T <= 0:
        return np.exp(-r * T) if S0 > K else 0.0
    d2 = (np.log(S0 / K) + (r - 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    return np.exp(-r * T) * norm.cdf(d2)


def con_put(vol, S0, r, T, K):
    if vol <= 0 or T <= 0:
        return np.exp(-r * T) if S0 < K else 0.0
    d2 = (np.log(S0 / K) + (r - 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    return np.exp(-r * T) * norm.cdf(-d2)


# =============================================================================
# CUSTOM STRUCTURED PRODUCT PAYOFF
# =============================================================================

def compute_structured_payoff(spot_returns, components, vol, S0, r, T, premium=0.0):
    """
    Compute returns of a custom structured product for each state.

    Parameters
    ----------
    spot_returns : np.ndarray  shape (l,)
        Return of the underlying security in each state
    components : list of dict
        Each dict has keys:
          'type'     : 'long_call'|'short_call'|'long_put'|'short_put'|
                       'long_digital_call'|'short_digital_call'|
                       'long_digital_put'|'short_digital_put'|'zcb'
          'strike'   : float (as fraction of spot, e.g. 1.0 = ATM)
          'notional' : float (e.g. 1.0)
          'maturity' : float (years, overrides T if provided)
    vol   : float   underlying volatility
    S0    : float   current spot (usually 1.0)
    r     : float   risk-free rate
    T     : float   default maturity

    Returns
    -------
    der_returns : np.ndarray  shape (l,)
    price0      : float  fair value of the structured product today
    """
    l = len(spot_returns)
    price0 = 0.0
    payoff = np.zeros(l)

    for comp in components:
        ctype    = comp['type']
        K        = comp.get('strike', 1.0)
        notional = comp.get('notional', 1.0)
        mat      = comp.get('maturity', T)
        sign     = -1.0 if 'short' in ctype else 1.0
        spot_T   = 1 + spot_returns  # terminal price (S0=1)

        if 'zcb' in ctype:
            price0  += notional * np.exp(-r * mat)
            payoff  += notional * np.ones(l)
        elif 'digital_call' in ctype:
            price0 += sign * notional * con_call(vol, S0, r, mat, K)
            payoff += sign * notional * (spot_T >= K).astype(float)
        elif 'digital_put' in ctype:
            price0 += sign * notional * con_put(vol, S0, r, mat, K)
            payoff += sign * notional * (spot_T <= K).astype(float)
        elif 'call' in ctype:
            price0 += sign * notional * bs_call(vol, S0, r, mat, K)
            payoff += sign * notional * np.maximum(0, spot_T - K)
        elif 'put' in ctype:
            price0 += sign * notional * bs_put(vol, S0, r, mat, K)
            payoff += sign * notional * np.maximum(0, K - spot_T)

    if price0 <= 0:
        price0 = 1.0  # fallback to avoid division by zero

    price0 = price0 * (1.0 + premium)  # institutional markup over replication cost
    der_returns = (payoff - price0) / price0
    return der_returns, price0


# =============================================================================
# STEP 1 — STATE SPACE CONSTRUCTION
# =============================================================================

def build_state_space(means, sigs, m=51, derivative_config=None):
    """
    Build full state space U.

    derivative_config can be:
      - Standard dict with 'type' key (put, call, cgn, etc.)
      - Dict with 'type' = 'custom' and 'components' list for structured products
      - None for no derivative
    """
    n_prime = len(means)
    means   = np.array(means)
    sigs    = np.array(sigs)

    grids     = []
    dr_values = []
    for i in range(n_prime):
        lo   = means[i] - 5 * sigs[i]
        hi   = means[i] + 5 * sigs[i]
        grid = np.linspace(lo, hi, m)
        grids.append(grid)
        dr_values.append(grid[1] - grid[0])

    mesh = np.array(list(cartesian_product(*grids)))
    l    = len(mesh)

    n_securities = n_prime + (1 if derivative_config is not None else 0)
    U = np.full((l, n_securities + 1), np.nan)
    U[:, :n_prime] = mesh

    if derivative_config is not None:
        sec_idx = derivative_config.get('underlying_index', n_prime - 1)
        vol     = derivative_config.get('vol', sigs[sec_idx])
        S0      = derivative_config.get('S0', 1.0)
        r       = derivative_config.get('r', 0.03)
        T       = derivative_config.get('T', 1.0)
        dtype   = derivative_config['type']

        if dtype == 'custom':
            components = derivative_config['components']
            der_returns, _ = compute_structured_payoff(
                mesh[:, sec_idx], components, vol, S0, r, T,
                premium=derivative_config.get('premium', 0.0))
            U[:, n_prime] = der_returns

        elif dtype == 'put':
            K  = derivative_config['strike']
            P0 = bs_put(vol, S0, r, T, K)
            P0 = max(P0, 1e-8)
            spot_T = 1 + mesh[:, sec_idx]
            U[:, n_prime] = (np.maximum(0, K - spot_T) - P0) / P0

        elif dtype == 'call':
            K  = derivative_config['strike']
            C0 = bs_call(vol, S0, r, T, K)
            C0 = max(C0, 1e-8)
            spot_T = 1 + mesh[:, sec_idx]
            U[:, n_prime] = (np.maximum(0, spot_T - K) - C0) / C0

        elif dtype == 'safety_collar':
            Kp = derivative_config['strike_p']
            Kc = derivative_config['strike_c']
            P0 = bs_put(vol, S0, r, T, Kp)
            C0 = bs_call(vol, S0, r, T, Kc)
            SC0 = P0 - C0
            denom = max(P0 + C0, 1e-8)
            spot_T = 1 + mesh[:, sec_idx]
            SCU = np.maximum(0, Kp - spot_T) - np.maximum(0, spot_T - Kc)
            U[:, n_prime] = (SCU - SC0) / denom

        elif dtype == 'aggressive_collar':
            Kp = derivative_config['strike_p']
            Kc = derivative_config['strike_c']
            P0 = bs_put(vol, S0, r, T, Kp)
            C0 = bs_call(vol, S0, r, T, Kc)
            AC0 = C0 - P0
            denom = max(P0 + C0, 1e-8)
            spot_T = 1 + mesh[:, sec_idx]
            ACU = np.maximum(0, spot_T - Kc) - np.maximum(0, Kp - spot_T)
            U[:, n_prime] = (ACU - AC0) / denom

        elif dtype == 'straddle':
            K  = derivative_config['strike']
            P0 = bs_put(vol, S0, r, T, K)
            C0 = bs_call(vol, S0, r, T, K)
            ST0 = max(P0 + C0, 1e-8)
            spot_T = 1 + mesh[:, sec_idx]
            STU = np.maximum(0, K - spot_T) + np.maximum(0, spot_T - K)
            U[:, n_prime] = (STU - ST0) / ST0

        elif dtype == 'strangle':
            Kp  = derivative_config['strike_kp']
            Kc  = derivative_config['strike_kc']
            P0  = bs_put(vol, S0, r, T, Kp)
            C0  = bs_call(vol, S0, r, T, Kc)
            ST0 = max(P0 + C0, 1e-8)
            spot_T = 1 + mesh[:, sec_idx]
            STU = np.maximum(0, Kp - spot_T) + np.maximum(0, spot_T - Kc)
            U[:, n_prime] = (STU - ST0) / ST0

        elif dtype == 'cgn':
            floor         = derivative_config.get('floor', 0.01)
            participation = derivative_config.get('participation', 1.0)
            cap           = derivative_config.get('cap', None)
            premium       = derivative_config.get('cgn_premium', 0.0)
            StrikeC1      = 1 + floor
            PVNotional    = np.exp(-r * T) * (1 + floor)
            if cap is None:
                UpsPayoff = participation * bs_call(vol, S0, r, T, StrikeC1)
            else:
                StrikeC2  = 1 + cap
                UpsPayoff = participation * (
                    bs_call(vol, S0, r, T, StrikeC1) -
                    bs_call(vol, S0, r, T, StrikeC2))
            # No-arbitrage replication cost of the note: a zero-coupon bond
            # paying (1+floor) at maturity, plus the participation upside
            # call(s). The note's return is measured against THIS cost (CG0),
            # consistent with every other derivative here (put/call/collar/
            # straddle each divide by their fair premium). Previously the note
            # was priced at par (cost 1.0), which underpriced it by CG0-1
            # (~7-11% at typical equity vols) because the full-participation
            # upside call was never funded — inflating every portfolio that
            # held it and lifting the derivative frontier above the MV frontier.
            CG0      = (PVNotional + UpsPayoff) * (1 + premium)
            spot_T   = 1 + mesh[:, sec_idx]
            if cap is None:
                ups = participation * np.maximum(0, spot_T - StrikeC1)
            else:
                ups = participation * (
                    np.maximum(0, spot_T - StrikeC1) -
                    np.maximum(0, spot_T - StrikeC2))
            CGU = (1 + floor) + ups
            U[:, n_prime] = (CGU - CG0) / max(CG0, 1e-8)

        elif dtype == 'barrier_m':
            M          = derivative_config.get('M', 0.40)
            premium_bm = derivative_config.get('premium_bm', 0.10)
            StrikeLC   = derivative_config.get('strike_lc', 1.0)
            StrikeLP   = derivative_config.get('strike_lp', 1.0)
            StrikeSC   = 1 + M
            StrikeSP   = 1 - M
            LC0  = bs_call(vol, S0, r, T, StrikeLC)
            LP0  = bs_put(vol, S0, r, T, StrikeLP)
            SC0  = bs_call(vol, S0, r, T, StrikeSC)
            SP0  = bs_put(vol, S0, r, T, StrikeSP)
            CONC0 = con_call(vol, S0, r, T, StrikeSC)
            CONP0 = con_put(vol, S0, r, T, StrikeSP)
            PVU  = S0 * np.exp(-r * T)
            BN0  = (PVU + LC0 + LP0 - SC0 - SP0
                    - M * CONC0 - M * CONP0) * (1 + premium_bm)
            BN0  = max(BN0, 1e-8)
            und_ret = mesh[:, sec_idx]
            payoff  = np.where(np.abs(und_ret) <= M,
                               np.abs(und_ret) / BN0, 0.0)
            U[:, n_prime] = payoff

    return U, dr_values


# =============================================================================
# STEP 2 — PROBABILITIES VIA GAUSSIAN COPULA
# =============================================================================

def assign_probabilities(U, means, sigs, cov_matrix, dr_values,
                         distribution='gaussian'):
    n_prime    = len(means)
    means      = np.array(means)
    sigs       = np.array(sigs)
    cov_matrix = np.array(cov_matrix)
    pidri      = np.prod(dr_values)

    if distribution == 'gaussian':
        mv_dist = multivariate_normal(mean=means, cov=cov_matrix,
                                      allow_singular=True)
        probs   = mv_dist.pdf(U[:, :n_prime]) * pidri
    else:
        from scipy.stats import t as t_dist
        df   = 5
        rho  = np.zeros((n_prime, n_prime))
        for i in range(n_prime):
            for j in range(n_prime):
                rho[i, j] = cov_matrix[i, j] / (sigs[i] * sigs[j])
        probs = np.zeros(len(U))
        for i in range(len(U)):
            returns_i   = U[i, :n_prime]
            standardised = [(returns_i[k] - means[k]) / sigs[k]
                            for k in range(n_prime)]
            uniform      = [t_dist.cdf(standardised[k], df=df)
                            for k in range(n_prime)]
            nq           = norm.ppf(np.clip(uniform, 1e-10, 1 - 1e-10))
            gc           = multivariate_normal(
                mean=np.zeros(n_prime), cov=rho).pdf(nq)
            marginal     = np.prod([t_dist.pdf(standardised[k], df=df) / sigs[k]
                                    for k in range(n_prime)])
            probs[i]     = gc * marginal * pidri

    total = probs.sum()
    if total <= 0:
        probs = np.ones(len(U)) / len(U)
    else:
        U[:, -1] = probs / total
    return U


# =============================================================================
# STEP 3 — OPTIMIZATION
# =============================================================================

def optimize_portfolio(U, n_securities, constraint_type='var',
                       H=-0.10, alpha=0.05,
                       m_prime=99, L=None, penalty=1e18,
                       method='auto'):
    """
    Optimize portfolio weights.

    method : 'auto'  — grid search if n<=4, differential evolution if n>=5
             'grid'  — force grid search (slow for n>=5)
             'de'    — force differential evolution

    penalty : quadratic penalty scalar applied when the shortfall constraint
              P(return < H) > alpha is violated. Set to 1e18 (effectively
              infinite) to enforce hard feasibility. This is a numerical
              implementation detail with no financial interpretation — do not
              expose to end users. Lowering it risks ignoring the constraint;
              raising it risks numerical overflow. Tune only in source if
              experiencing solver instability with specific security configurations.
    """
    probs          = U[:, -1]
    returns_matrix = U[:, :n_securities]

    use_de = (method == 'de') or (method == 'auto' and n_securities >= 5)

    # ── Stage 1a: Grid search ─────────────────────────────────────────────────
    if not use_de:
        step       = 1.0 / m_prime
        w_grid     = np.arange(step, 1.0, step)
        n_free     = n_securities - 1

        if n_free == 1:
            combos = [(w,) for w in w_grid if w <= 1.0]
        elif n_free == 2:
            combos = [(w1, w2) for w1 in w_grid for w2 in w_grid
                      if w1 + w2 <= 1.0]
        else:
            combos = [(w1, w2, w3) for w1 in w_grid for w2 in w_grid
                      for w3 in w_grid if w1 + w2 + w3 <= 1.0]

        best_return  = -np.inf
        best_weights = None
        eligible_count = 0

        for combo in combos:
            w        = np.array(list(combo) + [1.0 - sum(combo)])
            if np.any(w < 0): continue
            port_ret = returns_matrix @ w
            exp_r    = float(port_ret @ probs)
            eligible = False
            if constraint_type == 'var':
                if float(probs[port_ret < H].sum()) <= alpha:
                    eligible = True
            elif constraint_type == 'es':
                tail = port_ret < H
                if tail.sum() > 0:
                    es = float((port_ret[tail] * probs[tail]).sum()
                               / probs[tail].sum())
                    if L is None or es >= L:
                        eligible = True
            if eligible:
                eligible_count += 1
                if exp_r > best_return:
                    best_return  = exp_r
                    best_weights = w.copy()

        if best_weights is None:
            raise ValueError(
                "No portfolio meets the risk limit at these settings. Try a more lenient loss threshold (H), allow a higher probability of loss (alpha), or use a higher resolution.")

    # ── Stage 1b: Differential evolution (5+ securities) ─────────────────────
    else:
        eligible_count = -1  # not applicable for DE

        def de_objective(w_partial):
            w_last = 1.0 - w_partial.sum()
            if w_last < 0 or w_last > 1:
                return 1e10
            w        = np.append(w_partial, w_last)
            port_ret = returns_matrix @ w
            exp_r    = float(port_ret @ probs)
            shortfall = float(probs[port_ret < H].sum())
            return -exp_r + penalty * (max(0, shortfall - alpha)) ** 2

        bounds = [(0, 1)] * (n_securities - 1)
        de_result = differential_evolution(
            de_objective, bounds,
            seed=42, maxiter=500, popsize=12,
            tol=1e-7, workers=1, polish=True)

        w_de = np.append(de_result.x, 1.0 - de_result.x.sum())
        w_de = np.clip(w_de, 0, 1)
        w_de /= w_de.sum()
        best_weights = w_de

    # ── Stage 2: Gradient refinement (COBYLA) ─────────────────────────────────
    def objective(wei):
        w_full   = np.append(wei, 1.0 - wei.sum())
        port_ret = returns_matrix @ w_full
        term1    = float(port_ret @ probs)
        shortfall = float(probs[port_ret < H].sum())
        term2    = penalty * (alpha - shortfall) ** 2
        return -term1 + term2

    x0     = best_weights[:-1]
    bounds = [(0.0, 1.0)] * (n_securities - 1)

    result = minimize(
        objective, x0=x0, method='COBYLA',
        bounds=bounds,
        options={'rhobeg': 0.01, 'maxiter': 10000, 'catol': 1e-8})

    w_opt = np.append(result.x, 1.0 - result.x.sum())
    w_opt = np.clip(w_opt, 0, 1)
    w_opt /= w_opt.sum()

    # ── Portfolio statistics ──────────────────────────────────────────────────
    port_ret = returns_matrix @ w_opt
    mean_r   = float(port_ret @ probs)
    variance = float(((port_ret - mean_r) ** 2) @ probs)
    std_dev  = np.sqrt(max(variance, 0))
    skewness = (float(((port_ret - mean_r) ** 3) @ probs)
                / std_dev ** 3 if std_dev > 0 else 0.0)
    kurtosis = (float(((port_ret - mean_r) ** 4) @ probs)
                / std_dev ** 4 - 3 if std_dev > 0 else 0.0)

    if constraint_type == 'var':
        q_stat = float(probs[port_ret < H].sum())
    else:
        tail = port_ret < H
        q_stat = (float((port_ret[tail] * probs[tail]).sum()
                        / probs[tail].sum())
                  if tail.sum() > 0 else 0.0)

    return {
        'weights':          w_opt,
        'expected_return':  mean_r,
        'std_dev':          std_dev,
        'skewness':         skewness,
        'excess_kurtosis':  kurtosis,
        'shortfall_stat':   q_stat,
        'eligible_count':   eligible_count,
        'method_used':      'differential_evolution' if use_de else 'grid_search',
    }


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def run_thesis_base_case(derivative_type='put', H=-0.10, alpha=0.05,
                         m=51, m_prime=99):
    means       = [0.05, 0.10, 0.25]
    sigs        = [0.05, 0.20, 0.50]
    covariances = [0.0025, 0, 0, 0, 0.04, 0.02, 0, 0.02, 0.25]
    cov_matrix  = np.array(covariances).reshape(3, 3)

    der_configs = {
        'put':               {'type':'put',               'underlying_index':2,'vol':sigs[2],'S0':1,'r':0.03,'T':1,'strike':1.4},
        'call':              {'type':'call',              'underlying_index':2,'vol':sigs[2],'S0':1,'r':0.03,'T':1,'strike':1.2},
        'safety_collar':     {'type':'safety_collar',     'underlying_index':2,'vol':sigs[2],'S0':1,'r':0.03,'T':1,'strike_c':1.6,'strike_p':1.2},
        'aggressive_collar': {'type':'aggressive_collar', 'underlying_index':2,'vol':sigs[2],'S0':1,'r':0.03,'T':1,'strike_c':1.6,'strike_p':1.2},
        'straddle':          {'type':'straddle',          'underlying_index':2,'vol':sigs[2],'S0':1,'r':0.03,'T':1,'strike':0.7},
        'strangle':          {'type':'strangle',          'underlying_index':2,'vol':sigs[2],'S0':1,'r':0.03,'T':1,'strike_kp':0.8,'strike_kc':0.9},
        'cgn':               {'type':'cgn',               'underlying_index':2,'vol':sigs[2],'S0':1,'r':0.03,'T':1,'floor':0.01,'participation':1.0,'cap':None,'cgn_premium':0.00},
        'barrier_m':         {'type':'barrier_m',         'underlying_index':2,'vol':sigs[2],'S0':1,'r':0.03,'T':1,'M':0.40,'premium_bm':0.10},
        'none':              None,
    }

    der_config = der_configs.get(derivative_type)
    print(f"\nRunning: {derivative_type.upper()} | H={H:.0%} | alpha={alpha:.0%} | m={m} | m'={m_prime}")
    print("Step 1: Building state space...")
    U, dr_values = build_state_space(means, sigs, m=m, derivative_config=der_config)
    n_sec = U.shape[1] - 1
    print("Step 2: Assigning probabilities...")
    U = assign_probabilities(U, means, sigs, cov_matrix, dr_values)
    print(f"Step 3: Optimising ({n_sec} securities, {len(U):,} states)...")
    result = optimize_portfolio(U, n_sec, H=H, alpha=alpha, m_prime=m_prime)

    print("\n--- Results ---")
    labels = ['Sec1 (low-risk)', 'Sec2 (mid-risk)', 'Sec3 (high-risk)', 'Derivative']
    for i, w in enumerate(result['weights']):
        print(f"  {labels[min(i,3)]}: {w:.4f} ({w*100:.1f}%)")
    print(f"  Expected return:  {result['expected_return']:.4f} ({result['expected_return']*100:.2f}%)")
    print(f"  Std deviation:    {result['std_dev']:.4f}")
    print(f"  Skewness:         {result['skewness']:.4f}")
    print(f"  Excess kurtosis:  {result['excess_kurtosis']:.4f}")
    print(f"  Shortfall prob:   {result['shortfall_stat']:.4f} (constraint: <= {alpha:.2f})")
    print(f"  Method:           {result['method_used']}")
    return result


if __name__ == "__main__":
    run_thesis_base_case(derivative_type='cgn', H=-0.10, alpha=0.05, m=21, m_prime=15)
