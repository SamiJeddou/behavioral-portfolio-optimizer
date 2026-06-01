"""
Behavioral Portfolio Optimizer — Python Translation
====================================================
Based on: "Beyond Mean-Variance: Options and Structured Products in Behavioral Portfolios"
Author of original R code: Sami Ben Jeddou (Master in Finance, University of Lugano, 2012)
Supervisor: Prof. Enrico De Giorgi

Reference paper: Das & Statman, "Beyond Mean-Variance: Portfolios with Derivatives
and Non-Normal Returns in Mental Accounts"

Algorithm overview (3 steps):
  Step 1 — Build state space U of all possible return vectors (primary + derivatives)
  Step 2 — Assign probabilities via Gaussian copula
  Step 3 — Grid search for best eligible weights, then gradient refinement

Supported derivatives: put, call, safety collar, aggressive collar,
                       straddle, strangle, capital-guaranteed note (CGN), barrier-M note
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
from itertools import product as cartesian_product


# =============================================================================
# BLACK-SCHOLES PRICING FUNCTIONS
# (translated from ThesisFunctions.R: BSPut, BSCall, CONCall, CONPut)
# =============================================================================

def bs_call(vol, S0, r, T, K):
    """Black-Scholes call price."""
    d1 = (np.log(S0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(vol, S0, r, T, K):
    """Black-Scholes put price."""
    d1 = (np.log(S0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def con_call(vol, S0, r, T, K):
    """Cash-or-nothing call (pays $1 if S_T > K)."""
    d2 = (np.log(S0 / K) + (r - 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    return np.exp(-r * T) * norm.cdf(d2)


def con_put(vol, S0, r, T, K):
    """Cash-or-nothing put (pays $1 if S_T < K)."""
    d2 = (np.log(S0 / K) + (r - 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    return np.exp(-r * T) * norm.cdf(-d2)


# =============================================================================
# STEP 1 — STATE SPACE CONSTRUCTION
# (translated from Main Program, Step 1 in thesis appendix)
# =============================================================================

def build_state_space(means, sigs, m=51, derivative_config=None):
    """
    Build the full state space U of return vectors.

    Parameters
    ----------
    means : list of float
        Mean returns for each primary security. e.g. [0.05, 0.10, 0.25]
    sigs : list of float
        Std deviations for each primary security. e.g. [0.05, 0.20, 0.50]
    m : int
        Number of grid steps per security (thesis default: 51).
        Higher = more accurate but slower.
    derivative_config : dict or None
        Configuration for a single derivative. Keys:
          - 'type': one of 'put', 'call', 'safety_collar', 'aggressive_collar',
                    'straddle', 'strangle', 'cgn', 'barrier_m'
          - 'underlying_index': which primary security it is based on (0-indexed)
          - 'vol', 'S0', 'r', 'T': Black-Scholes params
          - type-specific params: 'strike', 'strike_c', 'strike_p',
                                  'strike_kc', 'strike_kp', 'participation',
                                  'floor', 'cap', 'M', 'premium_bm'
        If None, no derivative is added.

    Returns
    -------
    U : np.ndarray, shape (m^n_prime, n_securities + 1)
        Columns: returns of each security + derivative (if any), last col = probability (filled in Step 2)
    grid_steps : list of float
        The dr values for each primary security (needed for probability normalization)
    """
    n_prime = len(means)

    # Build the return grid for each primary security
    # Thesis uses: x = seq(-0.25, 0.35, 0.012), y = seq(-0.70, 0.90, 0.032), z = seq(-1.35, 1.85, 0.063)
    # We derive supporting ranges as mean ± 5*sigma, matching that grid density
    grids = []
    dr_values = []
    for i in range(n_prime):
        lo = means[i] - 5 * sigs[i]
        hi = means[i] + 5 * sigs[i]
        grid = np.linspace(lo, hi, m)
        grids.append(grid)
        dr_values.append(grid[1] - grid[0])

    # U_prime: all combinations of primary security returns (m^n_prime rows)
    mesh = np.array(list(cartesian_product(*grids)))  # shape: (m^n_prime, n_prime)
    l = len(mesh)

    # Determine total number of securities
    n_securities = n_prime + (1 if derivative_config is not None else 0)

    # Initialise U with NaN probability column
    U = np.full((l, n_securities + 1), np.nan)
    U[:, :n_prime] = mesh

    # Compute derivative returns for each state
    if derivative_config is not None:
        sec_idx = derivative_config.get('underlying_index', n_prime - 1)
        dtype = derivative_config['type']
        vol = derivative_config.get('vol', sigs[sec_idx])
        S0 = derivative_config.get('S0', 1.0)
        r = derivative_config.get('r', 0.03)
        T = derivative_config.get('T', 1.0)

        der_returns = np.zeros(l)

        if dtype == 'put':
            K = derivative_config['strike']
            P0 = bs_put(vol, S0, r, T, K)
            for i in range(l):
                PU = max(0, K - (1 + mesh[i, sec_idx]))
                der_returns[i] = (PU - P0) / P0

        elif dtype == 'call':
            K = derivative_config['strike']
            C0 = bs_call(vol, S0, r, T, K)
            for i in range(l):
                CU = max(0, (1 + mesh[i, sec_idx]) - K)
                der_returns[i] = (CU - C0) / C0

        elif dtype == 'safety_collar':
            Kp = derivative_config['strike_p']
            Kc = derivative_config['strike_c']
            P0 = bs_put(vol, S0, r, T, Kp)
            C0 = bs_call(vol, S0, r, T, Kc)
            SC0 = P0 - C0
            for i in range(l):
                CU = max(0, (1 + mesh[i, sec_idx]) - Kc)
                PU = max(0, Kp - (1 + mesh[i, sec_idx]))
                SCU = PU - CU
                der_returns[i] = (SCU - SC0) / (P0 + C0)

        elif dtype == 'aggressive_collar':
            Kp = derivative_config['strike_p']
            Kc = derivative_config['strike_c']
            P0 = bs_put(vol, S0, r, T, Kp)
            C0 = bs_call(vol, S0, r, T, Kc)
            AC0 = C0 - P0
            for i in range(l):
                CU = max(0, (1 + mesh[i, sec_idx]) - Kc)
                PU = max(0, Kp - (1 + mesh[i, sec_idx]))
                ACU = CU - PU
                der_returns[i] = (ACU - AC0) / (P0 + C0)

        elif dtype == 'straddle':
            K = derivative_config['strike']
            P0 = bs_put(vol, S0, r, T, K)
            C0 = bs_call(vol, S0, r, T, K)
            ST0 = P0 + C0
            for i in range(l):
                CU = max(0, (1 + mesh[i, sec_idx]) - K)
                PU = max(0, K - (1 + mesh[i, sec_idx]))
                der_returns[i] = (PU + CU - ST0) / ST0

        elif dtype == 'strangle':
            Kp = derivative_config['strike_kp']
            Kc = derivative_config['strike_kc']
            P0 = bs_put(vol, S0, r, T, Kp)
            C0 = bs_call(vol, S0, r, T, Kc)
            ST0 = P0 + C0
            for i in range(l):
                CU = max(0, (1 + mesh[i, sec_idx]) - Kc)
                PU = max(0, Kp - (1 + mesh[i, sec_idx]))
                der_returns[i] = (PU + CU - ST0) / ST0

        elif dtype == 'cgn':
            # Capital-guaranteed note (uncapped or capped)
            floor = derivative_config.get('floor', 0.01)
            participation = derivative_config.get('participation', 1.0)
            cap = derivative_config.get('cap', None)  # None = uncapped
            premium = derivative_config.get('cgn_premium', 0.0)
            StrikeC1 = 1 + floor
            PVNotional = np.exp(-r * T) * (1 + floor)
            if cap is None:
                UpsPayoff = participation * bs_call(vol, S0, r, T, StrikeC1)
            else:
                StrikeC2 = 1 + cap
                UpsPayoff = participation * (
                    bs_call(vol, S0, r, T, StrikeC1) - bs_call(vol, S0, r, T, StrikeC2)
                )
            CG0 = (PVNotional + UpsPayoff) * (1 + premium)
            eff_floor = floor / CG0
            StrikeC1Eff = 1 + eff_floor
            Notional = 1 + eff_floor
            CG0_norm = 1.0
            for i in range(l):
                if cap is None:
                    ups = participation * max(0, (1 + mesh[i, sec_idx]) - StrikeC1Eff)
                else:
                    eff_cap = cap / CG0
                    StrikeC2Eff = 1 + eff_cap
                    ups = participation * (
                        max(0, (1 + mesh[i, sec_idx]) - StrikeC1Eff)
                        - max(0, (1 + mesh[i, sec_idx]) - StrikeC2Eff)
                    )
                CGU = Notional + ups
                der_returns[i] = (CGU - CG0_norm) / CG0_norm

        elif dtype == 'barrier_m':
            M = derivative_config.get('M', 0.40)
            premium_bm = derivative_config.get('premium_bm', 0.10)
            StrikeLC = derivative_config.get('strike_lc', 1.0)
            StrikeLP = derivative_config.get('strike_lp', 1.0)
            StrikeSC = 1 + M
            StrikeSP = 1 - M
            LC0 = bs_call(vol, S0, r, T, StrikeLC)
            LP0 = bs_put(vol, S0, r, T, StrikeLP)
            SC0 = bs_call(vol, S0, r, T, StrikeSC)
            SP0 = bs_put(vol, S0, r, T, StrikeSP)
            CONC0 = con_call(vol, S0, r, T, StrikeSC)
            CONP0 = con_put(vol, S0, r, T, StrikeSP)
            PVUnderlying = S0 * np.exp(-r * T)
            OptionsPrice = LC0 + LP0 - SC0 - SP0 - M * CONC0 - M * CONP0
            BN0 = (PVUnderlying + OptionsPrice) * (1 + premium_bm)
            for i in range(l):
                und_ret = mesh[i, sec_idx]
                if abs(und_ret) <= M:
                    der_returns[i] = abs(und_ret) * (1 / BN0)
                else:
                    der_returns[i] = 0.0

        U[:, n_prime] = der_returns

    return U, dr_values


# =============================================================================
# STEP 2 — PROBABILITIES VIA GAUSSIAN COPULA
# (translated from Main Program, Step 2 in thesis appendix)
# =============================================================================

def assign_probabilities(U, means, sigs, cov_matrix, dr_values, distribution='gaussian'):
    """
    Assign a probability to each state in U using a Gaussian copula.

    Parameters
    ----------
    U : np.ndarray
        State space matrix from build_state_space (last column = probabilities, currently NaN)
    means : list of float
    sigs : list of float
    cov_matrix : np.ndarray
        Covariance matrix of primary security returns
    dr_values : list of float
        Grid step sizes for each primary security
    distribution : str
        'gaussian' for normal marginals + Gaussian copula (default)
        'student_t' for t marginals (df=5) + Gaussian copula — matches thesis extension

    Returns
    -------
    U : np.ndarray
        Same matrix, last column now filled with normalised probabilities
    """
    n_prime = len(means)
    l = len(U)

    # Correlation matrix from covariance
    rho = np.zeros((n_prime, n_prime))
    for i in range(n_prime):
        for j in range(n_prime):
            rho[i, j] = cov_matrix[i, j] / (sigs[i] * sigs[j])

    # Volume element (product of all dr steps)
    pidri = np.prod(dr_values)

    # Gaussian copula density: C(u1,..,un) = phi_n(Phi^-1(u1),..,Phi^-1(un)) / prod(phi(Phi^-1(ui)))
    # For Gaussian marginals this simplifies to the multivariate normal density directly
    if distribution == 'gaussian':
        mv_dist = multivariate_normal(mean=means, cov=cov_matrix)
        probs = mv_dist.pdf(U[:, :n_prime]) * pidri
    else:
        # Student-t marginals with Gaussian copula (thesis extension, df=5)
        df = 5
        from scipy.stats import t as t_dist, multivariate_normal as mvn
        probs = np.zeros(l)
        for i in range(l):
            returns_i = U[i, :n_prime]
            # Standardise
            standardised = [(returns_i[k] - means[k]) / sigs[k] for k in range(n_prime)]
            # Uniform scores via t CDF
            uniform = [t_dist.cdf(standardised[k], df=df) for k in range(n_prime)]
            # Gaussian copula density at those uniform scores
            normal_quantiles = norm.ppf(uniform)
            gc = multivariate_normal(mean=np.zeros(n_prime), cov=rho).pdf(normal_quantiles)
            # Marginal t densities
            marginal = np.prod([t_dist.pdf(standardised[k], df=df) / sigs[k]
                                for k in range(n_prime)])
            probs[i] = gc * marginal * pidri

    # Normalise so probabilities sum to 1
    total = probs.sum()
    U[:, -1] = probs / total
    return U


# =============================================================================
# STEP 3 — GRID SEARCH + GRADIENT OPTIMISATION
# (translated from Main Program, Step 3 in thesis appendix)
# =============================================================================

def optimize_portfolio(U, n_securities, constraint_type='var', H=-0.10, alpha=0.05,
                       m_prime=99, L=None, penalty=1e18):
    """
    Find the optimal portfolio weights maximising expected return subject to
    a mental-accounting (VaR or ES) constraint.

    Stage 1: Grid search over weight space to find approximate global optimum.
    Stage 2: Gradient-based refinement from that starting point.

    Parameters
    ----------
    U : np.ndarray
        Full state space with probabilities (from Steps 1+2)
    n_securities : int
        Total number of securities (primaries + derivative)
    constraint_type : str
        'var' — probability of return < H must be <= alpha (VaR constraint)
        'es'  — expected shortfall of returns below H must be >= L (ES constraint)
    H : float
        Mental-account threshold (e.g. -0.10 = -10%)
    alpha : float
        Maximum allowed shortfall probability (VaR case, e.g. 0.05)
    m_prime : int
        Number of grid steps per weight dimension for grid search (thesis default: 99)
        Reduce for speed (e.g. 20), increase for accuracy.
    L : float or None
        ES lower bound (ES case only, e.g. -0.15 means expected loss in tail <= -15%)
    penalty : float
        Penalty scalar X for constraint violation in objective function (thesis: 1e18)

    Returns
    -------
    dict with keys:
        'weights'         : optimal weight vector
        'expected_return' : portfolio expected return
        'std_dev'         : portfolio standard deviation
        'skewness'        : portfolio skewness
        'kurtosis'        : portfolio excess kurtosis
        'quantile'        : shortfall probability (VaR) or expected shortfall (ES)
        'eligible_count'  : number of eligible weight combinations found
    """
    probs = U[:, -1]
    returns_matrix = U[:, :n_securities]
    l = len(U)

    # -------------------------------------------------------------------------
    # Stage 1: Grid search
    # -------------------------------------------------------------------------
    # Generate weight grid for first (n-1) securities; last weight = 1 - sum
    step = 1.0 / m_prime
    weight_grid = np.arange(step, 1.0, step)

    # Build all combinations of n-1 weights where sum <= 1
    if n_securities == 2:
        combos = [(w1,) for w1 in weight_grid if w1 <= 1.0]
    elif n_securities == 3:
        combos = [(w1, w2) for w1 in weight_grid for w2 in weight_grid
                  if w1 + w2 <= 1.0]
    else:
        combos = [(w1, w2, w3) for w1 in weight_grid for w2 in weight_grid
                  for w3 in weight_grid if w1 + w2 + w3 <= 1.0]

    best_return = -np.inf
    best_weights = None
    eligible_count = 0

    for combo in combos:
        w = np.array(list(combo) + [1.0 - sum(combo)])
        if np.any(w < 0):
            continue

        portfolio_returns = returns_matrix @ w
        expected_r = float(portfolio_returns @ probs)

        # Check mental-account constraint
        eligible = False
        if constraint_type == 'var':
            shortfall_prob = float(probs[portfolio_returns < H].sum())
            if shortfall_prob <= alpha:
                eligible = True
        elif constraint_type == 'es':
            tail_mask = portfolio_returns < H
            if tail_mask.sum() > 0:
                es = float((portfolio_returns[tail_mask] * probs[tail_mask]).sum()
                           / probs[tail_mask].sum())
                if L is not None and es >= L:
                    eligible = True
                elif L is None:
                    eligible = True

        if eligible:
            eligible_count += 1
            if expected_r > best_return:
                best_return = expected_r
                best_weights = w.copy()

    if best_weights is None:
        raise ValueError(
            "No eligible portfolios found. Try relaxing H, alpha, or increasing m_prime."
        )

    # -------------------------------------------------------------------------
    # Stage 2: Gradient optimisation (scipy, COBYLA — matches thesis nloptr COBYLA)
    # -------------------------------------------------------------------------
    # Objective: maximise E[r] - penalty * (alpha - P(r < H))^2
    # We minimise the negative (scipy minimises)

    def objective(wei):
        w_full = np.append(wei, 1.0 - wei.sum())
        port_ret = returns_matrix @ w_full
        # Expected return
        term1 = float(port_ret @ probs)
        # Constraint penalty
        shortfall = float(probs[port_ret < H].sum())
        term2 = penalty * (alpha - shortfall) ** 2
        return -term1 + term2  # minimise this = maximise expected return

    x0 = best_weights[:-1]
    bounds = [(0.0, 1.0)] * (n_securities - 1)

    result = minimize(
        objective,
        x0=x0,
        method='COBYLA',
        bounds=bounds,
        options={'rhobeg': 0.01, 'maxiter': 10000, 'catol': 1e-8}
    )

    w_opt = np.append(result.x, 1.0 - result.x.sum())
    w_opt = np.clip(w_opt, 0, 1)

    # -------------------------------------------------------------------------
    # Portfolio statistics
    # -------------------------------------------------------------------------
    port_ret = returns_matrix @ w_opt
    mean_r = float(port_ret @ probs)
    variance = float(((port_ret - mean_r) ** 2) @ probs)
    std_dev = np.sqrt(variance)
    skewness = float(((port_ret - mean_r) ** 3) @ probs) / (std_dev ** 3)
    kurtosis = float(((port_ret - mean_r) ** 4) @ probs) / (std_dev ** 4) - 3

    if constraint_type == 'var':
        quantile_stat = float(probs[port_ret < H].sum())
    else:
        tail_mask = port_ret < H
        if tail_mask.sum() > 0:
            quantile_stat = float(
                (port_ret[tail_mask] * probs[tail_mask]).sum() / probs[tail_mask].sum()
            )
        else:
            quantile_stat = 0.0

    return {
        'weights': w_opt,
        'expected_return': mean_r,
        'std_dev': std_dev,
        'skewness': skewness,
        'excess_kurtosis': kurtosis,
        'shortfall_stat': quantile_stat,
        'eligible_count': eligible_count,
    }


# =============================================================================
# CONVENIENCE WRAPPER — matches the thesis base case exactly
# =============================================================================

def run_thesis_base_case(derivative_type='put', H=-0.10, alpha=0.05, m=51, m_prime=20):
    """
    Replicates the base example from Das & Statman used in the thesis.

    3 primary securities + 1 derivative on the highest-risk security (Sec3).

    Parameters
    ----------
    derivative_type : str
        One of: 'put', 'call', 'safety_collar', 'straddle', 'strangle',
                'cgn', 'barrier_m', 'none'
    H : float      Mental-account threshold (default: -10%)
    alpha : float  Max shortfall probability (default: 5%)
    m : int        Return-space grid steps (thesis: 51; reduce for speed)
    m_prime : int  Weight-space grid steps (thesis: 99; reduce for speed)

    Returns
    -------
    dict  Optimization results
    """
    means = [0.05, 0.10, 0.25]
    sigs = [0.05, 0.20, 0.50]
    covariances = [0.0025, 0, 0,
                   0,     0.04, 0.02,
                   0,     0.02, 0.25]
    cov_matrix = np.array(covariances).reshape(3, 3)

    # Derivative configs matching thesis parameters
    derivative_configs = {
        'put':              {'type': 'put',              'underlying_index': 2, 'vol': sigs[2], 'S0': 1, 'r': 0.03, 'T': 1, 'strike': 1.4},
        'call':             {'type': 'call',             'underlying_index': 2, 'vol': sigs[2], 'S0': 1, 'r': 0.03, 'T': 1, 'strike': 1.2},
        'safety_collar':    {'type': 'safety_collar',    'underlying_index': 2, 'vol': sigs[2], 'S0': 1, 'r': 0.03, 'T': 1, 'strike_c': 1.6, 'strike_p': 1.2},
        'aggressive_collar':{'type': 'aggressive_collar','underlying_index': 2, 'vol': sigs[2], 'S0': 1, 'r': 0.03, 'T': 1, 'strike_c': 1.6, 'strike_p': 1.2},
        'straddle':         {'type': 'straddle',         'underlying_index': 2, 'vol': sigs[2], 'S0': 1, 'r': 0.03, 'T': 1, 'strike': 0.7},
        'strangle':         {'type': 'strangle',         'underlying_index': 2, 'vol': sigs[2], 'S0': 1, 'r': 0.03, 'T': 1, 'strike_kp': 0.8, 'strike_kc': 0.9},
        'cgn':              {'type': 'cgn',              'underlying_index': 2, 'vol': sigs[2], 'S0': 1, 'r': 0.03, 'T': 1, 'floor': 0.01, 'participation': 1.0},
        'barrier_m':        {'type': 'barrier_m',        'underlying_index': 2, 'vol': sigs[2], 'S0': 1, 'r': 0.03, 'T': 1, 'M': 0.40, 'premium_bm': 0.10},
        'none':             None,
    }

    der_config = derivative_configs.get(derivative_type)
    print(f"\nRunning: {derivative_type.upper()} | H={H:.0%} | alpha={alpha:.0%} | m={m} | m'={m_prime}")
    print("Step 1: Building state space...")
    U, dr_values = build_state_space(means, sigs, m=m, derivative_config=der_config)
    n_sec = U.shape[1] - 1

    print("Step 2: Assigning probabilities via Gaussian copula...")
    U = assign_probabilities(U, means, sigs, cov_matrix, dr_values)

    print(f"Step 3: Optimising over {n_sec} securities ({len(U):,} states)...")
    result = optimize_portfolio(U, n_sec, constraint_type='var', H=H, alpha=alpha, m_prime=m_prime)

    print("\n--- Results ---")
    labels = ['Sec1 (low-risk)', 'Sec2 (mid-risk)', 'Sec3 (high-risk)', 'Derivative']
    for i, w in enumerate(result['weights']):
        label = labels[i] if i < len(labels) else f'Asset {i+1}'
        print(f"  {label}: {w:.4f} ({w*100:.1f}%)")
    print(f"  Expected return:   {result['expected_return']:.4f} ({result['expected_return']*100:.2f}%)")
    print(f"  Std deviation:     {result['std_dev']:.4f}")
    print(f"  Skewness:          {result['skewness']:.4f}")
    print(f"  Excess kurtosis:   {result['excess_kurtosis']:.4f}")
    print(f"  Shortfall prob:    {result['shortfall_stat']:.4f} (constraint: <= {alpha:.2f})")
    print(f"  Eligible combos:   {result['eligible_count']:,}")

    return result


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Quick demo with m=21 and m_prime=15 for speed
    # Increase to m=51, m_prime=99 for thesis-level accuracy (takes several minutes)
    result = run_thesis_base_case(
        derivative_type='put',
        H=-0.10,
        alpha=0.05,
        m=21,
        m_prime=15
    )
