"""  # v4
Behavioral Portfolio Optimizer — Streamlit Dashboard
Full version with: live market data, manual input, CSV upload,
custom structured product composer, and extended optimizer (5+ securities).
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import anthropic
from scipy.optimize import minimize
from io import StringIO
from datetime import date, timedelta
from behavioral_portfolio_optimizer import (
    build_state_space, assign_probabilities, optimize_portfolio,
    compute_structured_payoff, bs_call, bs_put
)
from scipy.stats import norm as _norm
from scipy.optimize import brentq as _brentq

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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beyond Mean-Variance Portfolio Optimiser",
    page_icon="📈", layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
.main{background:#0d1117}.block-container{padding-top:1.5rem}
h1{color:#fff;font-size:1.6rem}h2,h3{color:#c0c8d8}
.info-box{background:#1a1a2e;border:1px solid #4a9eff;border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem;color:#ffffff !important}
.warn-box{background:#1a1200;border:1px solid #f59e0b;border-radius:6px;padding:.5rem 1rem;color:#f59e0b;font-size:.82rem;margin-top:.3rem}
.ok-box{background:#001a0f;border:1px solid #10b981;border-radius:6px;padding:.5rem 1rem;color:#10b981;font-size:.82rem;margin-top:.3rem}
.section-header{border-left:4px solid #4a9eff;background:#1a1a2e;padding:.4rem .8rem;border-radius:0 6px 6px 0;margin-bottom:.5rem;color:#ffffff;font-weight:600;font-size:1.05rem;letter-spacing:.02em}
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
# ── Static explanations dictionary (no API cost) ─────────────────────────────
EXPLANATIONS = {
    # ── Derivatives ───────────────────────────────────────────────────────────
    "Put option": (
        "A put option gives the holder the right to sell the underlying asset at a fixed strike price. "
        "Its payoff is max(K - S_T, 0), increasing in value when the underlying falls below the strike. "
        "In a behavioural portfolio, a long put acts as downside insurance — it reduces the probability "
        "of breaching the mental-account threshold H, allowing the optimizer to allocate more to "
        "high-return risky assets while satisfying the shortfall constraint."
    ),
    "Call option": (
        "A call option gives the holder the right to buy the underlying asset at a fixed strike price. "
        "Its payoff is max(S_T - K, 0), increasing in value when the underlying rises above the strike. "
        "In a behavioural portfolio, a long call provides leveraged upside participation with limited "
        "downside. It adds positive skewness to the return distribution, which can raise expected return "
        "for a given mental-account constraint level."
    ),
    "Safety collar (long put + short call)": (
        "A safety collar combines a long put (downside protection) and a short call (capping upside). "
        "The short call premium offsets part of the put cost, making the hedge cheaper. "
        "The result is a return profile bounded on both sides: losses are limited below the put strike, "
        "but gains are also capped above the call strike. "
        "Useful when the investor wants cheap downside protection and is willing to sacrifice extreme upside."
    ),
    "Aggressive collar (long call + short put)": (
        "An aggressive collar combines a long call (upside participation) and a short put (accepting downside risk). "
        "The short put premium finances the call, making upside exposure cheaper. "
        "Unlike the safety collar, this structure increases rather than reduces downside risk — "
        "it suits investors with a strong upward view who are comfortable bearing more tail risk "
        "in exchange for leveraged upside."
    ),
    "Straddle (long call + long put)": (
        "A straddle combines a long call and a long put at the same strike, profiting when the "
        "underlying moves significantly in either direction. "
        "It is a bet on high volatility regardless of direction. "
        "In a behavioural portfolio context, a straddle adds fat tails and positive excess kurtosis "
        "to the return distribution — it performs well in extreme market moves and can help "
        "satisfy mental-account constraints when large moves are expected."
    ),
    "Strangle (long call + long put, diff strikes)": (
        "A strangle is similar to a straddle but uses different strikes for the call and put, "
        "making it cheaper since both options are out-of-the-money. "
        "It profits from large moves in either direction but requires a bigger move than a straddle to break even. "
        "In a behavioural portfolio, it provides asymmetric tail protection at lower cost than a straddle, "
        "useful when extreme but not moderate moves are expected."
    ),
    "Capital-guaranteed note — uncapped": (
        "A capital-guaranteed note (CGN) is a structured product that guarantees return of capital "
        "(plus a floor F) at maturity, while providing participation in the upside of an underlying asset. "
        "The uncapped version has no ceiling on the upside participation. "
        "In a behavioural portfolio it is extremely powerful: the capital guarantee directly satisfies "
        "the mental-account downside constraint, freeing the optimizer to allocate heavily to the CGN "
        "and achieve significantly higher expected returns than a portfolio of primary securities alone."
    ),
    "Capital-guaranteed note — capped": (
        "A capped CGN is identical to the uncapped version but limits upside participation above a cap level. "
        "The cap reduces the cost of the product (the issuer saves on the call spread), making it cheaper "
        "than the uncapped version. "
        "The trade-off is sacrificed upside beyond the cap. "
        "In a behavioural portfolio, it still provides strong downside protection but produces lower "
        "expected returns than the uncapped version when the underlying performs very strongly."
    ),
    "Barrier-M note": (
        "A barrier-M note pays the absolute value of the underlying return when that return stays "
        "within a corridor [-M, +M], and pays zero outside it. "
        "It profits from moderate moves in either direction but loses value in extreme moves — "
        "the opposite of a straddle. "
        "In a behavioural portfolio it is useful when low-volatility environments are expected, "
        "providing income from small fluctuations while the mental-account constraint limits tail risk."
    ),
    # ── Risk measures ─────────────────────────────────────────────────────────
    "Value at Risk (VaR)": (
        "Value at Risk (VaR) at level α is the return threshold H such that losses exceed H with "
        "probability at most α. For example, a 5% VaR of -10% means there is a 5% chance of "
        "losing more than 10%. "
        "In this app, the VaR constraint is the mental-account threshold: "
        "P(portfolio return < H) ≤ α. The optimizer finds the highest expected return portfolio "
        "satisfying this constraint."
    ),
    "Expected Shortfall (ES)": (
        "Expected Shortfall (ES), also called Conditional VaR or CVaR, measures the average loss "
        "in the worst α% of scenarios. Unlike VaR which only gives a threshold, ES captures the "
        "severity of losses beyond that threshold. "
        "In this app, the ES constraint requires E[return | return < H] ≥ L — "
        "the average loss in the tail must not be worse than L. "
        "ES is considered a more complete risk measure than VaR as it is coherent and convex."
    ),
    "Shortfall probability": (
        "The shortfall probability is P(portfolio return < H) — the probability that the portfolio "
        "return falls below the mental-account threshold H. "
        "It is the key output of the VaR constraint mode. "
        "The optimizer ensures this probability stays at or below α. "
        "A result of 4.4% with α=5% means the constraint is satisfied with 0.6% margin."
    ),
    "Skewness": (
        "Skewness measures the asymmetry of a return distribution. "
        "Positive skewness means occasional large gains and frequent small losses — "
        "preferred by investors. Negative skewness means occasional large losses. "
        "Derivatives like calls and CGNs add positive skewness to portfolio returns, "
        "which is one reason behavioral portfolios including them can outperform "
        "mean-variance portfolios that ignore higher moments."
    ),
    "Excess kurtosis": (
        "Excess kurtosis measures the fatness of the tails of a return distribution relative to "
        "a normal distribution. Positive excess kurtosis (leptokurtosis) means fatter tails — "
        "more extreme events than a normal distribution would predict. "
        "This is why mean-variance theory, which assumes normality, is insufficient for portfolios "
        "containing derivatives: options have highly non-normal payoff distributions."
    ),
    # ── Portfolio theory ──────────────────────────────────────────────────────
    "Mean-variance efficient frontier": (
        "The mean-variance efficient frontier, introduced by Markowitz (1952), is the set of portfolios "
        "that maximise expected return for a given level of variance (risk). "
        "It is the foundation of modern portfolio theory. "
        "However it assumes normally distributed returns and ignores higher moments — "
        "making it inadequate for portfolios containing derivatives. "
        "This app shows the MV frontier alongside behavioral frontiers to illustrate this limitation."
    ),
    "Markowitz optimization": (
        "Markowitz optimization solves: max w'μ - (λ/2) w'Σw subject to sum(w)=1, w≥0. "
        "The parameter λ is the risk-aversion coefficient. Higher λ penalises variance more, "
        "producing lower-risk portfolios. "
        "A key result in this app is that for any mental-account constraint (H, α), "
        "there exists an implied λ such that Markowitz and behavioral optimization yield "
        "identical portfolios — when no derivatives are present."
    ),
    "Mental accounting": (
        "Mental accounting, developed by Richard Thaler, is the tendency of individuals to "
        "categorise and evaluate financial outcomes in separate mental 'accounts' rather than "
        "as a unified portfolio. "
        "In portfolio theory, Das & Statman (2009) formalise this as a downside constraint: "
        "investors set a threshold H and maximum acceptable probability α of breaching it. "
        "This framework naturally accommodates derivatives whose asymmetric payoffs "
        "provide targeted protection for specific mental accounts."
    ),
    "Behavioral portfolio theory": (
        "Behavioral portfolio theory (BPT), developed by Shefrin & Statman (2000) and extended "
        "by Das & Statman (2009), integrates psychological insights into portfolio construction. "
        "Rather than maximising a utility function over total wealth, investors set safety-first "
        "constraints (mental accounts) and maximise expected return subject to them. "
        "BPT explains observed investor behaviour such as holding both lottery tickets and "
        "insurance, and provides a framework for including derivatives in optimal portfolios."
    ),
    "MVT/MAT equivalence": (
        "The MVT/MAT equivalence, proven in Das, Markowitz, Scheid & Statman (2010), shows that "
        "mean-variance theory (MVT) and mental-accounting theory (MAT) are equivalent "
        "when no derivatives are present. "
        "For any threshold H and shortfall probability α, there exists a unique implied "
        "risk-aversion coefficient λ such that both methods produce the same optimal portfolio. "
        "This equivalence breaks down when derivatives are added — the behavioral approach "
        "can then exploit asymmetric payoffs that mean-variance cannot capture."
    ),
    "Implied risk aversion lambda": (
        "The implied risk-aversion coefficient λ is the value such that the Markowitz optimal "
        "portfolio (maximising w'μ - (λ/2)w'Σw) is identical to the behavioral optimal portfolio "
        "under the mental-account constraint (H, α). "
        "This app computes λ dynamically as you adjust H and α in the sidebar. "
        "At H=-10% and α=5%, λ=3.795 for the default base case. "
        "Higher α (more risk tolerance) implies lower λ; tighter H implies higher λ."
    ),
    "Gaussian copula": (
        "A Gaussian copula models the dependence structure between assets independently of their "
        "marginal distributions. It maps each asset's returns through their individual CDFs to "
        "uniform scores, then models their joint dependence using a multivariate normal distribution. "
        "This allows non-normal marginal distributions (as produced by options) while still "
        "capturing realistic correlations between assets. "
        "This app uses a Gaussian copula in Step 2 to assign probabilities to the state space."
    ),
    "Black-Scholes pricing": (
        "The Black-Scholes model prices European options under assumptions of log-normal asset "
        "prices, constant volatility, and no arbitrage. "
        "The formula gives call price = S·N(d1) - K·e^(-rT)·N(d2) where d1 and d2 depend on "
        "spot price, strike, volatility, risk-free rate, and time to maturity. "
        "This app uses Black-Scholes to compute derivative payoffs in each scenario of the "
        "state space, enabling the optimizer to price the derivative's contribution to portfolio risk and return."
    ),
    # ── Academic references ───────────────────────────────────────────────────
    "Das & Statman (2009) — Beyond Mean-Variance": (
        "Das, Sanjiv and Meir Statman (2009) — 'Beyond Mean-Variance: Portfolios with Derivatives "
        "and Non-Normal Returns in Mental Accounts'. "
        "This paper introduces the core algorithm used in this app. It shows how to construct "
        "optimal portfolios including derivatives under a mental-accounting downside constraint, "
        "using a discrete state space with Gaussian copula probabilities and a grid search optimizer. "
        "It demonstrates that derivatives — especially capital-guaranteed notes — can substantially "
        "improve portfolio expected returns while satisfying the same downside constraint."
    ),
    "Das, Markowitz, Scheid & Statman (2010) JFQA": (
        "Das, Sanjiv, Harry Markowitz, Jonathan Scheid and Meir Statman (2010) — "
        "'Portfolio Optimization with Mental Accounts', Journal of Financial and Quantitative Analysis, "
        "Vol. 45, No. 2, pp. 311-334. "
        "This paper proves the MVT/MAT equivalence: for any mental-account constraint (H, α), "
        "there exists an implied risk-aversion λ such that Markowitz mean-variance optimization "
        "and behavioral optimization produce identical portfolios when no derivatives are present. "
        "This is the theoretical foundation for the equivalence point shown on the frontier chart."
    ),
    "Jeddou (2012) MSc thesis USI Lugano": (
        "Jeddou, Sami (2012) — 'Beyond Mean-Variance: Options and Structured Products in "
        "Behavioral Portfolios', MSc Finance Thesis, Università della Svizzera italiana (USI Lugano), "
        "supervised by Prof. Enrico De Giorgi. "
        "This thesis implements the full Das & Statman (2009) algorithm in R and extends it to "
        "all major derivative and structured product types: puts, calls, safety and aggressive collars, "
        "straddles, strangles, capital-guaranteed notes (capped and uncapped), and barrier-M notes. "
        "This app is a Python reimplementation and extension of that work."
    ),
}

def get_explanation(term):
    """Look up explanation from static dictionary. No API call."""
    return EXPLANATIONS.get(term,
        f"No pre-written explanation available for '{term}'. "
        "Please use the custom question box below for AI-generated answers.")

def get_ai_chat_response(question, portfolio_context=""):
    """Get AI response for custom questions via Anthropic API."""
    try:
        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        system = (
            "You are a financial expert assistant embedded in a behavioral portfolio optimization app. "
            "Give clear, concise answers in 3-5 sentences. Focus on portfolio optimization context.")
        paper_ctx = (
            "The app is based on: Das & Statman (2009) Beyond Mean-Variance; "
            "Das, Markowitz, Scheid & Statman (2010) JFQA MVT/MAT equivalence; "
            "Jeddou (2012) MSc thesis USI Lugano.")
        prompt = f"{paper_ctx}\n{f'Portfolio context: {portfolio_context}' if portfolio_context else ''}\n\nQuestion: {question}"
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception:
        return ("AI response unavailable — the custom question feature requires an Anthropic API key. "
                "Please check the pre-written explanations above for common terms.")


DEFAULT_MEANS = [0.05, 0.10, 0.25]
DEFAULT_SIGS  = [0.05, 0.20, 0.50]
DEFAULT_CORR  = [[1.0,0.0,0.0],[0.0,1.0,0.4],[0.0,0.4,1.0]]
DEFAULT_NAMES = ["Sec 1 — Low risk","Sec 2 — Mid risk","Sec 3 — High risk"]

GRID_OPTIONS = {
    "⚡ Fast (m=21, m'=15)":           (21,15),
    "⚖️  Standard (m=35, m'=50)":      (35,50),
    "🎯 High precision (m=51, m'=99)": (51,99),
}

GRID_EXPLANATIONS = {
    "⚡ Fast (m=21, m'=15)": (
        "Uses a coarse grid of 21 return scenarios per security and 15 weight steps per dimension. "
        "Runs in ~10-20 seconds. Results are directionally correct and useful for exploring "
        "parameters, but weights and expected returns may differ from the precise solution by "
        "a few percentage points. Recommended for initial exploration and parameter sensitivity testing."
    ),
    "⚖️  Standard (m=35, m'=50)": (
        "Uses a medium grid with 35 return scenarios and 50 weight steps per dimension. "
        "Runs in ~1-2 minutes. Provides a good balance between speed and accuracy — "
        "results are close to the precise solution in most cases. "
        "Recommended for most use cases once you have identified the right parameters."
    ),
    "🎯 High precision (m=51, m'=99)": (
        "Matches the original thesis parameters exactly — 51 return scenarios and 99 weight steps, "
        "the same values used in Das & Statman (2009) and Jeddou (2012). "
        "Results are publication-quality and directly comparable to academic benchmarks. "
        "May take 5-10 minutes depending on the number of securities. "
        "Recommended for final results and for verifying the equivalence point."
    ),
}

CONSTRAINT_EXPLANATIONS = {
    "var": (
        "The **Value at Risk (VaR) constraint** requires that the probability of the portfolio "
        "return falling below the threshold H does not exceed α. "
        "Formally: P(return < H) ≤ α. "
        "For example, with H = -10% and α = 5%, the optimizer finds the highest expected return "
        "portfolio where there is at most a 5% chance of losing more than 10%. "
        "A key theoretical result: this constraint is equivalent to a Markowitz portfolio with "
        "an implied risk-aversion coefficient λ — shown dynamically below the sliders."
    ),
    "es": (
        "The **Expected Shortfall (ES) constraint** — also called Conditional VaR (CVaR) — "
        "requires that the average portfolio return in the worst scenarios (those where return "
        "falls below H) is at least L. "
        "Formally: E[return | return < H] ≥ L. "
        "ES captures the severity of losses beyond the threshold, not just their probability, "
        "making it a more complete risk measure than VaR. "
        "It is a coherent risk measure and is preferred by regulators under Basel III/IV. "
        "For example, with H = -10% and L = -15%, the optimizer ensures that when losses exceed "
        "10%, their average is no worse than 15%."
    ),
}

PREDEFINED_DERIVATIVES = {
    "None — primary securities only":               None,
    "Put option":                                    "put",
    "Call option":                                   "call",
    "Safety collar (long put + short call)":         "safety_collar",
    "Aggressive collar (long call + short put)":     "aggressive_collar",
    "Straddle (long call + long put)":               "straddle",
    "Strangle (long call + long put, diff strikes)": "strangle",
    "Capital-guaranteed note — uncapped":            "cgn_uncapped",
    "Capital-guaranteed note — capped":              "cgn_capped",
    "Barrier-M note":                                "barrier_m",
    "🔧 Custom structured product":                  "custom",
}

COMPONENT_TYPES = [
    "long_call","short_call","long_put","short_put",
    "long_digital_call","short_digital_call",
    "long_digital_put","short_digital_put","zcb"
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def corr_to_cov(sigs, corr):
    s = np.array(sigs); c = np.array(corr)
    return np.outer(s,s)*c

def clean_returns(rets, outlier_threshold=5.0):
    """
    Clean a returns DataFrame:
    1. Remove rows where ALL returns are exactly zero (stale prices)
    2. Winsorise outliers beyond +/- outlier_threshold standard deviations
    3. Return cleaned returns and a cleaning report dict
    """
    report = {}
    n_before = len(rets)

    # Step 1: remove all-zero rows (stale prices)
    all_zero_mask = (rets.abs() < 1e-10).all(axis=1)
    n_stale = all_zero_mask.sum()
    rets = rets[~all_zero_mask]
    if n_stale > 0:
        report['stale_rows_removed'] = int(n_stale)

    # Step 2: winsorise outliers per column
    n_outliers = 0
    for col in rets.columns:
        mean = rets[col].mean()
        std  = rets[col].std()
        if std > 0:
            lo = mean - outlier_threshold * std
            hi = mean + outlier_threshold * std
            mask = (rets[col] < lo) | (rets[col] > hi)
            n_col = mask.sum()
            if n_col > 0:
                rets[col] = rets[col].clip(lo, hi)
                n_outliers += n_col
    if n_outliers > 0:
        report['outliers_winsorised'] = int(n_outliers)

    # Step 3: minimum data warning
    n_after = len(rets)
    report['observations'] = n_after
    if n_after < 60:
        report['warning'] = f'Only {n_after} observations after cleaning — results may be unreliable. Consider a longer date range.'
    elif n_after < 252:
        report['warning'] = f'{n_after} observations — less than 1 year of data. Consider extending the date range for more reliable estimates.'

    report['removed_total'] = n_before - n_after
    return rets, report

def parse_csv(f):
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    rets = df.pct_change().dropna()
    rets, _ = clean_returns(rets)
    return rets.mean().tolist(), rets.std().tolist(), rets.corr().values.tolist(), list(rets.columns)

def fetch_tickers(tickers, start, end, freq):
    try:
        import yfinance as yf
        raw = yf.download(tickers, start=str(start), end=str(end),
                          auto_adjust=True, progress=False)['Close']
        if isinstance(raw, pd.Series): raw = raw.to_frame(tickers[0])
        raw = raw.dropna()
        if freq == "Monthly":
            raw = raw.resample('ME').last()
        rets = raw.pct_change().dropna()
        rets, cleaning_report = clean_returns(rets.copy())
        factor = 252 if freq == "Daily" else 12
        means = (rets.mean() * factor).tolist()
        sigs  = (rets.std() * np.sqrt(factor)).tolist()
        corr  = rets.corr().values.tolist()
        names = list(rets.columns)
        last_prices = raw.iloc[-1].to_dict()
        return means, sigs, corr, names, last_prices, None, cleaning_report
    except Exception as e:
        return None, None, None, None, None, str(e), {}

def build_der_config(der_type, der_params, sigs, underlying_idx):
    base = {"underlying_index": underlying_idx,
            "vol": sigs[underlying_idx], "S0":1.0, "r":0.03, "T":1.0}
    if der_type == "put":
        return {**base, "type":"put", "strike":der_params["strike"]}
    elif der_type == "call":
        return {**base, "type":"call", "strike":der_params["strike"]}
    elif der_type == "straddle":
        return {**base, "type":"straddle", "strike":der_params["strike"]}
    elif der_type == "safety_collar":
        return {**base, "type":"safety_collar",
                "strike_p":der_params["strike_p"],"strike_c":der_params["strike_c"]}
    elif der_type == "aggressive_collar":
        return {**base, "type":"aggressive_collar",
                "strike_p":der_params["strike_p"],"strike_c":der_params["strike_c"]}
    elif der_type == "strangle":
        return {**base, "type":"strangle",
                "strike_kp":der_params["strike_kp"],"strike_kc":der_params["strike_kc"]}
    elif der_type == "cgn_uncapped":
        return {**base, "type":"cgn","floor":der_params["floor"],
                "participation":der_params["participation"],
                "cap":None,"cgn_premium":der_params["premium"]}
    elif der_type == "cgn_capped":
        return {**base, "type":"cgn","floor":der_params["floor"],
                "participation":der_params["participation"],
                "cap":der_params["cap"],"cgn_premium":der_params["premium"]}
    elif der_type == "barrier_m":
        return {**base, "type":"barrier_m",
                "M":der_params["M"],"premium_bm":der_params["premium_bm"]}
    elif der_type == "custom":
        return {**base, "type":"custom","components":der_params["components"]}
    return None

@st.cache_data
def compute_mv_frontier(means_t, cov_t):
    means = np.array(means_t); cov = np.array(cov_t); n = len(means)
    def mv_opt(lam):
        def obj(w): return -(w@means-(lam/2)*(w@cov@w))
        cons=[{"type":"eq","fun":lambda w:w.sum()-1}]; bounds=[(0,1)]*n
        best=None
        for x0 in [np.ones(n)/n]+[np.eye(n)[i]*0.6+np.ones(n)*0.4/n for i in range(n)]:
            r=minimize(obj,x0,method="SLSQP",bounds=bounds,constraints=cons)
            if r.success and (best is None or r.fun<best.fun): best=r
        if best is None: return None
        w=best.x
        return float(np.sqrt(w@cov@w))*100, float(w@means)*100
    pts=[mv_opt(l) for l in np.concatenate([np.linspace(0.5,3,40), np.linspace(3,25,60), np.linspace(25,200,40)])]
    pts=[p for p in pts if p]
    # Sort by std dev ascending so line draws left to right
    pts=sorted(set(pts), key=lambda p: p[0])
    eq=mv_opt(3.7950)
    return [p[0] for p in pts],[p[1] for p in pts],eq

def run_opt(means,sigs,cov,der_config,H,alpha,m,mp,
            constraint_type='var',L=None):
    U,dr=build_state_space(means,sigs,m=m,derivative_config=der_config)
    U=assign_probabilities(U,means,sigs,cov,dr)
    n=U.shape[1]-1
    res=optimize_portfolio(U,n,H=H,alpha=alpha if alpha is not None else 0.05,
                           m_prime=mp,constraint_type=constraint_type,L=L)
    return res,n

def build_frontier(means,sigs,cov,der_config,alpha,m,mp,
                   constraint_type='var',L=None):
    H_vals=[-0.05,-0.08,-0.10,-0.12,-0.15,-0.18,-0.20]
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

def plot_frontier_plotly(mv_x, mv_y, mv_eq,
                         nd_x, nd_y, nd_lbls,
                         der_x, der_y, der_lbls,
                         der_label, H_sel, alpha):
    """Interactive Plotly version of the frontier chart with hover tooltips."""
    fig = go.Figure()

    # ── Mean-variance frontier ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=mv_x, y=mv_y, mode='lines',
        name='Mean-variance efficient frontier (Markowitz)',
        line=dict(color='#6b7280', width=2, dash='dash'),
        hovertemplate='<b>Mean-Variance Efficient Frontier (Markowitz)</b><br>Std Dev: %{x:.2f}%<br>Expected Return: %{y:.2f}%<extra></extra>'
    ))

    # ── Behavioral — no derivative ────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=nd_x, y=nd_y, mode='lines+markers',
        name='Behavioural — no derivative',
        line=dict(color='#4a9eff', width=2.5),
        marker=dict(size=9, color='#4a9eff', symbol='circle'),
        text=nd_lbls,
        hovertemplate='<b>Behavioral (no derivative)</b><br>Threshold: %{text}<br>Std Dev: %{x:.2f}%<br>Expected Return: %{y:.2f}%<extra></extra>'
    ))

    # ── Behavioral — with derivative ──────────────────────────────────────────
    if der_x:
        fig.add_trace(go.Scatter(
            x=der_x, y=der_y, mode='markers',
            name=f'Behavioural — {der_label}',
            marker=dict(size=10, color='#f59e0b', symbol='square'),
            text=der_lbls,
            hovertemplate=f'<b>Behavioural ({der_label})</b><br>Threshold: %{{text}}<br>Std Dev: %{{x:.2f}}%<br>Expected Return: %{{y:.2f}}%<extra></extra>'
        ))

        # Gain arrow at selected H
        try:
            i0 = nd_lbls.index(f'H={H_sel:.0%}')
            i1 = der_lbls.index(f'H={H_sel:.0%}')
            x0, y0 = nd_x[i0], nd_y[i0]
            x1, y1 = der_x[i1], der_y[i1]
            gain = y1 - y0
            fig.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1.2,
                arrowwidth=2, arrowcolor='#ffffff',
                text=f'+{gain:.1f} pp<br>same constraint',
                font=dict(color='#ffffff', size=10),
                bgcolor='rgba(13,17,23,0.8)',
                bordercolor='#ffffff', borderwidth=1,
                xanchor='left'
            )
        except (ValueError, IndexError):
            pass

    # ── Equivalence point ─────────────────────────────────────────────────────
    if mv_eq:
        fig.add_trace(go.Scatter(
            x=[mv_eq[0]], y=[mv_eq[1]], mode='markers',
            name='Equivalence point — MV = Behavioral (λ=3.795, H=-10%, α=5%)',
            marker=dict(size=13, color='#10b981', symbol='diamond',
                        line=dict(width=0)),
            showlegend=True,
            legendrank=4,
            hovertemplate='<b>Equivalence point</b><br>At this point, the mean-variance optimal portfolio<br>and the behavioral optimal portfolio are identical.<br>λ=3.795 ↔ H=-10%, α=5%<br>Std Dev: %{x:.2f}%<br>Expected Return: %{y:.2f}%<extra></extra>'
        ))
        fig.add_annotation(
            x=mv_eq[0], y=mv_eq[1],
            text=f'Equivalence point<br>λ=3.795 ↔ H=-10%, α=5%<br>MV = Behavioral = {mv_eq[1]:.1f}%',
            showarrow=True, arrowhead=2, arrowcolor='#10b981',
            arrowwidth=1.5, ax=40, ay=60,
            font=dict(color='#10b981', size=9),
            bgcolor='rgba(13,17,23,0.9)',
            bordercolor='#10b981', borderwidth=1
        )

    # ── MVT/MAT note ──────────────────────────────────────────────────────────
    fig.add_annotation(
        xref='paper', yref='paper', x=0.5, y=1.0,
        text='MV and behavioral frontiers converge without derivatives (MVT = Mean-Variance Theory / MAT = Mental Accounts Theory)',
        showarrow=False,
        font=dict(color='#6b7280', size=10, style='italic'),
        bgcolor='rgba(13,17,23,0.85)',
        bordercolor='#3a3a5a', borderwidth=1,
        xanchor='center', yanchor='bottom'
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        title=dict(
            text='Mean-Variance vs Behavioral Portfolio Efficient Frontier',
            font=dict(color='white', size=15),
            x=0.5,
            xanchor='center',
            xref='paper'
        ),
        xaxis=dict(
            title='Portfolio Risk — Standard Deviation (%)',
            gridcolor='#1e2130', gridwidth=0.5,
            color='#c0c8d8', zerolinecolor='#2a2a3a',
            range=[max(0, min(mv_x) - 1),
                   max(max(mv_x), max(der_x) if der_x else 0) * 1.06]
        ),
        yaxis=dict(
            title='Expected Return (%)',
            gridcolor='#1e2130', gridwidth=0.5,
            color='#c0c8d8', zerolinecolor='#2a2a3a',
            range=[min(min(mv_y), min(nd_y)) - 2,
                   max(max(mv_y), max(der_y) if der_y else 0) * 1.08]
        ),
        legend=dict(
            bgcolor='rgba(26,26,46,0.9)',
            bordercolor='#3a3a5a', borderwidth=1,
            font=dict(color='white', size=10),
            x=0.01, y=0.99
        ),
        hoverlabel=dict(
            bgcolor='#1a1a2e',
            bordercolor='#4a9eff',
            font=dict(color='white', size=11)
        ),
        margin=dict(t=80, b=60, l=60, r=20),
        height=520
    )

    # Footer annotation
    fig.add_annotation(
        xref='paper', yref='paper', x=0.5, y=-0.1,
        text='Das & Statman (2009)  |  Das, Markowitz, Scheid & Statman (2010) JFQA  |  Sami Jeddou, MSc Finance USI Lugano 2012',
        showarrow=False,
        font=dict(color='#4a5568', size=8, style='italic'),
        xanchor='center'
    )

    return fig


def plot_payoff(components, vol, S0, r, T, asset_name):
    returns = np.linspace(-0.8, 1.5, 300)
    payoffs, price0 = compute_structured_payoff(returns, components, vol, S0, r, T)
    fig,ax=plt.subplots(figsize=(8,3.5))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    ax.axhline(0,color="#3a3a5a",linewidth=0.8,linestyle="--")
    ax.axvline(0,color="#3a3a5a",linewidth=0.8,linestyle="--")
    pos=payoffs>=0; neg=payoffs<0
    ax.fill_between(returns*100,payoffs*100,where=pos,color="#10b981",alpha=0.25)
    ax.fill_between(returns*100,payoffs*100,where=neg,color="#ef4444",alpha=0.25)
    ax.plot(returns*100,payoffs*100,color="#f59e0b",linewidth=2)
    ax.set_xlabel(f"Return of {asset_name} (%)",color="#c0c8d8",fontsize=10)
    ax.set_ylabel("Structured product return (%)",color="#c0c8d8",fontsize=10)
    ax.set_title(f"Payoff diagram  |  Fair value = {price0:.4f}",
                 color="white",fontsize=11,fontweight="bold")
    ax.tick_params(colors="#8896a8",labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a3a")
    ax.grid(True,color="#1e2130",linewidth=0.5,linestyle="--",alpha=0.7)
    plt.tight_layout()
    return fig

def plot_frontier(mv_x,mv_y,mv_eq,nd_x,nd_y,nd_lbls,
                  der_x,der_y,der_lbls,der_label,H_sel,alpha):
    fig,ax=plt.subplots(figsize=(11,6.5))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    ax.grid(True,color="#1e2130",linewidth=0.6,linestyle="--",alpha=0.8)
    ax.set_axisbelow(True)
    ax.plot(mv_x,mv_y,color="#6b7280",linewidth=2,linestyle="--",
            label="Mean-variance frontier (Markowitz)",zorder=2,alpha=0.9)
    ax.plot(nd_x,nd_y,color="#4a9eff",linewidth=2.5,marker="o",markersize=7,
            markerfacecolor="#4a9eff",label="Behavioral — no derivative",zorder=3)
    for x,y,l in zip(nd_x,nd_y,nd_lbls):
        ax.annotate(l,xy=(x,y),xytext=(x,y-1.8),
                    color="#7fb3e8",fontsize=7.5,ha="center",zorder=4)
    if der_x:
        ax.scatter(der_x,der_y,color="#f59e0b",s=65,marker="s",zorder=3,
                   label=f"Behavioral — {der_label}")
        for x,y,l in zip(der_x,der_y,der_lbls):
            if l==f"H={H_sel:.0%}":
                ax.annotate(f"{l}, α={alpha:.0%}",xy=(x,y),
                            xytext=(x-8,y+2),color="#f59e0b",fontsize=8,
                            arrowprops=dict(arrowstyle="->",color="#f59e0b",lw=1.2),
                            bbox=dict(boxstyle="round,pad=0.3",facecolor="#0d1117",
                                      edgecolor="#f59e0b",alpha=0.85))
        try:
            i0=nd_lbls.index(f"H={H_sel:.0%}")
            i1=der_lbls.index(f"H={H_sel:.0%}")
            x0,y0=nd_x[i0],nd_y[i0]; x1,y1=der_x[i1],der_y[i1]
            ax.annotate("",xy=(x1,y1),xytext=(x0,y0),
                        arrowprops=dict(arrowstyle="->",color="#ffffff",
                                        lw=1.6,linestyle="dotted"))
            ax.text((x0+x1)/2+1,(y0+y1)/2,
                    f"+{y1-y0:.1f} pp\nsame constraint",
                    color="#e2e8f0",fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3",facecolor="#0d1117",
                              edgecolor="#ffffff",alpha=0.8))
        except (ValueError,IndexError): pass
    if mv_eq:
        ax.scatter(*mv_eq,color="#10b981",s=130,zorder=5,marker="D")
        ax.annotate(f"Equivalence point\nλ=3.795 ↔ H=-10%, α=5%\n={mv_eq[1]:.1f}%",
                    xy=mv_eq,xytext=(mv_eq[0]+3,mv_eq[1]-5),color="#10b981",fontsize=8,
                    arrowprops=dict(arrowstyle="->",color="#10b981",lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.3",facecolor="#0d1117",
                              edgecolor="#10b981",alpha=0.9),zorder=6)
    ax.text(0.5,0.97,
            "MV and behavioral frontiers converge without derivatives\n"
            "(MVT/MAT equivalence — Das, Markowitz, Scheid & Statman 2010)",
            transform=ax.transAxes,color="#6b7280",fontsize=7.5,
            ha="center",va="top",style="italic",
            bbox=dict(boxstyle="round,pad=0.3",facecolor="#0d1117",
                      edgecolor="#3a3a5a",alpha=0.95),zorder=10)
    ax.set_xlabel("Portfolio Risk — Standard Deviation (%)",color="#c0c8d8",fontsize=10,labelpad=6)
    ax.set_ylabel("Expected Return (%)",color="#c0c8d8",fontsize=10,labelpad=6)
    ax.set_title("Mean-Variance vs Behavioral Portfolio Frontier",
                 color="white",fontsize=13,fontweight="bold",pad=12)
    ax.tick_params(colors="#8896a8",labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a3a")
    ax.legend(loc="upper left",fontsize=9,facecolor="#1a1a2e",
              edgecolor="#3a3a5a",labelcolor="white",framealpha=0.9)
    fig.text(0.5,0.001,
             "Das & Statman (2004)  |  Das, Markowitz, Scheid & Statman (2010) JFQA  |  "
             "Sami Jeddou, MSc Finance USI Lugano 2012",
             ha="center",color="#4a5568",fontsize=7,style="italic")
    all_x=mv_x+nd_x+(der_x if der_x else [])
    all_y=mv_y+nd_y+(der_y if der_y else [])
    ax.set_xlim(0,max(all_x)*1.15); ax.set_ylim(min(all_y)-3,max(all_y)+6)
    plt.tight_layout(rect=[0,0.02,1,1])
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown("---")

    # ── 1. Data source ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span style="font-size:1.3rem">①</span> 📂 Portfolio data</div>', unsafe_allow_html=True)
    data_mode = st.radio("Data source",
        ["Default (Das & Statman base case)",
         "Live market data (Yahoo Finance)",
         "Enter manually",
         "Upload CSV"],
        index=0, label_visibility="collapsed")

    means_in=DEFAULT_MEANS[:]; sigs_in=DEFAULT_SIGS[:]
    corr_in=[r[:] for r in DEFAULT_CORR]; names_in=DEFAULT_NAMES[:]
    last_prices={}; data_valid=True

    if data_mode=="Default (Das & Statman base case)":
        st.markdown('<div class="ok-box">✓ Default base case loaded — Means: 5%, 10%, 25% | Std devs: 5%, 20%, 50%</div>',
                    unsafe_allow_html=True)

    elif data_mode=="Live market data (Yahoo Finance)":
        ticker_str=st.text_input("Ticker symbols (comma-separated)",
                                  value="AAPL, MSFT, JPM",
                                  placeholder="e.g. AAPL, MSFT, BNP.PA, GS")
        col1,col2=st.columns(2)
        d_start=col1.date_input("From", value=date(2020,1,1))
        d_end  =col2.date_input("To",   value=date.today()-timedelta(days=1))
        freq   =st.radio("Return frequency",["Daily","Monthly"],horizontal=True)
        fetch_btn=st.button("🔄 Fetch data", use_container_width=True)
        if fetch_btn:
            tickers=[t.strip().upper() for t in ticker_str.split(",") if t.strip()]
            with st.spinner(f"Fetching {len(tickers)} tickers from Yahoo Finance..."):
                m,s,c,n,lp,err,cleaning=fetch_tickers(tickers,d_start,d_end,freq)
            if err:
                st.error(f"Fetch failed: {err}"); data_valid=False
            else:
                st.session_state["live_data"]=(m,s,c,n,lp)
                st.markdown(f'<div class="ok-box">✓ Loaded: {", ".join(n)} '                            f'({cleaning.get("observations","?")} observations after cleaning)</div>',
                            unsafe_allow_html=True)
                if cleaning.get("stale_rows_removed"):
                    st.markdown(f'<div class="warn-box">⚠️ {cleaning["stale_rows_removed"]} stale price rows removed.</div>',
                                unsafe_allow_html=True)
                if cleaning.get("outliers_winsorised"):
                    st.markdown(f'<div class="warn-box">⚠️ {cleaning["outliers_winsorised"]} outlier returns winsorised (±5σ).</div>',
                                unsafe_allow_html=True)
                if cleaning.get("warning"):
                    st.markdown(f'<div class="warn-box">⚠️ {cleaning["warning"]}</div>',
                                unsafe_allow_html=True)
        if "live_data" in st.session_state:
            means_in,sigs_in,corr_in,names_in,last_prices=st.session_state["live_data"]
            factor_label="annualised" if freq=="Daily" else "annualised (monthly)"
            with st.expander("Preview statistics"):
                df_prev=pd.DataFrame({
                    "Asset":names_in,
                    f"Mean ({factor_label})":[f"{m*100:.2f}%" for m in means_in],
                    "Std dev":[f"{s*100:.2f}%" for s in sigs_in]})
                st.dataframe(df_prev,hide_index=True)
                st.markdown("**Correlation matrix**")
                corr_df=pd.DataFrame(corr_in,index=names_in,columns=names_in)
                st.dataframe(corr_df.round(3))
        else:
            data_valid=False

    elif data_mode=="Enter manually":
        n_assets=st.number_input("Number of securities",2,10,3,1)
        names_in,means_in,sigs_in=[],[],[]
        st.markdown("**Returns (annualised)**")
        for i in range(n_assets):
            c1,c2,c3=st.columns([2,1,1])
            nm=c1.text_input(f"Name",value=f"Asset {i+1}",key=f"nm_{i}")
            mn=c2.number_input("Mean%",value=DEFAULT_MEANS[i]*100 if i<3 else 10.0,
                                key=f"mn_{i}",format="%.1f")/100
            sg=c3.number_input("Std%", value=DEFAULT_SIGS[i]*100  if i<3 else 20.0,
                                key=f"sg_{i}",format="%.1f")/100
            names_in.append(nm); means_in.append(mn); sigs_in.append(sg)
        st.markdown("**Correlations**")
        corr_in=[[1.0]*n_assets for _ in range(n_assets)]
        for i in range(n_assets):
            for j in range(i+1,n_assets):
                dv=DEFAULT_CORR[i][j] if i<3 and j<3 else 0.0
                v=st.slider(f"ρ({names_in[i]}, {names_in[j]})",-1.0,1.0,dv,0.05,
                             key=f"cr_{i}_{j}")
                corr_in[i][j]=corr_in[j][i]=v
        cv=corr_to_cov(sigs_in,corr_in)
        if np.any(np.linalg.eigvalsh(cv)<-1e-8):
            st.error("⚠️ Correlation matrix not positive semi-definite."); data_valid=False

    elif data_mode=="Upload CSV":
        st.markdown('<div class="info-box" style="font-size:.82rem">'
                    '📋 <b>Format:</b> First col = dates, remaining cols = asset prices.</div>',
                    unsafe_allow_html=True)
        sample="""Date,Low_Risk,Mid_Risk,High_Risk
2020-01-02,100,100,100
2020-01-03,100.05,100.15,100.40
2020-01-06,100.08,100.30,100.85
2020-01-07,100.12,100.10,101.20
2020-01-08,100.09,100.45,100.60"""
        st.download_button("⬇ Sample CSV",sample,"sample.csv","text/csv")
        uploaded=st.file_uploader("Upload CSV",type=["csv"])
        if uploaded:
            try:
                means_in,sigs_in,corr_in,names_in=parse_csv(uploaded)
                st.markdown(f'<div class="ok-box">✓ {len(means_in)} assets: '
                            f'{", ".join(names_in)}</div>',unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Parse error: {e}"); data_valid=False
        else:
            data_valid=False

    # Method notice
    n_sec_total=len(means_in)
    if n_sec_total>=5:
        st.markdown('<div class="warn-box">⚡ 5+ securities detected — '
                    'differential evolution optimizer will be used automatically.</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. Derivative ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span style="font-size:1.3rem">②</span> 📊 Derivative / Structured product</div>', unsafe_allow_html=True)
    der_label_sel=st.selectbox("Type",list(PREDEFINED_DERIVATIVES.keys()),
                                index=0,label_visibility="collapsed")
    der_type=PREDEFINED_DERIVATIVES[der_label_sel]
    der_params={}

    # AI tooltip for selected derivative
    if der_type is not None and der_type != "custom":
        with st.expander("✨ AI-powered: What is this instrument?", expanded=True):
            explanation = get_explanation(der_label_sel)
            st.markdown(f'<div style="background:#0f1923;border:1px solid #4a9eff;'
                       f'border-radius:6px;padding:.8rem 1rem;color:#c0c8d8;font-size:.85rem">'
                       f'{explanation}</div>',
                       unsafe_allow_html=True)

    # Underlying selector (shown for all non-None derivative types)
    if der_type is not None:
        underlying_idx=st.selectbox(
            "Underlying security",
            options=list(range(len(names_in))),
            format_func=lambda i: names_in[i],
            index=min(len(names_in)-1, 2))
        der_params["underlying_idx"]=underlying_idx

        # Vol auto-filled from data
        auto_vol=sigs_in[underlying_idx]
        vol_override=st.number_input(
            "Volatility (annualised %)",
            value=round(auto_vol*100,1), min_value=1.0, max_value=200.0,
            format="%.1f", step=0.5) / 100
        der_params["vol"]=vol_override

        rf=st.number_input("Risk-free rate (%)",value=3.0,min_value=0.0,
                            max_value=20.0,format="%.1f",step=0.1)/100
        mat=st.slider("Maturity (years)",0.25,3.0,1.0,0.05)
        der_params["r"]=rf; der_params["T"]=mat

    if der_type in ("put","call","straddle"):
        der_params["strike"]=st.slider(
            "Strike (fraction of spot)",0.5,2.0,
            1.4 if der_type=="put" else (1.2 if der_type=="call" else 0.7),0.05)
    elif der_type in ("safety_collar","aggressive_collar"):
        der_params["strike_p"]=st.slider("Put strike",0.5,1.5,1.2,0.05)
        der_params["strike_c"]=st.slider("Call strike",1.0,2.0,1.6,0.05)
    elif der_type=="strangle":
        der_params["strike_kp"]=st.slider("Put strike (Kp)",0.5,1.2,0.8,0.05)
        der_params["strike_kc"]=st.slider("Call strike (Kc)",0.8,1.5,0.9,0.05)
    elif der_type in ("cgn_uncapped","cgn_capped"):
        der_params["floor"]        =st.slider("Floor (%)",0.0,10.0,1.0,0.5)/100
        der_params["participation"]=st.slider("Participation (%)",50,150,100,10)/100
        der_params["premium"]      =st.slider("Premium (%)",0.0,20.0,0.0,1.0)/100
        if der_type=="cgn_capped":
            der_params["cap"]=st.slider("Cap (%)",5.0,50.0,20.0,5.0)/100
    elif der_type=="barrier_m":
        der_params["M"]         =st.slider("Barrier M (%)",10,60,40,5)/100
        der_params["premium_bm"]=st.slider("Premium (%)",0.0,20.0,10.0,1.0)/100

    elif der_type=="custom":
        st.markdown("**Build your structured product**")
        st.markdown("*Add components one by one:*")
        if "components" not in st.session_state:
            st.session_state["components"]=[]

        with st.expander("➕ Add component"):
            ct=st.selectbox("Component type",COMPONENT_TYPES,key="ct_sel")
            if "zcb" not in ct:
                k_val=st.number_input("Strike (fraction of spot)",
                                       0.5,2.0,1.0,0.05,key="k_inp")
            else:
                k_val=1.0
            notional=st.number_input("Notional",0.01,10.0,1.0,0.1,key="n_inp")
            mat_c=st.number_input("Maturity (years)",0.25,3.0,
                                    der_params.get("T",1.0),0.05,key="mc_inp")
            if st.button("Add component"):
                st.session_state["components"].append({
                    "type":ct,"strike":k_val,
                    "notional":notional,"maturity":mat_c})

        if st.session_state["components"]:
            st.markdown("**Current components:**")
            for i,c in enumerate(st.session_state["components"]):
                cols=st.columns([4,1])
                label=f"{c['type']} | K={c['strike']} | N={c['notional']} | T={c['maturity']}y"
                cols[0].markdown(f"`{label}`")
                if cols[1].button("✕",key=f"rm_{i}"):
                    st.session_state["components"].pop(i)
                    st.rerun()
            der_params["components"]=st.session_state["components"]

            # Live payoff diagram
            if len(der_params["components"])>0:
                vol_c=der_params.get("vol",sigs_in[der_params.get("underlying_idx",0)])
                fig_pay=plot_payoff(
                    der_params["components"],vol_c,1.0,
                    der_params.get("r",0.03),der_params.get("T",1.0),
                    names_in[der_params.get("underlying_idx",0)])
                st.pyplot(fig_pay,use_container_width=True)
                plt.close(fig_pay)
        else:
            data_valid=False
            st.info("Add at least one component to continue.")

    st.markdown("---")

    # ── 3. Constraint ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span style="font-size:1.3rem">③</span> 🎯 Mental-account constraint</div>', unsafe_allow_html=True)

    # VaR / ES toggle
    constraint_type = st.radio(
        "Constraint type",
        ["VaR — Value at Risk", "ES — Expected Shortfall"],
        index=0, horizontal=True)
    use_es = constraint_type.startswith("ES")

    H_val = st.slider("Threshold H (%)", -25, -3, -10, 1) / 100

    if not use_es:
        alpha_val = st.slider("Shortfall probability α (%)", 1, 15, 5, 1) / 100
        L_val     = None
        st.markdown(
            '<div style="background:#1a1a2e;border:1px solid #3a3a5a;border-radius:6px;'
            'padding:.4rem 1rem;color:#8896a8;font-size:.78rem;margin-top:.3rem">'
            'VaR constraint: P(return &lt; H) ≤ α</div>',
            unsafe_allow_html=True)
        with st.expander("✨ AI-powered: What is the VaR constraint?", expanded=False):
            st.markdown(
                f'<div style="background:#0f1923;border:1px solid #4a9eff;'
                f'border-radius:6px;padding:.8rem 1rem;color:#c0c8d8;font-size:.85rem">'
                f'{CONSTRAINT_EXPLANATIONS["var"]}</div>',
                unsafe_allow_html=True)
    else:
        alpha_val = None
        L_val     = st.slider("ES lower bound L (%)", -50, -1, -15, 1) / 100
        st.markdown(
            '<div style="background:#1a1a2e;border:1px solid #3a3a5a;border-radius:6px;'
            'padding:.4rem 1rem;color:#8896a8;font-size:.78rem;margin-top:.3rem">'
            'ES constraint: E[return | return &lt; H] ≥ L</div>',
            unsafe_allow_html=True)
        with st.expander("✨ AI-powered: What is the ES constraint?", expanded=False):
            st.markdown(
                f'<div style="background:#0f1923;border:1px solid #4a9eff;'
                f'border-radius:6px;padding:.8rem 1rem;color:#c0c8d8;font-size:.85rem">'
                f'{CONSTRAINT_EXPLANATIONS["es"]}</div>',
                unsafe_allow_html=True)

    # Implied lambda — only for VaR
    if not use_es:
        cov_for_lam = corr_to_cov(sigs_in, corr_in)
        lam = implied_lambda(H_val, alpha_val, means_in, cov_for_lam)
        if lam is not None:
            st.markdown(
                f'<div style="background:#0f1923;border:1px solid #10b981;border-radius:6px;'
                f'padding:.5rem 1rem;margin-top:.3rem;color:#10b981;font-size:.85rem">'
                f'<b>Implied risk-aversion λ = {lam:.4f}</b><br>'
                f'<span style="color:#8896a8;font-size:.78rem">'
                f'MV optimal at λ={lam:.2f} ≡ behavioral optimal at H={H_val:.0%}, α={alpha_val:.0%}'
                f'</span></div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#1a0a00;border:1px solid #f59e0b;border-radius:6px;'
                        'padding:.4rem 1rem;color:#f59e0b;font-size:.78rem;margin-top:.3rem">'
                        '⚠️ Implied λ not available — the VaR constraint may be too tight or too loose for the current portfolio.</div>',
                        unsafe_allow_html=True)

    st.markdown("---")

    # ── 4. Grid ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span style="font-size:1.3rem">④</span> ⚡ Grid resolution</div>', unsafe_allow_html=True)
    grid_lbl=st.selectbox("Resolution",list(GRID_OPTIONS.keys()),
                           index=0,label_visibility="collapsed")
    m_val,mp_val=GRID_OPTIONS[grid_lbl]

    # AI-powered grid explanation
    with st.expander("✨ AI-powered: What does this resolution mean?", expanded=True):
        st.markdown(
            f'<div style="background:#0f1923;border:1px solid #4a9eff;'
            f'border-radius:6px;padding:.8rem 1rem;color:#c0c8d8;font-size:.85rem">'
            f'{GRID_EXPLANATIONS.get(grid_lbl, "No explanation available.")}</div>',
            unsafe_allow_html=True)

    if "High" in grid_lbl:
        st.markdown('<div class="warn-box">⚠️ May take 5–10 min.</div>',
                    unsafe_allow_html=True)
    elif "Standard" in grid_lbl:
        st.markdown('<div class="warn-box">⏱️ ~1–2 min.</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    run_btn=st.button("➎  ▶  Run optimiser",type="primary",
                       use_container_width=True,disabled=not data_valid)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def show_portfolio_data(names_in, means_in, sigs_in, corr_in):
    with st.expander("📋 Portfolio data used in this simulation", expanded=True):
        hs = "background:#4a9eff;color:#ffffff;font-weight:bold;padding:6px 10px;text-align:left"
        cs = "background:#ffffff;color:#111111;padding:5px 10px;border-bottom:1px solid #e0e0e0"
        rows = "".join(
            f"<tr><td style='{cs}'>{names_in[i]}</td>"
            f"<td style='{cs}'>{means_in[i]*100:.2f}%</td>"
            f"<td style='{cs}'>{sigs_in[i]*100:.2f}%</td></tr>"
            for i in range(len(names_in)))
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse'>"
            f"<tr><th style='{hs}'>Asset</th><th style='{hs}'>Mean return</th>"
            f"<th style='{hs}'>Std deviation</th></tr>{rows}</table>",
            unsafe_allow_html=True)
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        st.markdown("**Correlation matrix**")
        n = len(names_in)
        corr_rows = "".join(
            f"<tr><td style='{hs}'>{names_in[i]}</td>"
            + "".join(f"<td style='{cs};text-align:center'>{corr_in[i][j]:.3f}</td>" for j in range(n))
            + "</tr>" for i in range(n))
        col_headers = "".join(f"<th style='{hs};text-align:center'>{names_in[j]}</th>" for j in range(n))
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse'>"
            f"<tr><th style='{hs}'></th>{col_headers}</tr>{corr_rows}</table>",
            unsafe_allow_html=True)

st.markdown("<div style='margin-top:2.5rem'></div>", unsafe_allow_html=True)
tab1,tab2,tab3=st.tabs(["📊 Optimizer","📖 About","📚 Glossary"])

with tab1:
    # Header row: title left, photo right
    import os
    col_title, col_photo = st.columns([5, 1])
    with col_title:
        st.markdown("## Beyond Mean-Variance: Portfolio Optimiser with Derivatives & Structured Products — A Mental Accounts Framework")
        st.markdown(
            "Most portfolio optimisers stop at stocks and bonds. This app goes further — "
            "incorporating derivatives and structured products, handling **non-normal return distributions**, "
            "and optimising under a risk constraint you define: either the probability of loss below "
            "a threshold (**Value-at-Risk / VaR**) or the expected loss in the worst scenarios "
            "(**Expected Shortfall / ES**).")
    with col_photo:
        if os.path.exists("profile.jpeg"):
            import base64
            with open("profile.jpeg","rb") as _f:
                _b64 = base64.b64encode(_f.read()).decode()
            st.markdown(
                f'''<a href="https://www.linkedin.com/in/sami-jeddou-25787a404" target="_blank"
                   style="text-decoration:none">
  <div style="position:relative;display:inline-block;width:90px">
    <img src="data:image/jpeg;base64,{_b64}" style="width:90px;border-radius:6px;display:block"/>
    <div style="position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,0.55);
                color:#ffffff;font-size:.65rem;text-align:center;padding:3px 0;
                border-radius:0 0 6px 6px">Sami Jeddou</div>
  </div>
</a>''',
                unsafe_allow_html=True)

    if not run_btn:
        st.markdown("""
<div class="info-box" style="color:#ffffff !important">

### 👈 How to use this tool

Follow these steps in the sidebar:

<table style="width:100%;border-collapse:collapse;color:#ffffff">
<tr style="border-bottom:1px solid #3a3a5a">
  <td style="padding:.5rem .8rem;white-space:nowrap"><span style="font-size:1.4rem;font-weight:700">①</span></td>
  <td style="padding:.5rem .8rem"><strong>Portfolio data</strong> — Choose a data source: default base case, live market tickers, manual entry, or CSV upload</td>
</tr>
<tr style="border-bottom:1px solid #3a3a5a">
  <td style="padding:.5rem .8rem;white-space:nowrap"><span style="font-size:1.4rem;font-weight:700">②</span></td>
  <td style="padding:.5rem .8rem"><strong>Derivative &amp; parameters</strong> — Select a derivative or structured product type and set its characteristics (strike, maturity, floor, participation, etc.)</td>
</tr>
<tr style="border-bottom:1px solid #3a3a5a">
  <td style="padding:.5rem .8rem;white-space:nowrap"><span style="font-size:1.4rem;font-weight:700">③</span></td>
  <td style="padding:.5rem .8rem"><strong>Constraint</strong> — Choose VaR or ES constraint type, set threshold H, and set α (VaR) or L (ES)</td>
</tr>
<tr style="border-bottom:1px solid #3a3a5a">
  <td style="padding:.5rem .8rem;white-space:nowrap"><span style="font-size:1.4rem;font-weight:700">④</span></td>
  <td style="padding:.5rem .8rem"><strong>Grid resolution</strong> — Choose Fast for a quick preview, High precision for thesis-level accuracy</td>
</tr>
<tr>
  <td style="padding:.5rem .8rem;white-space:nowrap"><span style="font-size:1.4rem;font-weight:700">⑤</span></td>
  <td style="padding:.5rem .8rem"><strong>Run</strong> — Click <strong>▶ Run optimiser</strong></td>
</tr>
</table>

The chart will show three curves and two markers:
- 🔘 **Grey dashed** — classical mean-variance efficient frontier (Markowitz)
- 🔵 **Blue** — behavioural optimiser frontier without derivatives
- 🟡 **Gold** — behavioural optimiser frontier including your selected derivative
- ➡️ **White dotted arrow** — return gap between the behavioral frontier without and with the derivative, at the selected H and α constraint
- 🟢 **Green diamond** — equivalence point: the unique portfolio where the mean-variance and behavioral approaches yield exactly the same result (here at λ=3.795, H=-10%, α=5%). To the right of this point, adding derivatives allows the behavioral approach to outperform mean-variance.

At the equivalence point (λ=3.795, H=-10%, α=5%), the grey and blue curves meet exactly —
confirming the MVT/MAT equivalence proven in Das, Markowitz, Scheid & Statman (2010).
The gold curve shows what the behavioral approach, with the right choice of derivatives or
structured products, can unlock beyond what mean-variance can achieve.

</div>
""", unsafe_allow_html=True)
        # Sample chart
        import os
        if os.path.exists("sample_output.png"):
            st.image("sample_output.png", caption="Sample output — default base case with Capital-Guaranteed Note (CGN)", use_container_width=True)

        st.markdown("---")

        show_portfolio_data(names_in, means_in, sigs_in, corr_in)

        st.markdown("---")

        # LinkedIn + contact
        st.markdown("""
<div style="background:#0f1923;border:1px solid #4a9eff;border-radius:8px;padding:1rem 1.4rem;color:#ffffff">

**👤 About the author**

**Sami Jeddou**

Senior Financial Services Executive — Transformation, Risk & Capital Markets

🔗 [Connect on LinkedIn](https://www.linkedin.com/in/sami-jeddou-25787a404) &nbsp;&nbsp;|&nbsp;&nbsp;
🐙 [View source on GitHub](https://github.com/SamiJeddou/behavioral-portfolio-optimizer)

</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        st.markdown("""
<div style="color:#ffffff !important;margin-bottom:.8rem">

**💬 Get in touch**

Whether you are exploring this tool for a project, considering a collaboration,
or looking for a senior transformation or risk professional —
I would be glad to hear from you.

</div>
""", unsafe_allow_html=True)
        with st.form("contact_form"):
            sender_name  = st.text_input("Your name")
            sender_email = st.text_input("Your email")
            message      = st.text_area("Message", height=100,
                                         placeholder="Introduce yourself, share feedback, or tell me about an opportunity...")
            submitted = st.form_submit_button("Send message")
            if submitted:
                if sender_name and sender_email and message:
                    import requests as _req
                    try:
                        resp = _req.post(
                            "https://formspree.io/f/xvzyepoe",
                            data={
                                "name":    sender_name,
                                "email":   sender_email,
                                "message": message,
                            },
                            headers={"Accept": "application/json"},
                            timeout=10
                        )
                        if resp.status_code == 200:
                            st.success("✓ Message sent successfully. I will get back to you shortly.")
                        else:
                            st.error(f"Could not send message (status {resp.status_code}). Please try again or email sami.jeddou@protonmail.com directly.")
                    except Exception as ex:
                        st.error(f"Could not send message: {ex}. Please email sami.jeddou@protonmail.com directly.")
                else:
                    st.warning("Please fill in all fields before sending.")

        pass  # welcome screen shown, About tab still renders

    if run_btn:
        means_arr=np.array(means_in); sigs_arr=np.array(sigs_in)
        cov_mat=corr_to_cov(sigs_arr,corr_in)

        # Build derivative config
        der_config=None
        if der_type is not None:
            ui=der_params.get("underlying_idx",len(means_in)-1)
            sigs_for_config=sigs_arr.copy()
            sigs_for_config[ui]=der_params.get("vol",sigs_arr[ui])
            dc=build_der_config(der_type,der_params,sigs_for_config,ui)
            if dc:
                dc["r"]=der_params.get("r",0.03)
                dc["T"]=der_params.get("T",1.0)
                der_config=dc

        asset_labels=names_in+(["Derivative"] if der_config else [])

        with st.spinner("Computing mean-variance frontier..."):
            mv_x,mv_y,mv_eq=compute_mv_frontier(
                tuple(means_in),tuple(map(tuple,cov_mat.tolist())))

        with st.spinner("Behavioural optimiser — no derivative..."):
            try:
                _ctype = 'es' if use_es else 'var'
                _alpha = alpha_val if not use_es else 0.05
                _L     = L_val if use_es else None
                nd_xs,nd_ys,nd_lbls=build_frontier(
                    means_arr,sigs_arr,cov_mat,None,_alpha,m_val,mp_val,
                    constraint_type=_ctype,L=_L)
            except Exception as e:
                st.error(f"Optimizer failed: {e}")
                nd_xs,nd_ys,nd_lbls=[],[],[]

        der_xs,der_ys,der_lbls=[],[],[]
        if der_config:
            with st.spinner(f"Behavioural optimiser — {der_label_sel}..."):
                try:
                    der_xs,der_ys,der_lbls=build_frontier(
                        means_arr,sigs_arr,cov_mat,der_config,_alpha,m_val,mp_val,
                        constraint_type=_ctype,L=_L)
                except Exception as e:
                    st.warning(f"Derivative frontier failed: {e}")

        with st.spinner("Rendering chart..."):
            fig_plotly=plot_frontier_plotly(mv_x,mv_y,mv_eq,nd_xs,nd_ys,nd_lbls,
                                            der_xs,der_ys,der_lbls,der_label_sel,H_val,alpha_val)
            st.plotly_chart(fig_plotly, use_container_width=True)

        # ── Results ───────────────────────────────────────────────────────────────
        st.markdown("---")
        constraint_label = f"H={H_val:.0%}, α={_alpha:.0%}" if not use_es else f"H={H_val:.0%}, L={_L:.0%}"
        st.markdown(f"### Optimal portfolio — {constraint_label}")
        c1,c2=st.columns(2)

        nd_res=None
        with c1:
            st.markdown("**Without derivative**")
            try:
                nd_res,_=run_opt(means_arr,sigs_arr,cov_mat,None,H_val,_alpha,m_val,mp_val,
                               constraint_type=_ctype,L=_L)
                m1,m2,m3,m4=st.columns(4)
                m1.metric("Return",f"{nd_res['expected_return']*100:.2f}%")
                m2.metric("Std dev",f"{nd_res['std_dev']*100:.2f}%")
                m3.metric("Skewness",f"{nd_res['skewness']:.3f}")
                m4.metric(
                    'Shortfall prob' if not use_es else 'Exp. tail loss',
                    f"{nd_res['shortfall_stat']*100:.2f}%")
                st.markdown("**Weights**")
                for i,w in enumerate(nd_res["weights"]):
                    lbl=names_in[i] if i<len(names_in) else f"Asset {i+1}"
                    st.progress(float(w),text=f"{lbl}: {w*100:.1f}%")
                st.caption(f"Method: {nd_res.get('method_used','—')}")
            except Exception as e:
                st.warning(
                    "⚠️ No eligible portfolio found for these parameters. "
                    "The constraint is too tight — try increasing H (e.g. -15% instead of -5%), "
                    "increasing α, or switching to Fast resolution to explore parameters quickly.")

        with c2:
            if der_config:
                st.markdown(f"**With {der_label_sel}**")
                try:
                    dr_res,_=run_opt(means_arr,sigs_arr,cov_mat,der_config,
                                      H_val,_alpha,m_val,mp_val,
                                      constraint_type=_ctype,L=_L)
                    delta=(dr_res['expected_return']-(nd_res['expected_return'] if nd_res else 0))*100
                    m1,m2,m3,m4=st.columns(4)
                    m1.metric("Return",f"{dr_res['expected_return']*100:.2f}%",
                               delta=f"+{delta:.2f}pp" if delta>0 else f"{delta:.2f}pp")
                    m2.metric("Std dev",f"{dr_res['std_dev']*100:.2f}%")
                    m3.metric("Skewness",f"{dr_res['skewness']:.3f}")
                    m4.metric(
                        'Shortfall prob' if not use_es else 'Exp. tail loss',
                        f"{dr_res['shortfall_stat']*100:.2f}%")
                    st.markdown("**Weights**")
                    for i,w in enumerate(dr_res["weights"]):
                        lbl=asset_labels[i] if i<len(asset_labels) else f"Asset {i+1}"
                        st.progress(float(w),text=f"{lbl}: {w*100:.1f}%")
                    st.caption(f"Method: {dr_res.get('method_used','—')}")
                except Exception as e:
                    st.warning(
                        "⚠️ No eligible portfolio found for these parameters. "
                        "The constraint is too tight — try increasing H (e.g. -15% instead of -5%), "
                        "increasing α, or switching to Fast resolution to explore parameters quickly.")
            else:
                st.info("Select a derivative to compare.")



with tab2:
    import os as _os
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if _os.path.exists("profile.jpeg"):
            import base64 as _b64mod
            with open("profile.jpeg","rb") as _f2:
                _b64_2 = _b64mod.b64encode(_f2.read()).decode()
            st.markdown(
                f'''<a href="https://www.linkedin.com/in/sami-jeddou-25787a404" target="_blank"
                   style="text-decoration:none" title="Connect on LinkedIn">
  <div style="position:relative;display:inline-block;width:160px">
    <img src="data:image/jpeg;base64,{_b64_2}"
         style="width:160px;border-radius:8px;display:block;cursor:pointer"/>
    <div style="position:absolute;bottom:0;left:0;right:0;background:rgba(0,91,181,0.75);
                color:#ffffff;font-size:.72rem;text-align:center;padding:4px 0;
                border-radius:0 0 8px 8px">🔗 LinkedIn</div>
  </div>
</a>''',
                unsafe_allow_html=True)
    with col_b:
        st.markdown("## Sami Jeddou")
        st.markdown("**Senior Financial Services Executive — Transformation, Risk & Capital Markets**")
        st.markdown("Risk · Capital Markets · Post-Trade & Clearing · High-Value Payments · Quantitative Finance · Front-to-Back Delivery · Regulatory Programs")
        st.markdown("📍 Paris, France", unsafe_allow_html=True)
        st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/sami-jeddou-25787a404) &nbsp;|&nbsp; 🐙 [GitHub](https://github.com/SamiJeddou/behavioral-portfolio-optimizer) &nbsp;|&nbsp; 📧 sami.jeddou@protonmail.com", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### About this app")
    st.markdown(
        "This app extends classical **Markowitz mean-variance theory** to portfolios that include "
        "**derivatives and structured products**, using a **mental-accounting framework** with a "
        "downside risk constraint. It is a **Python** reimplementation and extension of the original "
        "R code developed as part of my MSc Finance thesis at the Università della Svizzera italiana "
        "(USI Lugano, 2012), supervised by Prof. Enrico De Giorgi. The Python version adds support "
        "for live market data, a custom structured product composer, and an extended optimizer for "
        "larger portfolios using differential evolution. It is based on the foundational work of "
        "Das, Markowitz, Scheid & Statman (2010).")

    st.markdown("### Professional background")
    st.markdown(
        "With over 20 years of experience in financial services transformation, I have delivered "
        "large-scale risk, regulatory, and front-to-back programs at and for tier-1 institutions — "
        "including BNP Paribas CIB, Crédit Agricole, BIL Luxembourg, and TMX Group — across roles "
        "as senior consultant, program director, and independent transformation lead.")
    st.markdown(
        "**Education:** Engineering and finance background — MEng, MSc Project and Program Management, "
        "École des Mines de Saint-Étienne · Master in Finance, USI Lugano · CFA Level I")
    st.markdown("""
**Key achievements:**
- Delivered €2M+ annual cost savings and reduced operational risk across global operations
- Designed and delivered greenfield risk and clearing platforms for CDS and OTC derivatives at leading central counterparties (CCPs)
- Built and led a €25M+ portfolio of concurrent risk and finance transformation initiatives
- Delivered major regulatory programs across multiple jurisdictions (EMIR, Basel IV, FRTB, IRRBB, IFRS 9, MiFID II, ISO 20022)

I am currently available for senior transformation, program director, or portfolio management
engagements — either freelance/contract or permanent — in France, Europe, or remote/hybrid.
""")

    st.markdown("### Algorithm")
    st.markdown(
        "The full algorithm is described in Das & Statman (2009) — *Beyond Mean-Variance: Portfolios with Derivatives and Non-Normal Returns in Mental Accounts*. "
        "The original R implementation is provided in the appendix of the thesis (Jeddou, 2012). "
        "This app is a Python reimplementation of that algorithm, with enhancements and extensions including support for live market data, a custom structured product composer, and an extended optimiser for larger portfolios.")
    st.markdown("""
**Step 1 — State space construction**
A discrete grid of return scenarios is built for all primary securities.
For each scenario, derivative returns are computed analytically using Black-Scholes pricing.

**Step 2 — Probability assignment**
Each state is assigned a probability using a Gaussian copula, correctly capturing the dependence structure between assets.

**Step 3 — Optimization**
For each candidate weight vector, the portfolio return distribution is evaluated against the mental-account constraint. Two constraint types are supported:
- **VaR constraint**: P(return < H) ≤ α — probability of loss beyond H must not exceed α
- **ES constraint**: E[return | return < H] ≥ L — expected loss in the tail must not exceed L

The best eligible portfolio (highest expected return satisfying the constraint) is selected via:
- *≤ 4 securities*: exhaustive grid search over all weight combinations
- *≥ 5 securities*: differential evolution — a global stochastic optimiser that scales to larger portfolios without exhaustive enumeration
""")

    st.markdown("### Data input & cleaning")
    st.markdown(
        "Four data input modes are supported. For live market data and CSV uploads, "
        "returns are automatically cleaned before being passed to the optimizer:")
    st.markdown("""
- **Default**: Das & Statman (2009) base case — 3 securities, pre-calibrated parameters, reproduces thesis results exactly
- **Live market data**: any global ticker from Yahoo Finance, daily or monthly frequency, over a user-defined date range. Auto-adjusted for splits and dividends. Cleaned automatically: stale price rows (zero returns) are removed and outliers beyond ±5 standard deviations are winsorised
- **Manual entry**: enter means, standard deviations, and correlations directly for 2–10 securities
- **CSV upload**: upload historical prices — returns computed automatically with the same cleaning applied as for live data
""")

    st.markdown("### MVT / MAT Equivalence")
    st.markdown(
        "When no derivatives are present, the mean-variance and behavioral frontiers converge exactly. "
        "For any choice of H and α, there exists a unique implied risk-aversion coefficient λ such that "
        "the mean-variance optimal portfolio and the behavioral optimal portfolio are identical. "
        "For example, at H = -10% and α = 5%, the implied λ = 3.795. "
        "This app computes and displays the implied λ dynamically — simply adjust the H and α sliders "
        "in the sidebar under the Mental-account constraint section to see the corresponding λ update in real time. "
        "Adding derivatives breaks this equivalence and reveals the superiority of the behavioral approach.")

    st.markdown("### Supported derivatives & structured products")
    st.markdown("""
| Type | Description |
|---|---|
| Put / Call | Standard European options |
| Safety collar | Long put + short call |
| Aggressive collar | Long call + short put |
| Straddle / Strangle | Long call + long put (same or different strikes) |
| Capital-guaranteed note | Uncapped or capped, with floor and participation rate |
| Barrier-M note | Corridor note with digital components |
| Custom composer | Build any payoff from calls, puts, digitals, and zero-coupon bonds |
""")

    st.markdown("### Academic references")
    st.markdown("""
- **Das, Sanjiv and Meir Statman (2009)** — *Beyond Mean-Variance: Portfolios with Derivatives and Non-Normal Returns in Mental Accounts*
- **Das, Sanjiv, Harry Markowitz, Jonathan Scheid and Meir Statman (2010)** — *Portfolio Optimization with Mental Accounts*, Journal of Financial and Quantitative Analysis, Vol. 45, No. 2, pp. 311–334
- **Jeddou, Sami (2012)** — *Beyond Mean-Variance: Options and Structured Products in Behavioral Portfolios*, MSc Finance Thesis, Università della Svizzera italiana (USI Lugano), supervised by Prof. Enrico De Giorgi. Available on [LinkedIn](https://www.linkedin.com/in/sami-jeddou-25787a404)
""")

with tab3:
    st.markdown("## 📚 AI Glossary & Reference")
    st.markdown(
        "Click any term below for an AI-generated explanation, or type your own question. "
        "Answers are tailored to the context of behavioral portfolio optimization.")

    GLOSSARY_TERMS = {
        "Derivatives & structured products": [
            "Put option", "Call option", "Safety collar", "Aggressive collar",
            "Straddle", "Strangle", "Capital-guaranteed note (CGN)", "Barrier-M note",
            "Digital option", "Zero-coupon bond"
        ],
        "Risk measures": [
            "Value at Risk (VaR)", "Expected Shortfall (ES)",
            "Shortfall probability", "Skewness", "Excess kurtosis"
        ],
        "Portfolio theory": [
            "Mean-variance efficient frontier", "Markowitz optimization",
            "Mental accounting", "Behavioral portfolio theory",
            "MVT/MAT equivalence", "Implied risk aversion lambda",
            "Gaussian copula", "Black-Scholes pricing"
        ],
        "Academic references": [
            "Das & Statman (2009) — Beyond Mean-Variance",
            "Das, Markowitz, Scheid & Statman (2010) JFQA",
            "Jeddou (2012) MSc thesis USI Lugano"
        ]
    }

    if "glossary_response" not in st.session_state:
        st.session_state["glossary_response"] = ""
    if "glossary_term" not in st.session_state:
        st.session_state["glossary_term"] = ""

    for category, terms in GLOSSARY_TERMS.items():
        st.markdown(f"**{category}**")
        cols = st.columns(3)
        for i, term in enumerate(terms):
            if cols[i % 3].button(term, key=f"gloss_{term}", use_container_width=True):
                st.session_state["glossary_term"] = term
                with st.spinner(f"Looking up: {term}..."):
                    st.session_state["glossary_response"] = get_explanation(term)
        st.markdown("")

    st.markdown("---")
    st.markdown("### Ask your own question")
    custom_q = st.text_input(
        "Type a term or question",
        placeholder="e.g. What is the difference between VaR and ES?")
    if st.button("🤖 Ask AI", type="primary"):
        if custom_q.strip():
            st.session_state["glossary_term"] = custom_q
            with st.spinner("Thinking..."):
                st.session_state["glossary_response"] = get_ai_chat_response(
                    custom_q,
                    portfolio_context=f"Portfolio has {len(means_in)} securities with means {[f'{m*100:.1f}%' for m in means_in]}")
        else:
            st.warning("Please enter a question first.")

    if st.session_state["glossary_response"]:
        st.markdown("---")
        st.markdown(f"**{st.session_state['glossary_term']}**")
        st.markdown(
            f'<div style="background:#0f1923;border:1px solid #4a9eff;border-radius:8px;'
            f'padding:1rem 1.2rem;color:#c0c8d8;font-size:.9rem;line-height:1.6">'
            f'{st.session_state["glossary_response"]}</div>',
            unsafe_allow_html=True)
        if st.button("Clear response"):
            st.session_state["glossary_response"] = ""
            st.session_state["glossary_term"] = ""
            st.rerun()
