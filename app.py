"""  # v4
Behavioral Portfolio Optimizer — Streamlit Dashboard
Full version with: live market data, manual input, CSV upload,
custom structured product composer, and extended optimizer (5+ securities).
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import plotly.graph_objects as go
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
.info-box{background:#1a1a2e;border:1px solid #1a6bbf;border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem;color:#ffffff !important}
.warn-box{background:#1a1200;border:1px solid #f59e0b;border-radius:6px;padding:.5rem 1rem;color:#f59e0b;font-size:.82rem;margin-top:.3rem}
.ok-box{background:#ffffff;border:1px solid #10b981;border-radius:6px;padding:.5rem 1rem;color:#1a5c3a;font-size:.82rem;margin-top:.3rem}

    /* Larger tab labels */
    button[data-baseweb="tab"] p {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }
    .section-header{border-left:4px solid #1a6bbf;background:#1a1a2e;padding:.4rem .8rem;border-radius:0 6px 6px 0;margin-top:1.2rem;margin-bottom:.5rem;color:#ffffff;font-weight:600;font-size:1.05rem;letter-spacing:.02em;text-align:center}

    .sidebar-divider{border:none;border-top:2px solid #2a3a4a;margin:1rem 0}
    section[data-testid="stSidebar"] div.stButton > button,section[data-testid="stSidebar"] div.stButton > button[kind="primary"]{background:linear-gradient(180deg,#5aabff 0%,#2d7dd2 100%) !important;border:none !important;border-bottom:3px solid #1a5fa0 !important;border-radius:8px !important;color:#ffffff !important;font-size:1.05rem !important;font-weight:700 !important;padding:.6rem 1rem !important;box-shadow:0 4px 8px rgba(0,0,0,0.5) !important;text-shadow:0 1px 2px rgba(0,0,0,0.3) !important;width:100% !important}
    section[data-testid="stSidebar"] div.stButton > button:hover{background:linear-gradient(180deg,#6bbfff 0%,#3a8de0 100%) !important;box-shadow:0 6px 14px rgba(0,0,0,0.6) !important;transform:translateY(-1px) !important}
    section[data-testid="stSidebar"] div.stButton > button:active{background:linear-gradient(180deg,#2d7dd2 0%,#1a5fa0 100%) !important;border-bottom:1px solid #1a5fa0 !important;transform:translateY(1px) !important}
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
        import anthropic
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
        "Runs in approximately 3–8 minutes depending on the number of securities and derivative type. Provides a good balance between speed and accuracy — "
        "results are close to the precise solution in most cases. "
        "Recommended for most use cases once you have identified the right parameters."
    ),
    "🎯 High precision (m=51, m'=99)": (
        "Matches the original thesis parameters exactly — 51 return scenarios and 99 weight steps, "
        "the same values used in Das & Statman (2009) and Jeddou (2012). "
        "Results are publication-quality and directly comparable to academic benchmarks. "
        "May take 15–30 minutes depending on the number of securities and derivative type. "
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

def plot_frontier_plotly(mv_x, mv_y, mv_eq,
                         nd_x, nd_y, nd_lbls,
                         der_x, der_y, der_lbls,
                         der_label, H_sel, alpha,
                         p3_x=None, p3_y=None):
    """Interactive Plotly version of the frontier chart with hover tooltips."""
    fig = go.Figure()

    # ── Mean-variance frontier ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=mv_x, y=mv_y, mode='lines',
        name='Mean-variance efficient frontier (Markowitz)',
        legendrank=1,
        line=dict(color='#a855f7', width=2, dash='dash'),
        hovertemplate='<b>Mean-Variance Efficient Frontier (Markowitz)</b><br>Std Dev: %{x:.2f}%<br>Expected Return: %{y:.2f}%<extra></extra>'
    ))

    # ── Behavioral — no derivative ────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=nd_x, y=nd_y, mode='lines+markers',
        name='Behavioural efficient frontier — no derivative',
        legendrank=2,
        line=dict(color='#1a6bbf', width=2.5),
        marker=dict(size=9, color='#1a6bbf', symbol='circle'),
        text=nd_lbls,
        hovertemplate='<b>Behavioral (no derivative)</b><br>Threshold: %{text}<br>Std Dev: %{x:.2f}%<br>Expected Return: %{y:.2f}%<extra></extra>'
    ))

    # ── Behavioral — with derivative ──────────────────────────────────────────
    if der_x:
        fig.add_trace(go.Scatter(
            x=der_x, y=der_y, mode='markers',
            name=f'Behavioural optimum portfolios — derivative frontier ({der_label})',
            legendrank=3,
            marker=dict(size=8, color='#f59e0b', symbol='square'),
            opacity=0.7,
            text=der_lbls,
            hovertemplate=f'<b>Behavioural optimal portfolio (with {der_label})</b><br>Threshold: %{{text}}<br>Std Dev: %{{x:.2f}}%<br>Expected Return: %{{y:.2f}}%<extra></extra>'
        ))

        # ── Portfolio (2) highlighted point at selected H ─────────────────────
        try:
            _i2 = der_lbls.index(f'H={H_sel:.0%}')
            fig.add_trace(go.Scatter(
                x=[der_x[_i2]], y=[der_y[_i2]], mode='markers',
                name=f'Portfolio (2) — optimum with {der_label} at H={H_sel:.0%}',
                legendrank=6,
                marker=dict(size=14, color='#ff6b00', symbol='square',
                           line=dict(color='white', width=1.5)),
                hovertemplate=(f'<b>Portfolio (2)</b><br>Optimum with {der_label}<br>'
                              f'Std Dev: %{{x:.2f}}%<br>Expected Return: %{{y:.2f}}%<extra></extra>')
            ))
        except (ValueError, IndexError):
            pass

        # Gain arrow at selected H
        try:
            i0 = nd_lbls.index(f'H={H_sel:.0%}')
            i1 = der_lbls.index(f'H={H_sel:.0%}')
            x0, y0 = nd_x[i0], nd_y[i0]
            x1, y1 = der_x[i1], der_y[i1]
            gain = y1 - y0
            # Dashed white line with triangle marker as arrowhead at end
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines+markers',
                line=dict(color='#ffffff', width=2, dash='dash'),
                marker=dict(
                    symbol=['circle', 'arrow'],
                    size=[0, 12],
                    color='#ffffff',
                    angleref='previous'
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
            # Text connected by arrow to the gold end point (behavioural with derivative)
            fig.add_annotation(
                x=x1, y=y1,
                ax=x1 + max((x1-x0)*0.3, 8),
                ay=y1 + (y1-y0)*0.2,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1.0,
                arrowwidth=1.5, arrowcolor='#ffffff',
                text=f'<b>+{gain:.1f} pp return (with derivative)</b><br>same H & α constraint<br>(same risk aversion λ)',
                font=dict(color='#ffffff', size=10),
                bgcolor='rgba(13,17,23,0.85)',
                bordercolor='#ffffff', borderwidth=1,
                align='left', xanchor='left'
            )
        except (ValueError, IndexError):
            pass

    # ── Equivalence point ─────────────────────────────────────────────────────
    if mv_eq:
        fig.add_trace(go.Scatter(
            x=[mv_eq[0]], y=[mv_eq[1]], mode='markers',
            name='Portfolio (1) — Equivalence point: MV = Behavioural (no derivatives) ↔ H=-10%, α=5%',
            legendrank=5,
            marker=dict(size=13, color='#10b981', symbol='diamond',
                        line=dict(width=0)),
            showlegend=True,
            hovertemplate='<b>Equivalence point</b><br>MV = Behavioural (no derivatives)<br>where λ=3.795 ↔ H=-10%, α=5%<br>Std Dev: %{x:.2f}%<br>Expected Return: %{y:.2f}%<extra></extra>'
        ))
        fig.add_annotation(
            x=mv_eq[0], y=mv_eq[1],
            text=f'Equivalence point<br>MV = Behavioural (no derivatives)<br>where λ=3.795 ↔ H=-10%, α=5%<br>Return = {mv_eq[1]:.1f}%',
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
        font=dict(color='#ffffff', size=10, style='italic'),
        bgcolor='rgba(13,17,23,0.85)',
        bordercolor='#3a3a5a', borderwidth=1,
        xanchor='center', yanchor='bottom'
    )

    # ── Portfolio (3) point ──────────────────────────────────────────────────
    if p3_x is not None and p3_y is not None:
        fig.add_trace(go.Scatter(
            x=[p3_x], y=[p3_y], mode='markers',
            name=f'Portfolio (3) — same variance as Portfolio (1), with {der_label}',
            legendrank=7,
            marker=dict(size=14, color='#e76f51', symbol='star',
                       line=dict(color='white', width=1)),
            hovertemplate=(f'<b>Portfolio (3)</b><br>Same std dev as Portfolio (1)<br>'
                          f'Std Dev: {p3_x:.2f}%<br>Expected Return: {p3_y:.2f}%<extra></extra>')
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        title=dict(
            text='Mean-Variance vs Behavioural Portfolio Efficient Frontier',
            font=dict(color='white', size=15),
            x=0.5,
            xanchor='center',
            xref='paper'
        ),
        xaxis=dict(
            title=dict(text='Portfolio Risk — Standard Deviation (%)',
                       font=dict(color='#c0c8d8', size=13)),
            gridcolor='#1e2130', gridwidth=0.5,
            color='#c0c8d8', zerolinecolor='#2a2a3a',
            range=[max(0, min(mv_x) - 1),
                   max(max(mv_x), max(der_x) if der_x else 0) * 1.06]
        ),
        yaxis=dict(
            title=dict(text='Expected Return (%)',
                       font=dict(color='#c0c8d8', size=13)),
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
            bordercolor='#1a6bbf',
            font=dict(color='white', size=11)
        ),
        margin=dict(t=80, b=60, l=60, r=20),
        height=560
    )

    # Update margin only
    fig.update_layout(margin=dict(t=80, b=140, l=60, r=20))
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.5, y=-0.22,
        text='Behavioural frontiers shown at discrete H levels (-5% to -20%) — each point optimal for that constraint | MV frontier continuous via λ sweep | Both converge via MVT/MAT equivalence when no derivatives present',
        showarrow=False,
        font=dict(color='#8896a8', size=9, style='italic'),
        xanchor='center'
    )
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.5, y=-0.27,
        text='Jeddou (2026) — Beyond Mean-Variance Portfolio Optimiser  |  Built on Das & Statman (2009), Das, Markowitz, Scheid & Statman (2010) JFQA & Jeddou (2012)',
        showarrow=False,
        font=dict(color='#ffffff', size=9, style='italic'),
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
    ax.plot(mv_x,mv_y,color="#a855f7",linewidth=2,linestyle="--",
            label="Mean-variance frontier (Markowitz)",zorder=2,alpha=0.9)
    ax.plot(nd_x,nd_y,color="#1a6bbf",linewidth=2.5,marker="o",markersize=7,
            markerfacecolor="#1a6bbf",label="Behavioral — no derivative",zorder=3)
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
                                        lw=1.6,linestyle="dashed"))
            ax.text(0.55, 0.45,
                    f"+{y1-y0:.1f} pp return\nsame H & α constraint\n(same risk aversion λ)",
                    color="#ffffff", fontsize=8, ha='center', va='center',
                    transform=ax.transAxes)
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
            transform=ax.transAxes,color="#ffffff",fontsize=7.5,
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
             "Jeddou (2026) — Beyond Mean-Variance Portfolio Optimiser  |  "
             "Built on Das & Statman (2009), Das, Markowitz, Scheid & Statman (2010) JFQA & Jeddou (2012)",
             ha="center",color="#ffffff",fontsize=7,style="italic")
    all_x=mv_x+nd_x+(der_x if der_x else [])
    all_y=mv_y+nd_y+(der_y if der_y else [])
    ax.set_xlim(0,max(all_x)*1.15); ax.set_ylim(min(all_y)-3,max(all_y)+6)
    plt.tight_layout(rect=[0,0.02,1,1])
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Optimisation Parameters")
    st.markdown("\n---\n")

    # ── 1. Data source ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.6rem;height:1.6rem;line-height:1.6rem;text-align:center;font-size:1rem;font-weight:700">1</span><span style="display:block">📂 PORTFOLIO DATA</span></div>', unsafe_allow_html=True)
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
        st.markdown('<div style="background:#ffffff;border:1px solid #1a6bbf;border-radius:8px;padding:.6rem 1rem;margin-bottom:.5rem;color:#111111;font-size:.82rem">'
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

    st.markdown("\n---\n")

    # ── 2. Derivative ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.6rem;height:1.6rem;line-height:1.6rem;text-align:center;font-size:1rem;font-weight:700">2</span><span style="display:block">📊 DERIVATIVE / STRUCTURED PRODUCT</span></div>', unsafe_allow_html=True)
    der_label_sel=st.selectbox("Type",list(PREDEFINED_DERIVATIVES.keys()),
                                index=0,label_visibility="collapsed")
    der_type=PREDEFINED_DERIVATIVES[der_label_sel]
    der_params={}

    # AI tooltip for selected derivative
    if der_type is not None and der_type != "custom":
        st.markdown(
            f'<div style="background:#ffffff;border:1px solid #1a6bbf;border-radius:6px;'
            f'padding:.6rem .8rem;color:#111111;font-size:.82rem;margin-top:.3rem">'
            f'<b style="color:#1a3a6b">✨ AI-powered: What is this instrument?</b><br>'
            f'{get_explanation(der_label_sel)}</div>',
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

    st.markdown("\n---\n")

    # ── 3. Constraint ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.6rem;height:1.6rem;line-height:1.6rem;text-align:center;font-size:1rem;font-weight:700">3</span><span style="display:block">🎯 MENTAL-ACCOUNT CONSTRAINT</span></div>', unsafe_allow_html=True)

    # VaR / ES toggle
    constraint_type = st.radio(
        "Constraint type",
        ["VaR — Value at Risk", "ES — Expected Shortfall"],
        index=0, horizontal=True)
    use_es = constraint_type.startswith("ES")

    H_val = st.slider("Threshold H (%)", -40, -1, -10, 1) / 100
    st.markdown(
        '<div style="background:#ffffff;border:1px solid #3a3a5a;border-radius:6px;'
        'padding:.3rem .8rem;color:#555555;font-size:.76rem;margin-top:.2rem">'
        'Range extended to -40% to accommodate highly volatile assets '
        '(e.g. cryptocurrencies, emerging market equities).</div>',
        unsafe_allow_html=True)

    if not use_es:
        alpha_val = st.slider("Shortfall probability α (%)", 1, 15, 5, 1) / 100
        L_val     = None
        # Formula box — white background
        st.markdown(
            '<div style="background:#ffffff;border:1px solid #3a3a5a;border-radius:6px;'
            'padding:.4rem 1rem;color:#333333;font-size:.78rem;margin-top:.3rem">'
            'VaR constraint: P(return &lt; H) ≤ α</div>',
            unsafe_allow_html=True)
        # Implied lambda — between formula and AI explanation
        cov_for_lam = corr_to_cov(sigs_in, corr_in)
        lam = implied_lambda(H_val, alpha_val, means_in, cov_for_lam)
        if lam is not None:
            st.markdown(
                f'<div style="background:#ffffff;border:1px solid #1a6bbf;border-radius:6px;'
                f'padding:.5rem 1rem;margin-top:.3rem;color:#1a3a6b;font-size:.85rem">'
                f'<b>Implied risk-aversion λ = {lam:.4f}</b><br>'
                f'<span style="color:#555555;font-size:.78rem">'
                f'MV optimal at λ={lam:.2f} ≡ behavioural optimal at H={H_val:.0%}, α={alpha_val:.0%}'
                f'</span></div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#fffbea;border:1px solid #f59e0b;border-radius:6px;'
                        'padding:.4rem 1rem;color:#7a4f00;font-size:.78rem;margin-top:.3rem">'
                        '⚠️ Implied λ not available — the VaR constraint may be too tight or too loose for the current portfolio.</div>',
                        unsafe_allow_html=True)
        # AI explanation last
        st.markdown(
            f'<div style="background:#ffffff;border:1px solid #1a6bbf;border-radius:6px;'
            f'padding:.6rem .8rem;color:#111111;font-size:.82rem;margin-top:.3rem">'
            f'<b style="color:#1a3a6b">✨ AI-powered: What is the VaR constraint?</b><br>'
            f'{CONSTRAINT_EXPLANATIONS["var"]}</div>',
            unsafe_allow_html=True)
    else:
        alpha_val = None
        L_val     = st.slider("ES lower bound L (%)", -50, -1, -15, 1) / 100
        # Formula box — white background
        st.markdown(
            '<div style="background:#ffffff;border:1px solid #3a3a5a;border-radius:6px;'
            'padding:.4rem 1rem;color:#333333;font-size:.78rem;margin-top:.3rem">'
            'ES constraint: E[return | return &lt; H] ≥ L</div>',
            unsafe_allow_html=True)
        # AI explanation
        st.markdown(
            f'<div style="background:#ffffff;border:1px solid #1a6bbf;border-radius:6px;'
            f'padding:.6rem .8rem;color:#111111;font-size:.82rem;margin-top:.3rem">'
            f'<b style="color:#1a3a6b">✨ AI-powered: What is the ES constraint?</b><br>'
            f'{CONSTRAINT_EXPLANATIONS["es"]}</div>',
            unsafe_allow_html=True)

    # Implied lambda block already handled above for VaR case
    if use_es:
        pass  # no lambda for ES

    st.markdown("\n---\n")

    # ── 4. Grid ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.6rem;height:1.6rem;line-height:1.6rem;text-align:center;font-size:1rem;font-weight:700">4</span><span style="display:block">⚡ GRID RESOLUTION</span></div>', unsafe_allow_html=True)
    grid_lbl=st.selectbox("Resolution",list(GRID_OPTIONS.keys()),
                           index=0,label_visibility="collapsed")
    m_val,mp_val=GRID_OPTIONS[grid_lbl]

    # AI-powered grid explanation
    st.markdown(
        f'<div style="background:#ffffff;border:1px solid #1a6bbf;border-radius:6px;'
        f'padding:.6rem .8rem;color:#111111;font-size:.82rem;margin-top:.3rem">'
        f'<b style="color:#1a3a6b">✨ AI-powered: What does this resolution mean?</b><br>'
        f'{GRID_EXPLANATIONS.get(grid_lbl, "No explanation available.")}</div>',
        unsafe_allow_html=True)

    if "High" in grid_lbl:
        st.markdown('<div class="warn-box">⚠️ May take 15–30 min. Recommended for final results only.</div>',
                    unsafe_allow_html=True)
    elif "Standard" in grid_lbl:
        st.markdown('<div class="warn-box">⏱️ ~3–8 min depending on securities and derivative type.</div>',
                    unsafe_allow_html=True)

    st.markdown("\n---\n")
    # Inject button style directly before the button
    st.markdown("""
<style>
div[data-testid="stSidebarContent"] button {
    background: linear-gradient(180deg, #5aabff 0%, #1a6bbf 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border-radius: 8px !important;
    border: none !important;
    border-bottom: 3px solid #0d4a8f !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.5) !important;
}
div[data-testid="stSidebarContent"] button p {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0 !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)
    run_btn=st.button(
        "5  ▶  RUN OPTIMISER",
        type="primary",
        use_container_width=True,
        disabled=not data_valid)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def show_portfolio_data(names_in, means_in, sigs_in, corr_in):
    with st.expander("📋 Portfolio data used in this simulation", expanded=True):
        hs = "background:#1a6bbf;color:#ffffff;font-weight:bold;padding:6px 10px;text-align:left"
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

# ── Finance banner ────────────────────────────────────────────────────────
st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
st.markdown(f'''
<div style="width:100%;background:#020c1b;padding:0;margin-bottom:0;display:flex;align-items:stretch;height:148px;overflow:hidden">
<div style="flex:1;overflow:hidden;height:148px;max-height:148px;background:#f0f4f8"><img src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAQDAwMDAgQDAwMEBAQFBgoGBgUFBgwICQcKDgwPDg4MDQ0PERYTDxAVEQ0NExoTFRcYGRkZDxIbHRsYHRYYGRj/2wBDAQQEBAYFBgsGBgsYEA0QGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBj/wAARCAHSAr0DASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD7+ooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiig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whB7EA0Aby/dpawfB+pR6x4E0rUI52n8y3UNI+ctIvyvnP8AtA1vUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRSZoAWikBzS0AFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFMbmn0wjPbrQBxXwyP2Twnd+Hs7zouoz6eZ+gmORJux2/1uMc9K7cHNcNoYFn8aPFViT5ENxa2d3BB9xZX/eiaVR3P+qDEeq57V3C45xQA6iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiig9KAEAqrPcxQXUUcmR5nAPbPvSXkxhhJR1WRvkj3+tZUhlUx2aEybgS8j8kDv+dNK5EpWN5WyMgjBpQawbe7uVu3jt0DW0ICEN14HQfmOv+NaVpfQXaB435wCUPUdev5H8qTVilJMvUUwdR0p9AwooqI5DUAS0VFu/wDr1Iv3aNgFooooAKKKKACiiigAooooA4PxKf7K+MXg/WTmU6gt1oXl9PL8yP7V5ue+Pse3H+3nPGD3KDGa4r4mQvB4XtPEFtGxvNG1G1u4JAuRArSCGeRh0wIJZsk8Ac8YzXZQuksQmjcSRuAVYHII9aAJqKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAbUM1wlvEZJn2oO9SM4VCxIAHXNYt60OpWyzxvIoiJZHGevrjoaaVyZOxHeLEVk+0yGWWY4j4/IAe3+NUGnn06EWc8m6WVf8Aj46nPfI7egqxbSeYpv7hdkoiGE5GF65GfXP9O1KX8q0a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PJsrRwszYA2j/Vr61Xl2aXALewi+e4YlUJyu7uSPc5JPryetS2BMkBvbgqJWBDAf8swCfl/CooY01NpLuXPkyDEAHGFwMv8Aj1+mKoCctFpem4L+pHH35CSeg7k5quLOODTRJcFFmD/aGkJz+8+vfGcfgKSDzLu/CSAhbMkElgfMbsenYc59SR2qW7xc30VgM7B+9mGf4ewI9zn8qAI9Kklktpru7QwzNId6PnCAdME9sc/iak05PMM2oEYM5wuRyI1zgHHuSfxpurvPJbLZWjoLmfpzwFHLH+Q/GknvEj08wW5UXGRbiMNyjEdBx2GT+FABaD7XqF3duTJFxBHkgjjOSPY5H/fFZymeHWLi7i867sbf92I92Xik/iOD6DHTnnitaR4tL0jPGIlxycbyTx+JJ/WlsYPs1gPNJ8wkyO79ST6/57UEplZ7iHUdStoreQSxQkyy7GG0YGAHHrk559Kl1WWQWHkRPtmnYRr82CATgke4GT+FUtPglihudTs7eN5biUuEf5C6gEKCe2OecGn28rX+uCcoyx28PzQyDkSt/wDWyPxqgNOKOOKFIowEQDA46VkWglh1O+1MB/JabyzD3TbwWTHqecdav6lcNbaZNLA6CfGyHeMgyHhRj3OBTrS3jtLGK3g+5GuAScn6mpAZe3iWmjzX6IZEjiLgR8l/pRp9obOwSJyDISZJXAwHkPLHHuSTWZeps1aG3wRZD/SZkI+5g8YPpnqK2wRtzxiqAgvb2ewtvMtiDOSIolOSCx6VoaTqjSytY32Fu1G7I4Eq/wB5f6jt9MVjMRd+IQAMizX5nz/Ew+5j6YOfetJflkWQKNy9DjkVE4XHCo46HQjmnVhafr1reatPpz4SaJ8D5shuAT9Dz09CPWtssBWFrHWndDqKKKCgprU6mOe1AHD/AA2H2WDxDpB/cpY6zcR29p0EFuSDEFXsnXHbriu7rg7AnS/j1q1pjzP7Z0yK9D9PK+zsIivvnzc54xjvmu6DZOMUAOooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAEJA61namZnhWKGfygxxI6/eC+3oferFzcCIY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" style="width:100%;height:148px;object-fit:cover;object-position:center top;opacity:0.95"/></div>
<div style="max-width:900px;flex-shrink:0;height:148px;background:linear-gradient(135deg,#020c1b 0%,#071428 40%,#0a1a35 70%,#020c1b 100%);border-radius:0;overflow:hidden;border:none;display:flex;align-items:stretch;font-family:monospace;margin-bottom:0">
  <div style="flex:1.5;padding:16px 18px;border-right:1px solid #1a3a5c;display:flex;flex-direction:column;justify-content:center;gap:10px">
    <div style="color:#ffffff;font-size:13px;font-weight:700;letter-spacing:0.03em;font-family:Georgia,serif;line-height:1.45">Portfolio Optimiser<br>with Derivatives &amp;<br>Structured Products</div>
    <div style="color:rgba(74,158,255,0.65);font-size:7.5px;letter-spacing:0.22em">BEYOND MEAN-VARIANCE · MENTAL ACCOUNTS FRAMEWORK</div>
    <div style="display:flex;flex-wrap:wrap;gap:4px">
      <span style="background:rgba(26,107,191,0.18);border:1px solid rgba(74,158,255,0.28);border-radius:3px;color:rgba(74,158,255,0.75);font-size:7px;letter-spacing:0.1em;padding:2px 5px">LIVE MARKET DATA</span>
      <span style="background:rgba(26,107,191,0.18);border:1px solid rgba(74,158,255,0.28);border-radius:3px;color:rgba(74,158,255,0.75);font-size:7px;letter-spacing:0.1em;padding:2px 5px">ALL FINANCIAL INSTRUMENTS</span>
      <span style="background:rgba(26,107,191,0.18);border:1px solid rgba(74,158,255,0.28);border-radius:3px;color:rgba(74,158,255,0.75);font-size:7px;letter-spacing:0.1em;padding:2px 5px">AI-POWERED</span>
      <span style="background:rgba(26,107,191,0.18);border:1px solid rgba(74,158,255,0.28);border-radius:3px;color:rgba(74,158,255,0.75);font-size:7px;letter-spacing:0.1em;padding:2px 5px">CRYPTO ASSETS</span>
      <span style="background:rgba(26,107,191,0.18);border:1px solid rgba(74,158,255,0.28);border-radius:3px;color:rgba(74,158,255,0.75);font-size:7px;letter-spacing:0.1em;padding:2px 5px">VaR / ES</span>
      <span style="background:rgba(26,107,191,0.18);border:1px solid rgba(74,158,255,0.28);border-radius:3px;color:rgba(74,158,255,0.75);font-size:7px;letter-spacing:0.1em;padding:2px 5px">MVT/MAT</span>
      <span style="background:rgba(26,107,191,0.18);border:1px solid rgba(74,158,255,0.28);border-radius:3px;color:rgba(74,158,255,0.75);font-size:7px;letter-spacing:0.1em;padding:2px 5px">GRID SEARCH · COBYLA</span>
    </div>
  </div>
  <div style="flex:3.2;display:grid;grid-template-columns:repeat(4,1fr)">
    <!-- Col 1: Live market ticker -->
    <div style="border-right:1px solid #1a3a5c;display:flex;flex-direction:column">
      <div style="height:82px;position:relative;overflow:hidden;border-bottom:1px solid #0d2a4a">
        <svg width="100%" height="82" viewBox="0 0 80 82" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
          <rect width="80" height="82" fill="#020c1b"/>
          <rect x="2" y="4" width="76" height="12" fill="rgba(16,185,129,0.08)" rx="1"/>
          <text x="6" y="13" fill="#10b981" font-size="6" font-family="monospace">AAPL</text>
          <text x="34" y="13" fill="rgba(255,255,255,0.7)" font-size="6" font-family="monospace">189.42</text>
          <text x="63" y="13" fill="#10b981" font-size="6" font-family="monospace">+1.2%</text>
          <rect x="2" y="18" width="76" height="12" fill="rgba(239,68,68,0.07)" rx="1"/>
          <text x="6" y="27" fill="#ef4444" font-size="6" font-family="monospace">BTC</text>
          <text x="34" y="27" fill="rgba(255,255,255,0.7)" font-size="6" font-family="monospace">67,340</text>
          <text x="63" y="27" fill="#ef4444" font-size="6" font-family="monospace">-0.8%</text>
          <rect x="2" y="32" width="76" height="12" fill="rgba(16,185,129,0.08)" rx="1"/>
          <text x="6" y="41" fill="#10b981" font-size="6" font-family="monospace">SPY</text>
          <text x="34" y="41" fill="rgba(255,255,255,0.7)" font-size="6" font-family="monospace">521.80</text>
          <text x="63" y="41" fill="#10b981" font-size="6" font-family="monospace">+0.4%</text>
          <rect x="2" y="46" width="76" height="12" fill="rgba(239,68,68,0.07)" rx="1"/>
          <text x="6" y="55" fill="#ef4444" font-size="6" font-family="monospace">ETH</text>
          <text x="34" y="55" fill="rgba(255,255,255,0.7)" font-size="6" font-family="monospace">3,421</text>
          <text x="63" y="55" fill="#ef4444" font-size="6" font-family="monospace">-1.1%</text>
          <path d="M4,74 L10,70 L16,72 L22,66 L28,68 L34,62 L40,58 L46,60 L52,54 L58,50 L64,46 L70,42 L76,38" fill="none" stroke="#4a9eff" stroke-width="1.2" opacity="0.7"/>
          <line x1="0" y1="64" x2="80" y2="64" stroke="#0d2a4a" stroke-width="0.5"/>
        </svg>
      </div>
      <div style="padding:7px 10px;height:68px;display:flex;flex-direction:column;justify-content:flex-start;align-items:center;text-align:center">
        <div style="color:rgba(200,220,255,0.9);font-size:7.5px;font-weight:600;letter-spacing:0.12em;margin-bottom:3px;min-height:18px">LIVE MARKET DATA</div>
        <div style="font-size:19px;font-weight:700;font-family:Georgia,serif;color:#4a9eff">10,000+</div>
        <div style="font-size:8px;margin-top:3px;color:rgba(150,180,220,0.55)">tickers · equities · crypto · ETFs</div>
        <div style="height:2px;margin-top:5px;border-radius:1px;background:#1a3a5c"><div style="width:75%;height:100%;border-radius:1px;background:#4a9eff"></div></div>
      </div>
    </div>
    <!-- Col 2: Candlesticks -->
    <div style="border-right:1px solid #1a3a5c;display:flex;flex-direction:column">
      <div style="height:82px;position:relative;overflow:hidden;border-bottom:1px solid #0d2a4a">
        <svg width="100%" height="82" viewBox="0 0 80 82" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
          <rect width="80" height="82" fill="#020c1b"/>
          <line x1="0" y1="41" x2="80" y2="41" stroke="#0d2a4a" stroke-width="0.6"/>
          <line x1="7" y1="62" x2="7" y2="74" stroke="#10b981" stroke-width="0.8"/><rect x="3" y="65" width="8" height="7" fill="#10b981" rx="0.5"/>
          <line x1="18" y1="48" x2="18" y2="64" stroke="#10b981" stroke-width="0.8"/><rect x="14" y="52" width="8" height="12" fill="#10b981" rx="0.5"/>
          <line x1="29" y1="46" x2="29" y2="60" stroke="#ef4444" stroke-width="0.8"/><rect x="25" y="50" width="8" height="8" fill="#ef4444" rx="0.5"/>
          <line x1="40" y1="44" x2="40" y2="62" stroke="#ef4444" stroke-width="0.8"/><rect x="36" y="48" width="8" height="14" fill="#ef4444" rx="0.5"/>
          <line x1="51" y1="30" x2="51" y2="48" stroke="#10b981" stroke-width="0.8"/><rect x="47" y="34" width="8" height="12" fill="#10b981" rx="0.5"/>
          <line x1="62" y1="14" x2="62" y2="34" stroke="#10b981" stroke-width="0.8"/><rect x="58" y="18" width="8" height="16" fill="#10b981" rx="0.5"/>
          <line x1="73" y1="16" x2="73" y2="32" stroke="#ef4444" stroke-width="0.8"/><rect x="69" y="20" width="8" height="8" fill="#ef4444" rx="0.5"/>
          <path d="M3,72 C16,64 30,57 44,52 C56,42 66,28 78,20" fill="none" stroke="#f59e0b" stroke-width="1.3" opacity="0.8"/>
          <rect x="3" y="70" width="8" height="8" fill="#10b981" opacity="0.35" rx="0.5"/>
          <rect x="14" y="66" width="8" height="12" fill="#10b981" opacity="0.35" rx="0.5"/>
          <rect x="25" y="72" width="8" height="6" fill="#ef4444" opacity="0.35" rx="0.5"/>
          <rect x="36" y="68" width="8" height="10" fill="#ef4444" opacity="0.35" rx="0.5"/>
          <rect x="47" y="63" width="8" height="15" fill="#10b981" opacity="0.35" rx="0.5"/>
          <rect x="58" y="60" width="8" height="18" fill="#10b981" opacity="0.35" rx="0.5"/>
          <rect x="69" y="70" width="8" height="8" fill="#ef4444" opacity="0.35" rx="0.5"/>
        </svg>
      </div>
      <div style="padding:7px 10px;height:68px;display:flex;flex-direction:column;justify-content:flex-start;align-items:center;text-align:center">
        <div style="color:rgba(200,220,255,0.9);font-size:7.5px;font-weight:600;letter-spacing:0.12em;margin-bottom:3px;min-height:18px">RETURN — WITH CGN</div>
        <div style="font-size:19px;font-weight:700;font-family:Georgia,serif;color:#f59e0b">33.6%</div>
        <div style="font-size:8px;margin-top:3px;color:#10b981">+23.4 pp vs no derivative</div>
        <div style="height:2px;margin-top:5px;border-radius:1px;background:#1a3a5c"><div style="width:100%;height:100%;border-radius:1px;background:#f59e0b"></div></div>
      </div>
    </div>
    <!-- Col 3: Portfolio weights -->
    <div style="border-right:1px solid #1a3a5c;display:flex;flex-direction:column">
      <div style="height:82px;position:relative;overflow:hidden;border-bottom:1px solid #0d2a4a">
        <svg width="100%" height="82" viewBox="0 0 80 82" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
          <rect width="80" height="82" fill="#020c1b"/>
          <text x="40" y="10" fill="rgba(100,160,220,0.6)" font-size="6" font-family="monospace" text-anchor="middle">PORTFOLIO WEIGHTS</text>
          <text x="6" y="24" fill="rgba(200,220,255,0.6)" font-size="5.5" font-family="monospace">EQ</text>
          <rect x="20" y="18" width="54" height="7" fill="#0d2a4a" rx="1"/><rect x="20" y="18" width="19" height="7" fill="#4a9eff" rx="1" opacity="0.8"/>
          <text x="41" y="24" fill="rgba(74,158,255,0.8)" font-size="5.5" font-family="monospace">35%</text>
          <text x="6" y="36" fill="rgba(200,220,255,0.6)" font-size="5.5" font-family="monospace">BD</text>
          <rect x="20" y="30" width="54" height="7" fill="#0d2a4a" rx="1"/><rect x="20" y="30" width="5" height="7" fill="#a855f7" rx="1" opacity="0.8"/>
          <text x="27" y="36" fill="rgba(168,85,247,0.8)" font-size="5.5" font-family="monospace">10%</text>
          <text x="6" y="48" fill="rgba(245,158,11,0.8)" font-size="5.5" font-family="monospace">CGN</text>
          <rect x="20" y="42" width="54" height="7" fill="#0d2a4a" rx="1"/><rect x="20" y="42" width="30" height="7" fill="#f59e0b" rx="1" opacity="0.85"/>
          <text x="52" y="48" fill="rgba(245,158,11,0.9)" font-size="5.5" font-family="monospace">55%</text>
          <rect x="6" y="56" width="68" height="10" fill="rgba(16,185,129,0.12)" rx="2"/>
          <text x="40" y="63" fill="#10b981" font-size="6" font-family="monospace" text-anchor="middle">P(r &lt; H) ≤ α ✓</text>
          <text x="6" y="76" fill="rgba(100,160,220,0.5)" font-size="5.5" font-family="monospace">No deriv.</text>
          <rect x="36" y="70" width="40" height="5" fill="#0d2a4a" rx="1"/><rect x="36" y="70" width="12" height="5" fill="#4a9eff" rx="1" opacity="0.7"/>
          <text x="50" y="76" fill="rgba(74,158,255,0.7)" font-size="5.5" font-family="monospace">10.2%</text>
        </svg>
      </div>
      <div style="padding:7px 10px;height:68px;display:flex;flex-direction:column;justify-content:flex-start;align-items:center;text-align:center">
        <div style="color:rgba(200,220,255,0.9);font-size:7.5px;font-weight:600;letter-spacing:0.12em;margin-bottom:3px;min-height:18px">RETURN — NO DERIV.</div>
        <div style="font-size:19px;font-weight:700;font-family:Georgia,serif;color:#4a9eff">10.2%</div>
        <div style="font-size:8px;margin-top:3px;color:rgba(150,180,220,0.55)">H=-10%, α=5%, λ=3.795</div>
        <div style="height:2px;margin-top:5px;border-radius:1px;background:#1a3a5c"><div style="width:31%;height:100%;border-radius:1px;background:#4a9eff"></div></div>
      </div>
    </div>
    <!-- Col 4: Real app screenshot -->
    <div style="display:flex;flex-direction:column">
      <div style="height:82px;overflow:hidden;border-bottom:1px solid #0d2a4a;display:flex;align-items:center;justify-content:center">
        <svg width="80" height="82" viewBox="-10 -10 100 100" xmlns="http://www.w3.org/2000/svg"><rect x="-10" y="-10" width="100" height="100" fill="#020c1b"/><path d="M 40.0 8.0 A 32 32 0 1 1 30.1 70.4 L 34.1 58.1 A 19 19 0 1 0 40.0 21.0 Z" fill="#e63946" stroke="#020c1b" stroke-width="1.5"/><text x="81.5" y="46.6" fill="#e63946" font-size="7" font-weight="700" font-family="monospace" text-anchor="middle" dominant-baseline="central">55%</text><path d="M 30.1 70.4 A 32 32 0 0 1 14.1 21.2 L 24.6 28.8 A 19 19 0 0 0 34.1 58.1 Z" fill="#f4a261" stroke="#020c1b" stroke-width="1.5"/><text x="0.1" y="53.0" fill="#f4a261" font-size="7" font-weight="700" font-family="monospace" text-anchor="middle" dominant-baseline="central">30%</text><path d="M 14.1 21.2 A 32 32 0 0 1 40.0 8.0 L 40.0 21.0 A 19 19 0 0 0 24.6 28.8 Z" fill="#2a9d8f" stroke="#020c1b" stroke-width="1.5"/><text x="20.9" y="2.6" fill="#2a9d8f" font-size="7" font-weight="700" font-family="monospace" text-anchor="middle" dominant-baseline="central">15%</text><text x="40" y="38" fill="rgba(255,255,255,0.7)" font-size="6" font-family="monospace" text-anchor="middle">Portfolio</text><text x="40" y="46" fill="rgba(255,255,255,0.5)" font-size="5.5" font-family="monospace" text-anchor="middle">weights</text><rect x="2" y="68" width="7" height="7" fill="#e63946" rx="1"/><text x="11" y="74" fill="rgba(200,220,255,0.8)" font-size="6" font-family="monospace">CGN</text><rect x="28" y="68" width="7" height="7" fill="#f4a261" rx="1"/><text x="37" y="74" fill="rgba(200,220,255,0.8)" font-size="6" font-family="monospace">EQ</text><rect x="52" y="68" width="7" height="7" fill="#2a9d8f" rx="1"/><text x="61" y="74" fill="rgba(200,220,255,0.8)" font-size="6" font-family="monospace">BD</text></svg>
      </div>
      <div style="padding:7px 10px;height:68px;display:flex;flex-direction:column;justify-content:flex-start;align-items:center;text-align:center">
        <div style="color:rgba(200,220,255,0.9);font-size:7.5px;font-weight:600;letter-spacing:0.12em;margin-bottom:3px;min-height:18px">DERIVATIVE TYPES</div>
        <div style="font-size:19px;font-weight:700;font-family:Georgia,serif;color:#a855f7">9+</div>
        <div style="font-size:8px;margin-top:3px;color:rgba(150,180,220,0.55)">puts · calls · CGNs · barrier</div>
        <div style="height:2px;margin-top:5px;border-radius:1px;background:#1a3a5c"><div style="width:85%;height:100%;border-radius:1px;background:#a855f7"></div></div>
      </div>
    </div>
  </div>
</div>
</div>
<div style="flex:1;overflow:hidden;height:148px;max-height:148px;background:#f0f4f8"><img src="data:image/jpeg;base64,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JEY5B8z8DB3k57Yr1zX9Ih13wvqGiznEd7byW7HGcbgRn9a8k0bwV441GTwVoXiLSLaw03wnOLj7fHdJL9vMcZji2RjlODk5xSW+onsbvxc8W6t4Zfw5Da6zF4c0i/vWi1PxJNHG8enIIyUyZAY4/MfC75OO3UitD4R+J9Z8UeCbm91a6XUI4dQmtrHV44PJXU7ZSPLuQvTDgnlfkOMrwRWT4v0HxXqnjLwj460/w8l3caFJdRy6LPdRgkSgKs0cn3RIAvftIa0/hZ4V1jw7H4k1PWbSHTpte1mXVBpUEokSyDxxptyOC7GMyPjjdIevWiPUH0PR6KKKYwoPSiigBp6VxfxJ8I3XjHwO2n6dcRQala3UN/ZSTZ8vzoZBIofHOCRg4rtqTA9KTQHl+h+GvGj+PNR+IPiGx0eHWDp406y0q0v5JIQm4OzPOYgckgcCPjHes3V/Bfjiw8f+IvEPhvTvD2qp4jsYra4g1W8khNk6qV+QrE/mx/NnZ+7+vNexYHpTWA2cCmwWh4vpt/aaD+zfrvhPwJ40tNR8Q+EtLlt5prcpJLBcRRlsGM5xnGBmvE9D+IXjXxx4zvn1TTLuy8X6TeR/Y47eG5kF+RFCDaREJ5UdsTJ5srscmOVH+Vfnr2Lxt8Lph8brbxf4djubSXWojZ3t7aDf9mmCHbJLH0khkUeXIp7hCMHmu5+HPgKLwX4csRdGGXWv7LtNP1C5tiwjufs6FI22HuASM4yRgHOBhLXVg9rIr+G/hRoXh/XVvxdahfQWrMdLsLyUSQ6UGxuWDjIHpknAJAwOK75I40UIihVAwABgCpcLS4FMSVhox6Uo60tFAwooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigBMUHrS0UAMHvXE+K/h1pXjTXLS48QXU1zpsBDHSjFF5M7A5Bd9vmEA4O3fjjpXc0UrARRxpFEscaBUUYAAwAKlHSiimKwUUUUDG0n4U+ilYBtIO1PopgFFFFABRRRQAUUUUAFIenNLRQAzC7qd3paKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD//2Q==" style="width:100%;height:148px;object-fit:cover;object-position:center top;opacity:0.95"/></div>
</div>''', unsafe_allow_html=True)

DONUT_COLORS = ['#e63946','#f4a261','#e9c46a','#2a9d8f','#264653','#023e8a','#e76f51','#457b9d']

def make_donut_svg(weights, labels, colors, size=160):
    """Generate an SVG doughnut chart for portfolio weights."""
    import math
    cx, cy, r_out, r_in = size//2, size//2, size//2 - 8, size//2 - 28
    
    # Filter zero weights
    items = [(w, l, c) for w, l, c in zip(weights, labels, colors) if w > 0.001]
    if not items:
        return ""
    
    total = sum(i[0] for i in items)
    
    def polar(cx, cy, r, angle_deg):
        a = math.radians(angle_deg - 90)
        return cx + r * math.cos(a), cy + r * math.sin(a)
    
    paths = []
    legend_items = []
    angle = 0
    
    for w, label, color in items:
        sweep = (w / total) * 360
        large = 1 if sweep > 180 else 0
        
        x1o, y1o = polar(cx, cy, r_out, angle)
        x2o, y2o = polar(cx, cy, r_out, angle + sweep)
        x1i, y1i = polar(cx, cy, r_in, angle + sweep)
        x2i, y2i = polar(cx, cy, r_in, angle)
        
        path = (f'<path d="M {x1o:.1f} {y1o:.1f} '
                f'A {r_out} {r_out} 0 {large} 1 {x2o:.1f} {y2o:.1f} '
                f'L {x1i:.1f} {y1i:.1f} '
                f'A {r_in} {r_in} 0 {large} 0 {x2i:.1f} {y2i:.1f} Z" '
                f'fill="{color}" stroke="#0d1117" stroke-width="1.5"/>')
        paths.append(path)

        pct = w / total * 100
        # Percentage label OUTSIDE the ring with security color
        if pct >= 5:
            mid_angle = angle + sweep / 2
            r_label = r_out + 14
            lx, ly = polar(cx, cy, r_label, mid_angle)
            paths.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" '
                f'fill="{color}" font-size="10" font-weight="700" '
                f'font-family="sans-serif" text-anchor="middle" dominant-baseline="central">'
                f'{pct:.0f}%</text>'
            )
        # Legend
        short_lbl = label[:12] + "…" if len(label) > 12 else label
        legend_items.append((color, short_lbl, pct))
        
        angle += sweep
    
    svg = (f'<svg width="{size+30}" height="{size+30}" viewBox="-15 -15 {size+30} {size+30}" '
           f'xmlns="http://www.w3.org/2000/svg">'
           + "".join(paths)
           + '</svg>')
    return svg

st.markdown("<div style='margin-top:2.5rem'></div>", unsafe_allow_html=True)
tab1,tab2,tab3=st.tabs(["📊 Optimiser","📖 About","📚 Glossary"])

with tab1:
    import os
    st.markdown("## Beyond Mean-Variance: Portfolio Optimiser with Derivatives & Structured Products — A Mental Accounts Framework")
    st.markdown(
        "Most portfolio optimisers stop at stocks and bonds. This app goes further — "
        "incorporating derivatives and structured products, handling **non-normal return distributions**, "
        "and optimising under a risk constraint you define: either the probability of loss below "
        "a threshold (**Value-at-Risk / VaR**) or the expected loss in the worst scenarios "
        "(**Expected Shortfall / ES**).")


    if not run_btn:
        st.markdown("""
<div class="info-box" style="color:#ffffff !important">

### 👈 How to use this tool

Follow these steps in the sidebar:

<table style="width:100%;border-collapse:collapse;color:#ffffff">
<tr style="border-bottom:1px solid #3a3a5a">
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">1</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem"><strong>Portfolio data</strong> — Choose a data source: default base case, live market tickers, manual entry, or CSV upload</td>
</tr>
<tr style="border-bottom:1px solid #3a3a5a">
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">2</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem"><strong>Derivative &amp; parameters</strong> — Select a derivative or structured product type and set its characteristics (strike, maturity, floor, participation, etc.)</td>
</tr>
<tr style="border-bottom:1px solid #3a3a5a">
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">3</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem"><strong>Constraint</strong> — Choose VaR or ES constraint type, set threshold H, and set α (VaR) or L (ES)</td>
</tr>
<tr style="border-bottom:1px solid #3a3a5a">
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">4</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem"><strong>Grid resolution</strong> — Choose Fast for a quick preview, High precision for thesis-level accuracy</td>
</tr>
<tr>
  <td style="padding:.5rem .4rem .5rem .8rem;white-space:nowrap"><span style="display:flex;align-items:center;gap:.4rem">Step <span style="display:inline-block;background:#ffffff;color:#0d1117;border-radius:50%;width:1.4rem;height:1.4rem;line-height:1.4rem;text-align:center;font-size:.9rem;font-weight:700">5</span></span></td>
  <td style="padding:.5rem .5rem .5rem .3rem"><strong>Run</strong> — Click <strong>▶ Run optimiser</strong></td>
</tr>
</table>

The chart will show three curves and two markers:

<table style="width:100%;border-collapse:collapse;color:#ffffff;margin-top:.5rem">
<tr><td colspan="2" style="padding:.3rem .5rem;font-weight:700;color:#1a6bbf;font-size:1.1rem">Curves</td></tr>
<tr style="border-bottom:1px solid #2a2a3a">
  <td style="padding:.3rem .5rem;white-space:nowrap">🟣 <strong>Purple dashed</strong></td>
  <td style="padding:.3rem .5rem">Classical mean-variance efficient frontier (Markowitz)</td>
</tr>
<tr style="border-bottom:1px solid #2a2a3a">
  <td style="padding:.3rem .5rem;white-space:nowrap">🔵 <strong>Blue</strong></td>
  <td style="padding:.3rem .5rem">Behavioural optimiser frontier without derivatives</td>
</tr>
<tr style="border-bottom:1px solid #2a2a3a">
  <td style="padding:.3rem .5rem;white-space:nowrap">🟡 <strong>Gold</strong></td>
  <td style="padding:.3rem .5rem">Behavioural optimiser frontier including your selected derivative</td>
</tr>
<tr><td colspan="2" style="padding:.5rem .5rem .3rem .5rem;font-weight:700;color:#1a6bbf;font-size:1.1rem">Markers</td></tr>
<tr style="border-bottom:1px solid #2a2a3a">
  <td style="padding:.3rem .5rem;white-space:nowrap">➡️ <strong>White dotted arrow</strong></td>
  <td style="padding:.3rem .5rem">Return gap between the behavioural frontier without and with the derivative, at the selected H and α constraint</td>
</tr>
<tr>
  <td style="padding:.3rem .5rem;white-space:nowrap">🟢 <strong>Green diamond</strong></td>
  <td style="padding:.3rem .5rem">Equivalence point: the unique portfolio where the mean-variance and behavioural approaches yield exactly the same result (here at λ=3.795, H=-10%, α=5%). To the right of this point, adding derivatives allows the behavioural approach to outperform mean-variance.</td>
</tr>
</table>

At the equivalence point (λ=3.795, H=-10%, α=5%), the grey and blue curves meet exactly —
confirming the MVT/MAT equivalence proven in Das, Markowitz, Scheid & Statman (2010).
The gold curve shows what the behavioural approach, with the right choice of derivatives or
structured products, can unlock beyond what mean-variance can achieve.

**Note on discrete vs continuous frontiers:** The behavioural frontiers are plotted at discrete constraint levels (H = -5%, -8%, -10%, -12%, -15%, -18%, -20%). Each point is the optimal portfolio for that specific mental-account threshold. The MV frontier is continuous as it is computed by sweeping the risk-aversion parameter λ — each MV portfolio corresponds to one behavioural portfolio via the MVT/MAT equivalence, demonstrating that both approaches converge to the same solution when no derivatives are present.

</div>
""", unsafe_allow_html=True)
        # Sample chart
        import os
        if os.path.exists("sample_output.png"):
            st.image("sample_output.png", caption="Sample output — default base case with Capital-Guaranteed Note (CGN)", use_container_width=True)

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

        # Three portfolio perspectives note
        st.markdown('''
<div style="background:#0d1a2e;border:1px solid #1a3a5c;border-radius:8px;padding:.8rem 1rem;margin-bottom:.8rem;color:#c0c8d8;font-size:.82rem">
<b style="color:#4a9eff">Three portfolios are generated as output of the optimisation:</b><br><br>
<b style="color:#10b981">Portfolio (1)</b> — Optimum portfolio without derivatives: identical to the Markowitz MV optimum, derived through the mental accounting framework (reference portfolio)<br>
<b style="color:#f59e0b">Portfolio (2)</b> — Optimum portfolio with derivative, same mental-accounting &amp; risk-aversion constraint (H, α ↔ λ): may reach higher expected returns by exploiting asymmetric derivative payoffs<br>
<b style="color:#e76f51">Portfolio (3)</b> — Portfolio with derivative and with the same variance as Portfolio (1): interpolated from the derivative frontier at equivalent risk level (see below)
</div>
''', unsafe_allow_html=True)

        with st.spinner("Rendering chart..."):
            # Compute Portfolio (3) point for chart overlay
            _p3_x, _p3_y = None, None
            if der_xs and len(der_xs) >= 2:
                try:
                    # nd_res may not be available yet — use equivalence point std dev from nd_xs
                    _key = f"H={H_val:.0%}"
                    if _key in nd_lbls:
                        _target_std = nd_xs[nd_lbls.index(_key)]
                    elif nd_xs:
                        _target_std = nd_xs[len(nd_xs)//2]  # middle point as fallback
                    else:
                        _target_std = None
                    if _target_std:
                        _fp = sorted(zip(der_xs, der_ys), key=lambda p: p[0])
                        _fx = [p[0] for p in _fp]
                        _fy = [p[1] for p in _fp]
                        if min(_fx) <= _target_std <= max(_fx):
                            _p3_x = _target_std
                            _p3_y = float(np.interp(_target_std, _fx, _fy))
                except Exception:
                    pass

            fig_plotly=plot_frontier_plotly(mv_x,mv_y,mv_eq,nd_xs,nd_ys,nd_lbls,
                                            der_xs,der_ys,der_lbls,der_label_sel,H_val,alpha_val,
                                            p3_x=_p3_x, p3_y=_p3_y)

            # ── Simulation summary + chart side by side ───────────────────────
            col_summary, col_chart = st.columns([1, 3.5])

            with col_summary:
                # Build derivative parameters string
                # Only show user-facing derivative parameters
                _SKIP_KEYS = {"type","underlying_index","S0","r","cgn_premium","vol"}
                _LABEL_MAP = {"T":"Maturity","floor":"Floor","participation":"Participation",
                              "cap":"Cap","K":"Strike","barrier":"Barrier","M":"Barrier M",
                              "call_strike":"Call strike","put_strike":"Put strike"}
                der_params_str = ""
                if der_config:
                    for k, v in der_config.items():
                        if k in _SKIP_KEYS or v is None:
                            continue
                        label = _LABEL_MAP.get(k, k.replace("_"," ").title())
                        if isinstance(v, float):
                            der_params_str += f"<br>• {label}: {v:.2%}" if abs(v) <= 5 else f"<br>• {label}: {v:.2f}"
                        else:
                            der_params_str += f"<br>• {label}: {v}"

                # Implied lambda
                lam_summary = "—"
                if not use_es:
                    _cov_s = corr_to_cov(sigs_in, corr_in)
                    _lam_s = implied_lambda(H_val, alpha_val, means_in, _cov_s)
                    if _lam_s is not None:
                        lam_summary = f"{_lam_s:.4f}"

                constraint_str = (
                    f"VaR — H={H_val:.0%}, α={_alpha:.0%}"
                    if not use_es else
                    f"ES — H={H_val:.0%}, L={_L:.0%}"
                )

                # Build HTML cleanly to avoid f-string quote issues
                _data_src = data_mode.split("(")[0].strip()
                _securities = ", ".join(names_in)
                _der_html = (
                    f'<span style="color:#f59e0b">{der_label_sel}</span>{der_params_str}'
                    if der_config else
                    '<span style="color:#8896a8">None</span>'
                )
                _resolution = grid_lbl.split("(")[0].strip()

                def _lbl(t): return f'<div style="color:#7fb3e8;font-size:.72rem;margin-bottom:.2rem">{t}</div>'
                def _val(v): return f'<div style="margin-bottom:.6rem">{v}</div>'

                _html = (
                    '<div style="background:#0d1a2e;border:1px solid #1a3a5c;border-radius:8px;min-height:560px;'
                    'padding:.8rem 1rem;color:#c0c8d8;font-size:.8rem">'
                    '<div style="color:#4a9eff;font-weight:700;font-size:.85rem;'
                    'margin-bottom:.6rem;border-bottom:1px solid #1a3a5c;padding-bottom:.4rem">'
                    '📌 Optimisation Parameters <span style="color:#556a8a;font-size:.65rem;font-weight:400">(summary)</span></div>'
                    + _lbl("DATA SOURCE") + _val(_data_src)
                    + _lbl("SECURITIES") + _val(_securities)
                    + _lbl("DERIVATIVE") + _val(_der_html)
                    + _lbl("CONSTRAINT") + _val(constraint_str)
                    + _lbl("IMPLIED λ")
                    + f'<div style="margin-bottom:.6rem;color:#10b981;font-weight:600">{lam_summary}</div>'
                    + _lbl("RESOLUTION") + _val(_resolution)
                    + '</div>'
                )
                st.markdown(_html, unsafe_allow_html=True)

            with col_chart:
                st.plotly_chart(fig_plotly, use_container_width=True)

        # ── Results ───────────────────────────────────────────────────────────────
        st.markdown("---")
        constraint_label = f"H={H_val:.0%}, α={_alpha:.0%}" if not use_es else f"H={H_val:.0%}, L={_L:.0%}"
        st.markdown(f"### Optimal portfolios — {constraint_label}")

        # ── Helper to render one portfolio column ────────────────────────────
        def _render_portfolio(border_color, header_html, caption_txt,
                              weights, labels, colors, stats,
                              delta_txt=None, method_txt=None, note_html=None):
            """Render one portfolio: header box, then metrics left / donut centre / bars right."""
            # Header box
            st.markdown(header_html, unsafe_allow_html=True)
            st.caption(caption_txt)
            # Three-column layout: metrics | donut | bars
            col_m, col_d, col_b = st.columns([1.2, 1, 1.4])
            with col_m:
                _mr1, _mr2 = st.columns(2)
                _mr1.metric("Expected return",
                            f"{stats['expected_return']*100:.2f}%",
                            delta=delta_txt)
                _mr2.metric("Std deviation", f"{stats['std_dev']*100:.2f}%")
                _mr3, _mr4 = st.columns(2)
                _mr3.metric("Skewness", f"{stats['skewness']:.3f}")
                _mr4.metric("Shortfall / ES", f"{stats['shortfall_stat']*100:.2f}%")
                if method_txt:
                    st.caption(f"Method: {method_txt}")
            with col_d:
                _svg = make_donut_svg(weights, labels, colors, size=150)
                if _svg:
                    st.markdown(f'<div style="display:flex;justify-content:center;margin-top:1.8rem">{_svg}</div>', unsafe_allow_html=True)
            with col_b:
                st.markdown('<div style="font-weight:600;font-size:.9rem;margin-bottom:.4rem">Portfolio weights</div>', unsafe_allow_html=True)
                for i, w in enumerate(weights):
                    _c = colors[i % len(colors)]
                    _l = labels[i]
                    st.markdown(
                        f'<div style="margin-bottom:.45rem">'
                        f'<div><span style="color:{_c};font-weight:600">{_l}</span>'
                        f'<span style="color:{_c}"> — {w*100:.1f}%</span></div>'
                        f'<div style="height:6px;background:#1a2a3a;border-radius:3px;margin-top:3px">'
                        f'<div style="height:6px;width:{w*100:.1f}%;background:{_c};border-radius:3px"></div>'
                        f'</div></div>',
                        unsafe_allow_html=True)
            if note_html:
                st.markdown(note_html, unsafe_allow_html=True)

        # ── Compute all three portfolios ─────────────────────────────────────
        nd_res = None
        dr_res = None
        p3_return = None
        p3_std = None

        try:
            nd_res, _ = run_opt(means_arr, sigs_arr, cov_mat, None, H_val, _alpha,
                                m_val, mp_val, constraint_type=_ctype, L=_L)
        except Exception:
            pass

        if der_config:
            try:
                dr_res, _ = run_opt(means_arr, sigs_arr, cov_mat, der_config,
                                    H_val, _alpha, m_val, mp_val,
                                    constraint_type=_ctype, L=_L)
            except Exception:
                pass

        # Compute Portfolio (3) by interpolation
        if nd_res and der_xs and len(der_xs) >= 2:
            try:
                _target_std = nd_res['std_dev'] * 100
                _fp = sorted(zip(der_xs, der_ys), key=lambda p: p[0])
                _fx = [p[0] for p in _fp]
                _fy = [p[1] for p in _fp]
                if min(_fx) <= _target_std <= max(_fx):
                    p3_std = _target_std
                    p3_return = float(np.interp(_target_std, _fx, _fy))
            except Exception:
                pass

        # ── Render Portfolio (1) and (2) side by side ────────────────────────
        with st.container():
            if nd_res:
                _nd_weights = nd_res["weights"]
                _nd_labels = [names_in[i] if i < len(names_in) else f"Asset {i+1}" for i in range(len(_nd_weights))]
                _nd_colors = [DONUT_COLORS[i % len(DONUT_COLORS)] for i in range(len(_nd_weights))]
                _method = ("Exhaustive grid search + COBYLA" if nd_res.get('method_used') == "grid_search"
                           else "Differential evolution + COBYLA" if nd_res.get('method_used') == "differential_evolution"
                           else nd_res.get('method_used', '—'))
                _render_portfolio(
                    border_color="#10b981",
                    header_html=(
                        '<div style="background:#0d1a2e;border:1px solid #10b981;border-radius:8px;'
                        'padding:.6rem 1rem;margin-bottom:.4rem;text-align:center">'
                        '<span style="color:#10b981;font-weight:700;font-size:.95rem">'
                        '<span style="color:#10b981;margin-right:.4rem">◆</span>Optimal portfolio (1) — no derivative</span></div>'
                    ),
                    caption_txt="Maximises return subject to the downside constraint — reference portfolio (equivalent to Markowitz MV optimum)",
                    weights=_nd_weights, labels=_nd_labels, colors=_nd_colors,
                    stats=nd_res, method_txt=_method)
            else:
                st.markdown("**Optimal portfolio (1) — no derivative**")
                st.warning("⚠️ No eligible portfolio found. Try increasing H or α, or use Fast resolution.")

        with st.container():
            if der_config:
                if dr_res:
                    _dr_weights = dr_res["weights"]
                    _dr_labels = [asset_labels[i] if i < len(asset_labels) else f"Asset {i+1}" for i in range(len(_dr_weights))]
                    _dr_colors = [DONUT_COLORS[i % len(DONUT_COLORS)] for i in range(len(_dr_weights))]
                    _delta = f"+{(dr_res['expected_return']-(nd_res['expected_return'] if nd_res else 0))*100:.2f}pp"
                    _method = ("Exhaustive grid search + COBYLA" if dr_res.get('method_used') == "grid_search"
                               else "Differential evolution + COBYLA" if dr_res.get('method_used') == "differential_evolution"
                               else dr_res.get('method_used', '—'))
                    _dr_ret = dr_res['expected_return']*100
                    _nd_ret = nd_res['expected_return']*100 if nd_res else 0
                    _p2_sign = "+" if _dr_ret >= _nd_ret else ""
                    _p2_diff = _dr_ret - _nd_ret
                    _p2_note = (
                        f'<div style="background:#1c1200;border:1px solid #f59e0b;border-radius:6px;'
                        f'padding:.6rem 1rem;color:#c0c8d8;font-size:.82rem;margin-top:.6rem">'
                        f'At the same mental-accounting constraint (H={H_val:.0%}, α={_alpha:.0%} ↔ λ), '
                        f'the optimum portfolio with <b style="color:#f59e0b">{der_label_sel}</b> '
                        f'achieves <b style="color:#f59e0b">{_dr_ret:.2f}%</b> expected return '
                        f'vs <b>{_nd_ret:.2f}%</b> for portfolio (1) without derivatives — '
                        f'a <b style="color:{"#10b981" if _p2_diff>=0 else "#ef4444"}">{_p2_sign}{_p2_diff:.2f} pp '
                        f'{"gain" if _p2_diff>=0 else "reduction"}</b> '
                        f'(note: higher return may come with higher variance).</div>'
                    )
                    _render_portfolio(
                        border_color="#f59e0b",
                        header_html=(
                            f'<div style="background:#0d1a2e;border:1px solid #f59e0b;border-radius:8px;'
                            f'padding:.6rem 1rem;margin-bottom:.4rem;text-align:center">'
                            f'<span style="color:#f59e0b;font-weight:700;font-size:.95rem">'
                            f'<span style="display:inline-block;width:12px;height:12px;background:#ff6b00;border:2px solid white;margin-right:.4rem;vertical-align:middle"></span>Optimal portfolio (2) — with {der_label_sel}</span></div>'
                        ),
                        caption_txt=f"Same mental-accounting & risk-aversion constraint (H={H_val:.0%}, α={_alpha:.0%} ↔ λ) — results may vary",
                        weights=_dr_weights, labels=_dr_labels, colors=_dr_colors,
                        stats=dr_res, delta_txt=f"{_p2_sign}{_p2_diff:.2f}pp vs portfolio (1)",
                        method_txt=_method, note_html=_p2_note)
                else:
                    st.markdown(f"**Optimal portfolio (2) — with {der_label_sel}**")
                    st.warning("⚠️ No eligible portfolio found with this derivative. Try different parameters.")
            else:
                st.info("Select a derivative in the sidebar to see Portfolio (2).")

        # ── Portfolio (3) — full width below ─────────────────────────────────
        if der_config and nd_res and p3_return is not None:
            st.markdown("---")
            _gain3 = p3_return - nd_res['expected_return'] * 100
            # Use derivative weights as approximation (closest frontier point)
            if dr_res:
                _p3_weights = dr_res["weights"]
                _p3_labels = [asset_labels[i] if i < len(asset_labels) else f"Asset {i+1}" for i in range(len(_p3_weights))]
                _p3_colors = [DONUT_COLORS[i % len(DONUT_COLORS)] for i in range(len(_p3_weights))]
                _p3_donut = make_donut_svg(_p3_weights, _p3_labels, _p3_colors, size=150)
            _gain3_sign = "+" if _gain3 >= 0 else ""
            _gain3_word = "gain" if _gain3 >= 0 else "reduction"
            _gain3_color = "#10b981" if _gain3 >= 0 else "#ef4444"
            _p3_interp_note = (
                f'<div style="background:#1a0a00;border:1px solid #e76f51;border-radius:6px;'
                f'padding:.6rem 1rem;color:#c0c8d8;font-size:.82rem;margin-top:.4rem">'
                f'At the <b style="color:#e76f51">same variance as portfolio (1)</b> ({p3_std:.1f}% std dev), '
                f'the derivative frontier achieves <b style="color:#e76f51">{p3_return:.2f}%</b> expected return '
                f'vs <b>{nd_res["expected_return"]*100:.2f}%</b> without derivatives — '
                f'a <b style="color:{_gain3_color}">{_gain3_sign}{_gain3:.2f} pp {_gain3_word}</b> '
                f'(indicative — interpolated from derivative frontier, not directly optimised).</div>'
            )
            _gain3_sign = "+" if _gain3 >= 0 else ""
            if dr_res:
                _p3_stats = {
                    'expected_return': p3_return / 100,
                    'std_dev': p3_std / 100,
                    'skewness': dr_res.get('skewness', 0),
                    'shortfall_stat': dr_res.get('shortfall_stat', 0)
                }
                _render_portfolio(
                    border_color="#e76f51",
                    header_html=(
                        f'<div style="background:#0d1a2e;border:1px solid #e76f51;border-radius:8px;'
                        f'padding:.6rem 1rem;margin-bottom:.4rem;text-align:center">'
                        f'<span style="color:#e76f51;font-weight:700;font-size:.95rem">'
                        f'<svg width="14" height="14" viewBox="0 0 24 24" style="margin-right:.4rem;vertical-align:middle" xmlns="http://www.w3.org/2000/svg"><polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26" fill="#e76f51" stroke="white" stroke-width="1.5"/></svg>Optimal portfolio (3) — same variance as Portfolio (1), with {der_label_sel}'
                        f'</span> <span style="color:#c0c8d8;font-size:.78rem">(interpolated)</span>'
                        f'</div>'
                    ),
                    caption_txt=f"Interpolated from the derivative frontier at the same std deviation as portfolio (1) — indicative only",
                    weights=_p3_weights, labels=_p3_labels, colors=_p3_colors,
                    stats=_p3_stats,
                    delta_txt=f"{_gain3_sign}{_gain3:.2f}pp vs portfolio (1)",
                    method_txt="Interpolated from derivative frontier — weights shown are from the closest optimised frontier point",
                    note_html=_p3_interp_note)
        elif der_config and nd_res and len(der_xs) >= 2:
            st.markdown("---")
            st.markdown(
                f'<div style="background:#0d1a2e;border:1px solid #e76f51;border-radius:8px;'
                f'padding:.8rem 1rem;color:#c0c8d8;font-size:.85rem">'
                f'<b style="color:#e76f51">Portfolio (3) — not available for this derivative</b><br><br>'
                f'Portfolio (3) requires the no-derivative portfolio std dev (<b>{nd_res["std_dev"]*100:.1f}%</b>) '
                f'to fall within the derivative frontier range '
                f'(<b>{min(der_xs):.1f}%–{max(der_xs):.1f}%</b>). '
                f'With a <b>{der_label_sel}</b>, the derivative portfolio always carries higher variance than the '
                f'no-derivative portfolio — so no same-variance comparison is possible at this constraint level.<br><br>'
                f'<b>To see Portfolio (3):</b> try a <b>put option</b> or <b>collar</b>, which have lower variance '
                f'impact and whose frontier may overlap with the no-derivative portfolio range.</div>',
                unsafe_allow_html=True)
        elif der_config and nd_res and len(der_xs) < 2:
            st.markdown("---")
            st.info(f"Portfolio (3) not available — derivative frontier has {len(der_xs)} point(s). Try Standard resolution.")

        # ── How to read these results ────────────────────────────────────────
        if der_config and nd_res:
            st.markdown("---")
            st.markdown(
                '<div style="background:#ffffff;border:1px solid #1a6bbf;border-radius:6px;'
                'padding:.8rem 1rem;margin-top:.5rem;color:#111111;font-size:.85rem">'
                '<b style="color:#1a3a6b">📌 How to read these results</b><br>'
                'Portfolio (1) and (2) are compared at the <b>same mental-accounting & risk-aversion constraint</b> '
                f'(H={H_val:.0%}, α={_alpha:.0%} — same risk-aversion λ). '
                'Depending on the derivative chosen, portfolio (2) may achieve a higher or lower expected return '
                'and may show higher variance. '
                'Portfolio (3) shows the derivative frontier return at the <b>same variance as Portfolio (1)</b>, '
                'providing a complementary risk-adjusted perspective.</div>',
                unsafe_allow_html=True)




    # Always visible — portfolio data and contact
    st.markdown("---")
    show_portfolio_data(names_in, means_in, sigs_in, corr_in)

    st.markdown("---")

    # LinkedIn + contact
    # About the author with photo
    import base64 as _b64mod
    _photo_html = ""
    if os.path.exists("profile.jpeg"):
        with open("profile.jpeg","rb") as _pf:
            _pb64 = _b64mod.b64encode(_pf.read()).decode()
        _photo_html = (
            f'<a href="https://www.linkedin.com/in/sami-jeddou-25787a404" target="_blank" style="text-decoration:none">'

            f'<div style="position:relative;display:inline-block;width:80px;margin-right:1rem;vertical-align:top">'

            f'<img src="data:image/jpeg;base64,{_pb64}" style="width:80px;border-radius:6px;display:block"/>'

            f'<div style="position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,0.55);'

            f'color:#ffffff;font-size:.6rem;text-align:center;padding:2px 0;border-radius:0 0 6px 6px">Sami Jeddou</div>'

            f'</div></a>'
        )
    st.markdown(
        f'''<div style="background:#0f1923;border:1px solid #1a6bbf;border-radius:8px;padding:1rem 1.4rem;color:#ffffff">
<b>👤 About the author</b><br><br>
<div style="display:flex;align-items:flex-start;gap:1rem">
{_photo_html}
<div>
<b>Sami Jeddou</b><br>
Senior Financial Services Executive — Transformation, Risk &amp; Capital Markets<br><br>
🔗 <a href="https://www.linkedin.com/in/sami-jeddou-25787a404" target="_blank" style="color:#7fb3e8">Connect on LinkedIn</a> &nbsp;&nbsp;|&nbsp;&nbsp;
🐙 <a href="https://github.com/SamiJeddou/behavioral-portfolio-optimizer" target="_blank" style="color:#7fb3e8">View source on GitHub</a>
</div>
</div>
</div>''',
        unsafe_allow_html=True)

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
                        data={"name": sender_name, "email": sender_email, "message": message},
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

    st.markdown("---")
    st.markdown("""
<div style="background:#0f1923;border:1px solid #1a6bbf;border-radius:8px;padding:1rem 1.4rem;color:#ffffff">

**📬 Get in touch**

Interested in collaborating, discussing an opportunity, or learning more about this work?
Use the contact form in the **Optimiser tab**, or connect directly:

🔗 [LinkedIn](https://www.linkedin.com/in/sami-jeddou-25787a404) &nbsp;&nbsp;|&nbsp;&nbsp; 📧 sami.jeddou@protonmail.com

</div>
""", unsafe_allow_html=True)

with tab3:
    st.markdown("## 📚 AI Glossary & Reference")
    st.markdown(
        "Click any term below for an AI-generated explanation, or type your own question. "
        "Answers are tailored to the context of behavioural portfolio optimisation.")
    st.info("💡 After clicking a term or submitting a question, **scroll down** to see the answer at the bottom of this page.", icon="👇")

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
            f'<div style="background:#0f1923;border:1px solid #1a6bbf;border-radius:8px;'
            f'padding:1rem 1.2rem;color:#c0c8d8;font-size:.9rem;line-height:1.6">'
            f'{st.session_state["glossary_response"]}</div>',
            unsafe_allow_html=True)
        if st.button("Clear response"):
            st.session_state["glossary_response"] = ""
            st.session_state["glossary_term"] = ""
            st.rerun()
