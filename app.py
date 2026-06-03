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
<div style="width:100%;background:#020c1b;padding:4px 0 0 0;margin-bottom:0;display:flex;align-items:stretch">
<div style="flex:1;overflow:hidden;min-height:148px"><img src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAGrAUADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD7LooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAoqNpOyAufUdKZicoS7oMdl4/MnNAE9FQow3Hah/3m6Gnbdw+Z8j0HFADi6g43c+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" style="width:100%;height:100%;object-fit:cover;object-position:right center;opacity:0.88"/></div>
<div style="max-width:900px;flex-shrink:0;background:linear-gradient(135deg,#020c1b 0%,#071428 40%,#0a1a35 70%,#020c1b 100%);border-radius:0;overflow:hidden;border:none;display:flex;align-items:stretch;font-family:monospace;margin-bottom:0">
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
<div style="flex:1;overflow:hidden;min-height:148px"><img src="data:image/jpeg;base64,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" style="width:100%;height:100%;object-fit:cover;object-position:left center;opacity:0.88"/></div>
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
