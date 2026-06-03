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
        line=dict(color='#a855f7', width=2, dash='dash'),
        hovertemplate='<b>Mean-Variance Efficient Frontier (Markowitz)</b><br>Std Dev: %{x:.2f}%<br>Expected Return: %{y:.2f}%<extra></extra>'
    ))

    # ── Behavioral — no derivative ────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=nd_x, y=nd_y, mode='lines+markers',
        name='Behavioural efficient frontier — no derivative',
        line=dict(color='#1a6bbf', width=2.5),
        marker=dict(size=9, color='#1a6bbf', symbol='circle'),
        text=nd_lbls,
        hovertemplate='<b>Behavioral (no derivative)</b><br>Threshold: %{text}<br>Std Dev: %{x:.2f}%<br>Expected Return: %{y:.2f}%<extra></extra>'
    ))

    # ── Behavioral — with derivative ──────────────────────────────────────────
    if der_x:
        fig.add_trace(go.Scatter(
            x=der_x, y=der_y, mode='markers',
            name=f'Portfolio (2) — Behavioural optimal portfolios — with {der_label}',
            marker=dict(size=10, color='#f59e0b', symbol='square'),
            text=der_lbls,
            hovertemplate=f'<b>Behavioural optimal portfolio (with {der_label})</b><br>Threshold: %{{text}}<br>Std Dev: %{{x:.2f}}%<br>Expected Return: %{{y:.2f}}%<extra></extra>'
        ))

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
            name='Portfolio (1) — Equivalence point: MV = Behavioural (no derivatives) where λ=3.795 ↔ H=-10%, α=5%',
            marker=dict(size=13, color='#10b981', symbol='diamond',
                        line=dict(width=0)),
            showlegend=True,
            legendrank=4,
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
            name=f'Portfolio (3) — same variance as (1), with {der_label}',
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
st.markdown(f'''
<div style="max-width:900px;margin:0 auto;background:linear-gradient(135deg,#020c1b 0%,#071428 40%,#0a1a35 70%,#020c1b 100%);border-radius:12px;overflow:hidden;border:1px solid #1a3a5c;display:flex;align-items:stretch;font-family:monospace;margin-bottom:1rem">
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
      <div style="padding:7px 10px;flex:1;display:flex;flex-direction:column;justify-content:center;text-align:center">
        <div style="color:rgba(200,220,255,0.9);font-size:7.5px;font-weight:600;letter-spacing:0.12em;margin-bottom:3px">LIVE MARKET DATA</div>
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
      <div style="padding:7px 10px;flex:1;display:flex;flex-direction:column;justify-content:center;text-align:center">
        <div style="color:rgba(200,220,255,0.9);font-size:7.5px;font-weight:600;letter-spacing:0.12em;margin-bottom:3px">RETURN — WITH CGN</div>
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
      <div style="padding:7px 10px;flex:1;display:flex;flex-direction:column;justify-content:center;text-align:center">
        <div style="color:rgba(200,220,255,0.9);font-size:7.5px;font-weight:600;letter-spacing:0.12em;margin-bottom:3px">RETURN — NO DERIV.</div>
        <div style="font-size:19px;font-weight:700;font-family:Georgia,serif;color:#4a9eff">10.2%</div>
        <div style="font-size:8px;margin-top:3px;color:rgba(150,180,220,0.55)">H=-10%, α=5%, λ=3.795</div>
        <div style="height:2px;margin-top:5px;border-radius:1px;background:#1a3a5c"><div style="width:31%;height:100%;border-radius:1px;background:#4a9eff"></div></div>
      </div>
    </div>
    <!-- Col 4: Real app screenshot -->
    <div style="display:flex;flex-direction:column">
      <div style="height:82px;overflow:hidden;border-bottom:1px solid #0d2a4a">
        <img src="data:image/png;base64,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59nB1x6reexY1i/b5bX5NRfpfpV6HbRWUdPQdHukKCGuqdUvtFPV02oZatr2sgc9wc2Rga3G1xdgktLS3JWtWTofprPDrenvE+mdWVNp09PWsfab5JigmY4tO8Nj7Txg9h2By/HIzFV4v39ul/QvHHfbs1y+rhqLr9y/R317b7RR19RUWLizTUkVRRNrC6poesvayJ07A3sjc5udpcRnuWdWfo26utlZSi537SppTdo7ZWup7nl1I9/NvEywBpcC3aD2svZkDKt4vaZ329PmEnSnNutf3cSRdb6UOiuz6S6e7doOkvba6111bSw7o6lklVTsle1jmy4YGsk5lwGCNpafet313+j/AGP9bBojQ9BriC+GdwiuOoYo47TUxsiL3iKWOPc5/djljk7KzFcTETG/2/lqabTMTp+9j82oux2v9HfV1xq6uKn1Ho3q9NPHSdcN2/1eWreP9lY4MO6YcgW4xk4yTnH3TX6OWvL1TcWSv07aJPaU1s4FyrnRSGoj/qNAYQ4u5lu0nIBPIK3ieez5jxjizOm3nbPtPhLjaL9BWvoApafQNp1BdL7aXXh+phbaq3S3QRQysbJw3U7HNjLusFwPIHk3JxkLFvXQBfr/AK+1PDpunsmm7Pb7k2gpo7jdHuZJO6NruDFIWFz3cweYH3gMnBTNGbLG21/T5gpm8X53/Dg6Lp9f0I6qtmg5tY3i6aetVFE6qiNPWVxjqHz08jmOgY3bh8jix20NJBA5kLmCt9bNTExtEREQREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERBcaHvn6s6zsuo+q9b9l18NZwOJs4vDeH7d2DjOMZwceC7dD+lLfjWyCt07DUW7/ALQ4VO2sLJG9ZcSzdJsIcIg5zRhrcg+5fnlFMSIxKPt1axr57fRaZyzeObP0A79I+Sp6OqLT9wsFwluNJQMt/FgvL4qSojaMB0tOGkOJA2ubnDgTzHIC01r+lDT32jpqWm0TUwMiutDcyJbyXta6nexxijZwsMYdgHLHMkkEnC/NaK1flVmnbe/fE39WaYimMsbLW7tjtF96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rSeorlcWW+ntU0c7w8g1RFPGAxjnvLnylrW4a1xOSO4rLdoPVApxUsoIJqcueBPDWwSRYY17nO3teW7MRSYfnaSxwBJBC2SLpUq31jY6qK5w0MU8skM1uuT6etZvD25MhDo87XuB2xtznkWq0p+m2sjrLg8WlsNNWVGWxwGJksEO2U4bKI8mQSyGUOIwHbsAbuUmarbN3nf4dLU3clg75P8Agyf4HKqW0akujL3qe73iOkbRsrZKioEDXbhGHBx25wM9/fgLV1mvWIuzVERNoERFzQREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBEW66cuFgissFK+Z0ZZWU0tQJoWjiP3O3HO4ksaMe7/FysRdJlpSK81ZNJMKAVdWyrr2QOFRK2YS5+0cWgvBIJDSPfyGB7sKkaWDO5rj+w4/8ARRZfFaMBdSUbWgkmMgAd57blXZi+B/nHorOKR0UNFLCXMcxm5pB5giR3NdcPaPV1vr2tLnUNSABkkxO5f3LGAJIABJPIAK4k1TqKSN0cl5rXMcC1wMpwQfcqiN7o3tewlrmkEEe4r04kYV4yTPf/AC3Vk/0sg264AEmhqgB3nhO9F4QxSTTMhhjfJJI4NYxgy5xPIAAd5Vu/VWo3sLHXquLXDBBlPMLDsNXFQXygrpmvdFTVMcrwzBcQ1wJxn38laows0RRM2335ljGmKaZnD1krLPd6KCaestddTRQVJpJny07mNjnAJMTiRyeACdp58u5YK/QdZ03aPuNfHJU6cqY2GvdWve+BkoNQaeWIVJaJGEv5wHaHDG1xDgQM40PSvoSKn3TWuaeWS7wV00cNoZFHmOqhkL4w6odt3RMeC129xc49trTheaiqZmInS9u7Zfn92Tvtzt55i/BUXTZekezzanst3rdOU1e+KljprnUTQlsz275BKI2skETswvDQXtJGBjGAVt1u6Vuj+kt0dMbPcnwRVtMYqZ9FE/bBTzU5jcHcUAOMMLg4FriXvOHhpOdRttKxrLhtTQ1lNS0tVUU0sUFWxz6eRzcNla1xaS0+/DgR+5Yy7jpLpX0pDV09x1FSV9RWxUfV58UEUkdQ01c0jmbRIz/unxtGTtG0gtcA1YlB0o6YNLQ0dZb6uMUVPTxUdS23wSOopG28wSTNaXDe7j7ZACRkAHIIASNkzw89q4kRRVaJvHHwcZVpp3T951DPPBZqCSskgi4soYQNrcho5kgZJcAB3kkAZXQ9Y6+03cujyOwUVK2orBMJKiaptIilqZBM55qOIyoLWOc0hpHDccZG7GMbVV9NunqaullstHUUUEggEbKS2NgfFEyrikML3GoeHgRMka3aGAF55AHsonXvt5bee+znXMxTeI5vz7XcR1JYLvp24i33midS1JZva3e14c3JGQ5pIIy0jke8FYlyoau2181BX00lNVQPLJYpG4cxw9xC6/qTpO01erfWthN0tVwnpo45K1ttgnfUtD6gvgdmQbWO4sRLgSTs5jkM3dD0xaDp55ZmaeLIn3aerlhfaWSyTNfMXseH9Yaxj2swzBY/lyBw44xFVVovHOnP6b0mjNv4c89+jgTaeodSvq2wSmnje2N8oYdjXuDi1pPcCQ1xA9+0+BWXfLNcrJUw011pTTTzQMqGxue0uDHjLS4AnaSOeDg4I5c10TVnSXQXzo/qrG19xpq+p6hNUPbSxCOqmhieyXeQ8FoJMbg4Ak7MEBX1f0yWevucLpY7iyB80jp5pKKGWSP/AFGGGF7QXdoxzsfKGkgfdPf3bjW/V58yzGsx137nDVm11qudBDxq6gqaVnHfTnjRlhErA0vYQeYcA5uR+IXWLl0q2NkM1NbKWr+2FV1qodQQRuq5XUEUMUxAcdh6xG6UgHlkHtHkrr/TDo+bUFZcp6WvqJ6iqrKinqq22MkNGJuq7GBrKhpcGiGVuQ9pw8HHNzVqYtv3cxz+nSKad8868+jg1FS1FbWwUdJC+aonkbFFGwZc97jgAfiSVnz6dv8ABM+KSzV+5lO+qO2nc4cBpLXTAgYMYII3js8u9dDn6QrRcekHR9TDCLfZ7ZI3jxtpWwRRyumeXTMY1zjhrXR4BcSNmOeMm2pulfTtFoqLSlVaquapp7NPaH18Bjdlj2ODmMOebDII3h37eXjzvPnPpOvjbxcIrm8aaWieyb7PC/fDi1LBLU1MVNAwyTSvDI2jvc4nAH8VK4UdTb6+ooK2F0NTTSuhmjd3se0kOB/YQQu66j6X9H8BzbFQ3BlXinibVmgijeYWVDpHMJMrjnhOLMjaCcgBrcBfavpb0JW3iquLLZc7fHLVNmkpBbqedtTAJJjJTOc54LRNxGSPd2iHbhzDWYkVzMXtbWfa08/zuL5pjdp7355jhFDS1FdWwUVHC+epqJGxRRsGXPe44AA8SSvSrt9dRzRRVlJPTPlbujEzCzc3cW7hnGRkEZ7sg+C7lH0xaEDbfUDS1VDXxSUstTLHTQ7XudPFJW4G7mHCFjWZ79zs7ffgUnSnYpLRTwSXLUNLVxWL2VDC23wS00chkfmYl07S77NwaAQA08+eAtTMxGka/q/ro1Frazzf41cauVFVW241FvroXQVVNK6KaN3ex7Tgj+IWOtn6UrhR3TXdxqqGYVEDeFA2Zp5TGKJkbpB4hxYXA/itZSmbxEo+IvqKj4i+og+IvqIPiL6iD4i+og+IvqIPiL6iD4i+og+IvqIJwd8n/Bk/wOVUranxukyDjgyf4HKszF8D/OPRYxNwginmL4H+ceiZi+B/nHouYginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H+ceiZi+B/nHogginmL4H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uLj94k5JJ96w0RRRERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAXpTRslnbHJPHA1x5ySBxa39u0E/wC80VGwWKjoG3IxVDaK50YaJKmoa6ZjaeMHtEZDDu7sZBBJAwSV6V1vt7LHJw6XbLFRwVQqd7t0hkeAWEZ24Ad7gDlp5qmt9zuVu3+z7hV0fExv4Ezo92O7ODz7yoTV9dNSillrKiSna4vETpSWhxzzx3Z5n+JSdhDHREUBERAREQEREBERAREQEREBelNGyadsclRHTtPfJIHFrf27QT/AAC80VF1a6Ki61V075aa4N6jNKyWIytEb2sc4feDSTy94I5rKulvoYrLJwqURzU8NJL1je4mUys3OaQTt5E8sAfdOcqlt9yuNuLzb6+qpDIAHmCZzN2PHB5qM1dWzUkdJLVzyU8ZzHE6QljT4gdw7z/FJ2c9aQx0RFFEREBERAREQEREBERAREQEREGx6Wt9DU08RqqUVBqq9lIXF7gYGuaSXjaR2v8AxZHZPJa4RgkZyvelraykZKylq54GzN2yCOQtDx4HHeOZ/ivBWdpGwREUBERAREQEREBERAREQEREBERBslBb6CeO0yupG5fT1MkzA932zomuc0HnkZwAduFXaip6eCrgfTQiBk9NFMYmuJEZc3JA3EnHvGSe9eT7xd3xwRvulc5lMQ6BrqhxERAwC0Z5Y/BYtTUT1U7p6maSaV5y58ji5x/aSrPVzz7JDzREUUREQEREBERAREQEREBERAREQbXTUVtlgpH1Frip6jgz1Ap2ySHjRMiLmF+XZG5wP3duRk4HIqm1FT09PXRmmiELJqaKYxBxIjL2BxAJJOMnlkk4PvUZr3epuDxrvcJOA4Oi31LzwyBgFuTyOPBYdTPNUzvnqJpJpXnLnvcXOcfxJVq1nTnnnaRo80RFAREQEREBERAREQEREBERAREVG3w2+1T2yC5MoKV5hiqHTRQvmDHOYxpax+92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0vHBm2To01PfdOUF5s1K2rbVvmHDMrItjWOa1p3PcA5z3bwGjJPDd3+63tnQzqmroaGWXq1NU1NxFHJDJVQAU7SIucjjKNsmZmgQkB558lQnpG1f1SajZcKWGlmgbAaeG3U0cTWB0jhsY2MNY7M0h3NAd2jz7lmu6W9eufvN2pCeP1n/ANlUmBPknigcLlJknt/e/FWb3551S2m3n9c8ELl0Xavp5J5KS3dbomSbY5hPC15aXRiPfHvLo3OE0RDHYdh47wCVCx6EdV8OG53aK21k91fa6anFO+oMs0YbxOcWe4vjaMZDi48wBz+0/SnrmCWOSO70+WBow63UzmuLTEWvc0x4c8GCHD3ZcNg5968LH0g6hsdBSwWuSlinp3VX281HDOXMn4e9mJGOA+4eY5kPI7uSUxMbdeeb6b+rVVrGmk+W/wBNO3qX1T0P3KmuVPQT6iszZHSyx1D2mV0cBZTmoHa2YdmIbuXIZAOFT0+gzUaWjv0N7ppI5KKpqWxCCRrt0HCL2ZcBkYlI3tyNzC38VCPpL1ix8TzcKOR0VO+naZbXSydh8TYnZ3RnLjGxrC49raMZwo3TpAvl1s9RQ3FlDLJJSRUEU8VHFTmClY8PMTRE1oILmRnLs4DSBjcVmYqtNp7PP3tPOti14v3+Pxzw1FERaQREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQf/Z" style="width:100%;height:100%;object-fit:cover;object-position:center top;opacity:0.92"/>
      </div>
      <div style="padding:7px 10px;flex:1;display:flex;flex-direction:column;justify-content:center;text-align:center">
        <div style="color:rgba(200,220,255,0.9);font-size:7.5px;font-weight:600;letter-spacing:0.12em;margin-bottom:3px">DERIVATIVE TYPES</div>
        <div style="font-size:19px;font-weight:700;font-family:Georgia,serif;color:#a855f7">9+</div>
        <div style="font-size:8px;margin-top:3px;color:rgba(150,180,220,0.55)">puts · calls · CGNs · barrier</div>
        <div style="height:2px;margin-top:5px;border-radius:1px;background:#1a3a5c"><div style="width:85%;height:100%;border-radius:1px;background:#a855f7"></div></div>
      </div>
    </div>
  </div>
</div>
''', unsafe_allow_html=True)

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
        
        # Legend
        pct = w / total * 100
        short_lbl = label[:12] + "…" if len(label) > 12 else label
        legend_items.append((color, short_lbl, pct))
        
        angle += sweep
    
    svg = (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
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
<b style="color:#4a9eff">Three optimum portfolios are generated as output of the optimisation:</b><br><br>
<b style="color:#10b981">Portfolio (1)</b> — Without derivatives: identical to the Markowitz MV optimum, derived through the mental accounting framework (reference portfolio)<br>
<b style="color:#f59e0b">Portfolio (2)</b> — With derivative, same mental-accounting &amp; risk-aversion constraint (H, α ↔ λ): may reach higher expected returns by exploiting asymmetric derivative payoffs<br>
<b style="color:#e76f51">Portfolio (3)</b> — With derivative, same variance as Portfolio (1): interpolated from the derivative frontier at equivalent risk level (see below)
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
        def _render_portfolio_col(title, caption_txt, weights, labels, colors, stats,
                                   delta_txt=None, method_txt=None, is_interp=False,
                                   interp_note=None):
            st.markdown(f'**{title}**', unsafe_allow_html=True)
            st.caption(caption_txt)
            st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Expected return", f"{stats['expected_return']*100:.2f}%" if not is_interp else f"{stats['expected_return']:.2f}%",
                      delta=delta_txt)
            m2.metric("Std deviation (risk)", f"{stats['std_dev']*100:.2f}%" if not is_interp else f"{stats['std_dev']:.2f}%")
            if not is_interp:
                m3.metric("Skewness", f"{stats['skewness']:.3f}")
                m4.metric("Shortfall / ES", f"{stats['shortfall_stat']*100:.2f}%")
            else:
                m3.metric("Return gain", delta_txt or "—")
                m4.metric("Source", "Interpolated")
            # When no delta, add a spacer matching the delta label height (~28px) so weights align
            if delta_txt is None:
                st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
            st.markdown('<div style="margin-top:.4rem;margin-bottom:.3rem;font-weight:600;font-size:.9rem">Portfolio weights</div>', unsafe_allow_html=True)
            _svg = make_donut_svg(weights, labels, colors, size=150)
            if _svg:
                st.markdown(f'<div style="display:flex;justify-content:center;margin-bottom:.5rem">{_svg}</div>', unsafe_allow_html=True)
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
            if method_txt:
                st.caption(f"Optimisation method: {method_txt}")
            if interp_note:
                st.markdown(interp_note, unsafe_allow_html=True)

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
        c1 = st.container()
        c2 = st.container()

        with c1:
            if nd_res:
                _nd_weights = nd_res["weights"]
                _nd_labels = [names_in[i] if i < len(names_in) else f"Asset {i+1}" for i in range(len(_nd_weights))]
                _nd_colors = [DONUT_COLORS[i % len(DONUT_COLORS)] for i in range(len(_nd_weights))]
                _method = ("Exhaustive grid search + COBYLA" if nd_res.get('method_used') == "grid_search"
                           else "Differential evolution + COBYLA" if nd_res.get('method_used') == "differential_evolution"
                           else nd_res.get('method_used', '—'))
                st.markdown(
                    '<div style="background:#0d1a2e;border:1px solid #10b981;border-radius:8px;'
                    'padding:.6rem 1rem;margin-bottom:.6rem">'
                    '<span style="color:#10b981;font-weight:700;font-size:.95rem">'
                    'Optimal portfolio (1) — no derivative</span></div>',
                    unsafe_allow_html=True)
                _render_portfolio_col(
                    title="",
                    caption_txt="Maximises return subject to the downside constraint — reference portfolio",
                    weights=_nd_weights, labels=_nd_labels, colors=_nd_colors,
                    stats=nd_res, method_txt=_method)
            else:
                st.markdown("**Optimal portfolio (1) — no derivative**")
                st.warning("⚠️ No eligible portfolio found. Try increasing H or α, or use Fast resolution.")

        with c2:
            if der_config:
                if dr_res:
                    _dr_weights = dr_res["weights"]
                    _dr_labels = [asset_labels[i] if i < len(asset_labels) else f"Asset {i+1}" for i in range(len(_dr_weights))]
                    _dr_colors = [DONUT_COLORS[i % len(DONUT_COLORS)] for i in range(len(_dr_weights))]
                    _delta = f"+{(dr_res['expected_return']-(nd_res['expected_return'] if nd_res else 0))*100:.2f}pp"
                    _method = ("Exhaustive grid search + COBYLA" if dr_res.get('method_used') == "grid_search"
                               else "Differential evolution + COBYLA" if dr_res.get('method_used') == "differential_evolution"
                               else dr_res.get('method_used', '—'))
                    st.markdown(
                        f'<div style="background:#0d1a2e;border:1px solid #f59e0b;border-radius:8px;'
                        f'padding:.6rem 1rem;margin-bottom:.6rem">'
                        f'<span style="color:#f59e0b;font-weight:700;font-size:.95rem">'
                        f'Optimal portfolio (2) — with {der_label_sel}</span></div>',
                        unsafe_allow_html=True)
                    _render_portfolio_col(
                        title="",
                        caption_txt=f"Same mental-accounting & risk-aversion constraint (H={{H_val:.0%}}, α={{_alpha:.0%}} ↔ λ) — results may vary",
                        weights=_dr_weights, labels=_dr_labels, colors=_dr_colors,
                        stats=dr_res, delta_txt=_delta, method_txt=_method)
                else:
                    st.markdown(f"**Optimal portfolio (2) — with {der_label_sel}**")
                    st.warning("⚠️ No eligible portfolio found with this derivative. Try different parameters.")
            else:
                st.info("Select a derivative in the sidebar to see Portfolio (2).")

        # ── Portfolio (3) — full width below ─────────────────────────────────
        if der_config and nd_res and p3_return is not None:
            st.markdown("---")
            st.markdown(
                f'<div style="background:#0d1a2e;border:1px solid #e76f51;border-radius:8px;'
                f'padding:.6rem 1rem;margin-bottom:.6rem">'
                f'<span style="color:#e76f51;font-weight:700;font-size:.95rem">'
                f'Optimal portfolio (3) — same variance as Portfolio (1), with {der_label_sel}'
                f'</span> <span style="color:#c0c8d8;font-size:.78rem">(interpolated from derivative frontier)</span>'
                f'</div>',
                unsafe_allow_html=True)
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
                f'<div style="background:#0d1a2e;border:1px solid #e76f51;border-radius:6px;'
                f'padding:.6rem 1rem;color:#c0c8d8;font-size:.82rem;margin-top:.4rem">'
                f'At the <b style="color:#e76f51">same variance as portfolio (1)</b> ({p3_std:.1f}% std dev), '
                f'the derivative frontier achieves <b style="color:#e76f51">{p3_return:.2f}%</b> expected return '
                f'vs <b>{nd_res["expected_return"]*100:.2f}%</b> without derivatives — '
                f'a <b style="color:{_gain3_color}">{_gain3_sign}{_gain3:.2f} pp {_gain3_word}</b> '
                f'(indicative — interpolated from derivative frontier, not directly optimised).</div>'
            )
            c3a, c3b = st.columns([1, 2])
            _gain3_sign = "+" if _gain3 >= 0 else ""
            with c3a:
                m1p3, m2p3, m3p3 = st.columns(3)
                m1p3.metric("Expected return", f"{p3_return:.2f}%",
                            delta=f"{_gain3_sign}{_gain3:.2f}pp vs portfolio (1)")
                m2p3.metric("Std deviation", f"{p3_std:.2f}%",
                            help="Same as portfolio (1) — the controlled variable")
                m3p3.metric("Return vs portfolio (1)", f"{_gain3_sign}{_gain3:.2f}pp",
                            help="Positive: derivative adds return at same risk. Negative: derivative reduces return at same risk.")
                st.markdown(_p3_interp_note, unsafe_allow_html=True)
            with c3b:
                if dr_res and _p3_donut:
                    st.caption("Method: interpolated from derivative frontier — weights shown are the closest optimised frontier point")
                    st.markdown('<div style="font-weight:600;font-size:.9rem;margin-bottom:.3rem">Portfolio weights</div>', unsafe_allow_html=True)
                    col_svg, col_bars = st.columns([1, 1])
                    with col_svg:
                        st.markdown(f'<div style="display:flex;justify-content:center">{_p3_donut}</div>', unsafe_allow_html=True)
                    with col_bars:
                        for i, w in enumerate(_p3_weights):
                            _c = _p3_colors[i % len(_p3_colors)]
                            _l = _p3_labels[i]
                            st.markdown(
                                f'<div style="margin-bottom:.45rem">'
                                f'<div><span style="color:{_c};font-weight:600">{_l}</span>'
                                f'<span style="color:{_c}"> — {w*100:.1f}%</span></div>'
                                f'<div style="height:6px;background:#1a2a3a;border-radius:3px;margin-top:3px">'
                                f'<div style="height:6px;width:{w*100:.1f}%;background:{_c};border-radius:3px"></div>'
                                f'</div></div>',
                                unsafe_allow_html=True)
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
