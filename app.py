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

def generate_pdf_report(constraint_label, nd_res, dr_res, p3_return, p3_std,
                         nd_labels, nd_weights, nd_colors,
                         dr_labels, dr_weights, dr_colors,
                         der_label_sel, H_val, _alpha, use_es, _L,
                         data_mode, names_in, grid_lbl, lam_summary,
                         fig_plotly=None, fig_png=None):
    """Generate a PDF summary report of optimisation results."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable, Image as RLImage)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
    from reportlab.platypus.frames import Frame
    import io, datetime

    # ── Page number canvas callback ───────────────────────────────────────────
    class _NumberedDoc(SimpleDocTemplate):
        def handle_pageEnd(self):
            self.canv.saveState()
            self.canv.setFont('Helvetica', 8)
            self.canv.setFillColor(colors.grey)
            self.canv.drawCentredString(
                A4[0] / 2, 1.2*cm,
                f"Page {self.canv.getPageNumber()} — Beyond Mean-Variance Portfolio Optimiser | Jeddou (2026)"
            )
            self.canv.restoreState()
            super().handle_pageEnd()

    buf = io.BytesIO()
    doc = _NumberedDoc(buf, pagesize=A4,
                       leftMargin=2*cm, rightMargin=2*cm,
                       topMargin=2*cm, bottomMargin=2.5*cm)

    styles = getSampleStyleSheet()
    navy   = colors.HexColor('#0d1a2e')
    blue   = colors.HexColor('#1a6bbf')
    green  = colors.HexColor('#10b981')
    gold   = colors.HexColor('#f59e0b')
    coral  = colors.HexColor('#e76f51')
    light  = colors.HexColor('#c0c8d8')
    white  = colors.white

    title_style = ParagraphStyle('Title2', parent=styles['Title'],
                                  textColor=navy, fontSize=16, spaceAfter=4)
    h1_style    = ParagraphStyle('H1', parent=styles['Heading1'],
                                  textColor=blue, fontSize=12, spaceAfter=4)
    h2_style    = ParagraphStyle('H2', parent=styles['Heading2'],
                                  textColor=navy, fontSize=10, spaceAfter=2)
    body_style  = ParagraphStyle('Body2', parent=styles['Normal'],
                                  fontSize=9, spaceAfter=3)
    caption_style = ParagraphStyle('Caption', parent=styles['Normal'],
                                    fontSize=8, textColor=colors.grey, spaceAfter=2)

    def section_header(text, color=blue):
        return [
            Paragraph(text, ParagraphStyle('SH', parent=styles['Heading2'],
                       textColor=white, backColor=color, fontSize=10,
                       spaceBefore=8, spaceAfter=4, leftIndent=-6, rightIndent=-6,
                       borderPadding=(4,6,4,6))),
        ]

    def weights_table(labels, weights, colors_list, title):
        data = [[title, 'Weight', 'Bar']]
        for i, (lbl, w) in enumerate(zip(labels, weights)):
            bar = '█' * int(w * 20) + '░' * (20 - int(w * 20))
            data.append([lbl, f'{w*100:.1f}%', bar])
        t = Table(data, colWidths=[6*cm, 2*cm, 6*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), navy),
            ('TEXTCOLOR',  (0,0), (-1,0), white),
            ('FONTSIZE',   (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), white]),
            ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
            ('ALIGN', (1,0), (1,-1), 'CENTER'),
            ('FONTNAME', (2,1), (2,-1), 'Courier'),
            ('FONTSIZE', (2,1), (2,-1), 7),
            ('TEXTCOLOR', (2,1), (2,-1), blue),
        ]))
        return t

    def metrics_table(res, is_interp=False, interp_ret=None, interp_std=None, gain=None):
        if is_interp:
            data = [
                ['Metric', 'Value'],
                ['Expected return (interpolated)', f'{interp_ret:.2f}%'],
                ['Std deviation', f'{interp_std:.2f}%'],
                ['Return vs Portfolio (1)', f'{gain:+.2f} pp'],
            ]
        else:
            data = [
                ['Metric', 'Value'],
                ['Expected return', f"{res['expected_return']*100:.2f}%"],
                ['Std deviation',   f"{res['std_dev']*100:.2f}%"],
                ['Skewness',        f"{res['skewness']:.3f}"],
                ['Shortfall / ES',  f"{res['shortfall_stat']*100:.2f}%"],
            ]
        t = Table(data, colWidths=[8*cm, 4*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), navy),
            ('TEXTCOLOR',  (0,0), (-1,0), white),
            ('FONTSIZE',   (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), white]),
            ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
            ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ]))
        return t

    story = []

    # ── Title ─────────────────────────────────────────────────────────────────
    story.append(Paragraph('Portfolio Optimisation Report', title_style))
    story.append(Paragraph(
        'Beyond Mean-Variance: Portfolio Optimiser with Derivatives &amp; Structured Products — A Mental Accounts Framework',
        ParagraphStyle('Sub', parent=styles['Normal'], fontSize=9, textColor=colors.grey)))
    story.append(Paragraph(
        f'Generated: {datetime.datetime.now().strftime("%d %B %Y, %H:%M")} | '
        f'Constraint: {constraint_label} | Data: {data_mode.split("(")[0].strip()}',
        caption_style))
    story.append(HRFlowable(width='100%', thickness=1, color=blue, spaceAfter=8))

    # ── Simulation parameters ─────────────────────────────────────────────────
    story += section_header('Simulation Parameters', navy)
    params = [
        ['Parameter', 'Value'],
        ['Data source', data_mode.split('(')[0].strip()],
        ['Securities', ', '.join(names_in)],
        ['Derivative', der_label_sel if der_label_sel else 'None'],
        ['Constraint type', 'Expected Shortfall' if use_es else 'Value-at-Risk (VaR)'],
        ['Threshold H', f'{H_val:.0%}'],
        ['Shortfall prob α' if not use_es else 'Tail limit L', f'{_alpha:.0%}' if not use_es else f'{_L:.0%}'],
        ['Implied risk-aversion λ', lam_summary],
        ['Grid resolution', grid_lbl.split('(')[0].strip()],
    ]
    t = Table(params, colWidths=[7*cm, 7*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), navy),
        ('TEXTCOLOR',  (0,0), (-1,0), white),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), white]),
        ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # ── Optimisation chart ────────────────────────────────────────────────────
    _chart_bytes = fig_png if fig_png is not None else (
        fig_plotly.to_image(format='png', width=700, height=400, scale=2)
        if fig_plotly is not None else None
    )
    if _chart_bytes is not None:
        try:
            _img_buf = io.BytesIO(_chart_bytes)
            _rl_img = RLImage(_img_buf, width=17*cm, height=9.7*cm)
            story += section_header('Optimisation Chart — Efficient Frontiers', navy)
            story.append(_rl_img)
            story.append(Paragraph(
                'Purple dashed: MV frontier (Markowitz) · Blue: Behavioural frontier without derivatives · '
                'Gold squares: Behavioural frontier with derivatives · '
                'Green diamond: Portfolio (1) · Orange square: Portfolio (2) · Coral star: Portfolio (3)',
                caption_style))
            story.append(Spacer(1, 8))
        except Exception as _chart_err:
            story.append(Paragraph(f'Chart export unavailable: {_chart_err}', caption_style))

    # ── Portfolio (1) ─────────────────────────────────────────────────────────
    if nd_res:
        story += section_header('Portfolio (1) — Optimum without derivatives', green)
        story.append(metrics_table(nd_res))
        story.append(Spacer(1, 4))
        story.append(weights_table(nd_labels, nd_weights, nd_colors, 'Security'))
        story.append(Spacer(1, 8))

    # ── Portfolio (2) ─────────────────────────────────────────────────────────
    if dr_res and der_label_sel:
        story += section_header(f'Portfolio (2) — Optimum with {der_label_sel}', gold)
        if nd_res:
            delta = (dr_res['expected_return'] - nd_res['expected_return']) * 100
            story.append(Paragraph(
                f'Return vs Portfolio (1): <b>{delta:+.2f} pp</b> at same constraint (H={H_val:.0%}, alpha={_alpha:.0%})',
                body_style))
        story.append(metrics_table(dr_res))
        story.append(Spacer(1, 4))
        story.append(weights_table(dr_labels, dr_weights, dr_colors, 'Security / Instrument'))
        story.append(Spacer(1, 8))

    # ── Portfolio (3) ─────────────────────────────────────────────────────────
    if p3_return is not None and nd_res:
        story += section_header(f'Portfolio (3) — Same variance as Portfolio (1), with {der_label_sel}', coral)
        story.append(Paragraph(
            'Interpolated from the derivative frontier — indicative only.',
            caption_style))
        gain = p3_return - nd_res['expected_return'] * 100
        story.append(metrics_table(None, is_interp=True,
                                   interp_ret=p3_return, interp_std=p3_std, gain=gain))
        story.append(Spacer(1, 8))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        'This report is for educational and research purposes only. '
        'It does not constitute financial advice or investment recommendations. '
        'Built on Das &amp; Statman (2009), Das, Markowitz, Scheid &amp; Statman (2010) JFQA, and Jeddou (2012) USI Lugano.',
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=7, textColor=colors.grey)))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
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
        # Download with group_by='ticker' to get consistent multi-ticker structure
        raw_full = yf.download(tickers, start=str(start), end=str(end),
                               auto_adjust=True, progress=False,
                               group_by='column')

        # Handle both single and multi-ticker cases robustly
        if raw_full.empty:
            return None, None, None, None, None, "No data returned — check tickers and date range.", {}

        # Extract Close prices — handle MultiIndex columns from newer yfinance
        if isinstance(raw_full.columns, pd.MultiIndex):
            # Multi-ticker: columns are (field, ticker)
            if 'Close' in raw_full.columns.get_level_values(0):
                raw = raw_full['Close'].copy()
            elif 'Adj Close' in raw_full.columns.get_level_values(0):
                raw = raw_full['Adj Close'].copy()
            else:
                raw = raw_full.xs('Close', axis=1, level=0) if 'Close' in raw_full.columns.get_level_values(0) else raw_full.iloc[:, :len(tickers)]
        else:
            # Single ticker or flat columns
            if 'Close' in raw_full.columns:
                raw = raw_full[['Close']].copy()
                raw.columns = [tickers[0]]
            elif 'Adj Close' in raw_full.columns:
                raw = raw_full[['Adj Close']].copy()
                raw.columns = [tickers[0]]
            else:
                raw = raw_full.copy()

        # Ensure DataFrame
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(tickers[0])

        # Reorder columns to match requested ticker order
        available = [t for t in tickers if t in raw.columns]
        if not available:
            return None, None, None, None, None, f"No Close price data found for tickers: {tickers}", {}
        raw = raw[available].copy()

        raw = raw.dropna(how='all').dropna(axis=1, how='all')
        if raw.empty or len(raw) < 5:
            return None, None, None, None, None, "Insufficient data after cleaning — try a wider date range.", {}

        if freq == "Monthly":
            raw = raw.resample('ME').last()

        rets = raw.pct_change().dropna()
        if rets.empty or len(rets) < 3:
            return None, None, None, None, None, "Insufficient return data after cleaning.", {}

        rets, cleaning_report = clean_returns(rets.copy())
        factor = 252 if freq == "Daily" else 12
        means = (rets.mean() * factor).tolist()
        sigs  = (rets.std() * np.sqrt(factor)).tolist()
        corr  = rets.corr().values.tolist()
        names = list(rets.columns)
        last_prices = {col: float(raw[col].dropna().iloc[-1]) for col in raw.columns if not raw[col].dropna().empty}
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
                         p3_x=None, p3_y=None,
                         nd_res_actual=None, lam_actual=None):
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
        if der_x and der_y and der_lbls:
            try:
                _target_h = f'H={H_sel:.0%}'
                if _target_h in der_lbls:
                    _i2 = der_lbls.index(_target_h)
                else:
                    _i2 = 0
                    _best = float('inf')
                    for _ii, _lbl in enumerate(der_lbls):
                        try:
                            _hv = float(_lbl.replace('H=','').replace('%','')) / 100
                            if abs(_hv - H_sel) < _best:
                                _best = abs(_hv - H_sel)
                                _i2 = _ii
                        except Exception:
                            pass
                fig.add_trace(go.Scatter(
                    x=[der_x[_i2]], y=[der_y[_i2]], mode='markers',
                    name=f'Portfolio (2) — optimum with {der_label} at H={H_sel:.0%}',
                    legendrank=6,
                    marker=dict(size=14, color='#ff6b00', symbol='square',
                               line=dict(color='white', width=1.5)),
                    hovertemplate=(f'<b>Portfolio (2)</b><br>Optimum with {der_label}<br>'
                                  f'Std Dev: %{{x:.2f}}%<br>Expected Return: %{{y:.2f}}%<extra></extra>')
                ))
                fig.add_annotation(
                    x=der_x[_i2], y=der_y[_i2],
                    ax=-90, ay=-70,
                    xref='x', yref='y', axref='pixel', ayref='pixel',
                    showarrow=True, arrowhead=2, arrowsize=1.0,
                    arrowwidth=1.5, arrowcolor='#ff6b00',
                    text=(f'<b>Portfolio (2)</b><br>'
                          f'Optimum with {der_label}<br>'
                          f'H={H_sel:.0%}, same constraint as (1)<br>'
                          f'Return = {der_y[_i2]:.1f}%'),
                    font=dict(color='#ff6b00', size=9),
                    bgcolor='rgba(13,17,23,0.9)',
                    bordercolor='#ff6b00', borderwidth=1,
                    align='right', xanchor='right'
                )
            except Exception:
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

    # ── Portfolio (1) / Equivalence point ───────────────────────────────────────
    # Use actual optimised Portfolio (1) point if available, else fall back to default
    if nd_res_actual:
        _p1_x = nd_res_actual['std_dev'] * 100
        _p1_y = nd_res_actual['expected_return'] * 100
        _lam_str = f"λ={lam_actual:.4f}" if lam_actual else "λ computed"
        _h_str = f"H={H_sel:.0%}, α={alpha:.0%}"
        fig.add_trace(go.Scatter(
            x=[_p1_x], y=[_p1_y], mode='markers',
            name=f'Portfolio (1) — Optimum without derivatives ({_h_str})',
            legendrank=5,
            marker=dict(size=13, color='#10b981', symbol='diamond',
                        line=dict(width=0)),
            showlegend=True,
            hovertemplate=f'<b>Portfolio (1)</b><br>Optimum without derivatives<br>{_h_str} ↔ {_lam_str}<br>Std Dev: %{{x:.2f}}%<br>Expected Return: %{{y:.2f}}%<extra></extra>'
        ))
        fig.add_annotation(
            x=_p1_x, y=_p1_y,
            ax=80, ay=70,
            xref='x', yref='y', axref='pixel', ayref='pixel',
            showarrow=True, arrowhead=2, arrowcolor='#10b981',
            arrowwidth=1.5,
            text=f'<b>Portfolio (1)</b><br>Optimum — no derivative<br>{_h_str} ↔ {_lam_str}<br>Return = {_p1_y:.1f}%',
            font=dict(color='#10b981', size=9),
            bgcolor='rgba(13,17,23,0.9)',
            bordercolor='#10b981', borderwidth=1,
            align='left', xanchor='left'
        )
    elif mv_eq:
        fig.add_trace(go.Scatter(
            x=[mv_eq[0]], y=[mv_eq[1]], mode='markers',
            name='Portfolio (1) — Equivalence point: MV = Behavioural (no derivatives) ↔ H=-10%, α=5%',
            legendrank=5,
            marker=dict(size=13, color='#10b981', symbol='diamond',
                        line=dict(width=0)),
            showlegend=True,
            hovertemplate='<b>Portfolio (1) — Equivalence point</b><br>MV = Behavioural (no derivatives)<br>where λ=3.795 ↔ H=-10%, α=5%<br>Std Dev: %{x:.2f}%<br>Expected Return: %{y:.2f}%<extra></extra>'
        ))
        fig.add_annotation(
            x=mv_eq[0], y=mv_eq[1],
            ax=80, ay=70,
            xref='x', yref='y', axref='pixel', ayref='pixel',
            showarrow=True, arrowhead=2, arrowcolor='#10b981',
            arrowwidth=1.5,
            text=f'Portfolio (1) — Equivalence point<br>MV = Behavioural (no derivatives)<br>λ=3.795 ↔ H=-10%, α=5%<br>Return = {mv_eq[1]:.1f}%',
            font=dict(color='#10b981', size=9),
            bgcolor='rgba(13,17,23,0.9)',
            bordercolor='#10b981', borderwidth=1,
            align='left', xanchor='left'
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
        fig.add_annotation(
            x=p3_x, y=p3_y,
            ax=90, ay=-70,
            xref='x', yref='y', axref='pixel', ayref='pixel',
            showarrow=True, arrowhead=2, arrowsize=1.0,
            arrowwidth=1.5, arrowcolor='#e76f51',
            text=(f'<b>Portfolio (3)</b><br>'
                  f'Same variance as (1)<br>'
                  f'({p3_x:.1f}% std dev)<br>'
                  f'Return = {p3_y:.1f}% (interpolated)'),
            font=dict(color='#e76f51', size=9),
            bgcolor='rgba(13,17,23,0.9)',
            bordercolor='#e76f51', borderwidth=1,
            align='left', xanchor='left'
        )

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

    reset_btn=st.button(
        "↩  Reset / New Simulation",
        type="secondary",
        use_container_width=True)

    if run_btn:
        # Run clicked — clear old results and mark as needing fresh computation
        st.session_state['_run_active'] = True
        st.session_state['_needs_compute'] = True
        st.session_state.pop('_cached_results', None)
        st.session_state.pop('_pdf_bytes', None)
        st.session_state.pop('_fig_png', None)

    if reset_btn:
        for _k in ['_run_active','_needs_compute','_cached_results',
                   '_pdf_bytes','_fig_png','_fig_plotly']:
            st.session_state.pop(_k, None)
        st.rerun()

    # Only show results if explicitly run — not on slider/widget reruns
    _run_active = st.session_state.get('_run_active', False)
    # On fresh page load session_state may have _run_active=True from previous session
    # Detect this: if _needs_compute is not set and no cached results exist → fresh load
    _has_results = st.session_state.get('_cached_results') is not None
    _needs_compute = st.session_state.get('_needs_compute', False)
    if _run_active and not _has_results and not _needs_compute:
        # Stale session state from previous session — reset
        st.session_state.pop('_run_active', None)
        _run_active = False


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
<div style="width:100%;background:#020c1b;padding:0;margin-bottom:0">
</div>
<div style="max-width:900px;margin:0 auto;background:linear-gradient(135deg,#020c1b 0%,#071428 40%,#0a1a35 70%,#020c1b 100%);border-radius:0;overflow:hidden;border:none;display:flex;align-items:stretch;font-family:monospace;margin-bottom:0">
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
        <div style="font-size:19px;font-weight:700;font-family:Georgia,serif;color:#10b981">10,000+</div>
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
</div>
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
    st.markdown('<h2 style="color:#4a9eff">Beyond Mean-Variance: Portfolio Optimiser with Derivatives &amp; Structured Products — A Mental Accounts Framework</h2>', unsafe_allow_html=True)
    st.markdown(
        "Most portfolio optimisers stop at stocks and bonds. This app goes further — "
        "incorporating derivatives and structured products, handling **non-normal return distributions**, "
        "and optimising under a risk constraint you define: either the probability of loss below "
        "a threshold (**Value-at-Risk / VaR**) or the expected loss in the worst scenarios "
        "(**Expected Shortfall / ES**).")


    if not _run_active:
        st.markdown("""
<div class="info-box" style="color:#ffffff !important">

<span style="color:#4a9eff;font-size:1.75rem;font-weight:700">👈 How to use this tool</span>

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

The chart shows the efficient frontiers and up to three portfolio markers (see sample output at the bottom of this section):

<table style="width:100%;border-collapse:collapse;color:#ffffff;margin-top:.5rem">
<tr><td colspan="2" style="padding:.3rem .5rem;font-weight:700;color:#1a6bbf;font-size:1.1rem">Curves</td></tr>
<tr style="border-bottom:1px solid #2a2a3a">
  <td style="padding:.3rem .5rem;white-space:nowrap">🟣 <strong>Purple dashed</strong></td>
  <td style="padding:.3rem .5rem">Mean-variance efficient frontier (Markowitz)</td>
</tr>
<tr style="border-bottom:1px solid #2a2a3a">
  <td style="padding:.3rem .5rem;white-space:nowrap">🔵 <strong>Blue dots</strong></td>
  <td style="padding:.3rem .5rem">Behavioural efficient frontier — no derivative (each dot is the optimum portfolio for one H constraint level)</td>
</tr>
<tr style="border-bottom:1px solid #2a2a3a">
  <td style="padding:.3rem .5rem;white-space:nowrap">🟡 <strong>Gold squares</strong></td>
  <td style="padding:.3rem .5rem">Behavioural optimum portfolios — derivative frontier (one point per H level, with the selected derivative)</td>
</tr>
<tr><td colspan="2" style="padding:.5rem .5rem .3rem .5rem;font-weight:700;color:#1a6bbf;font-size:1.1rem">Portfolio markers</td></tr>
<tr style="border-bottom:1px solid #2a2a3a">
  <td style="padding:.3rem .5rem;white-space:nowrap">🟢 <strong>Green diamond</strong></td>
  <td style="padding:.3rem .5rem"><strong>Portfolio (1)</strong> — Equivalence point: the unique portfolio where mean-variance and behavioural approaches yield exactly the same result. At H=-10%, α=5%, the implied risk-aversion is λ=3.795. Both curves meet here, confirming the MVT/MAT equivalence (Das, Markowitz, Scheid &amp; Statman, 2010).</td>
</tr>
<tr style="border-bottom:1px solid #2a2a3a">
  <td style="padding:.3rem .5rem;white-space:nowrap">🟠 <strong>Orange square (white frame)</strong></td>
  <td style="padding:.3rem .5rem"><strong>Portfolio (2)</strong> — Behavioural optimum portfolio with the selected derivative at the chosen H and α constraint. Highlighted separately from the frontier squares to identify the specific selected constraint point.</td>
</tr>
<tr style="border-bottom:1px solid #2a2a3a">
  <td style="padding:.3rem .5rem;white-space:nowrap">🔴 <strong>Coral star (white frame)</strong></td>
  <td style="padding:.3rem .5rem"><strong>Portfolio (3)</strong> — Interpolated point on the derivative frontier at the same standard deviation as Portfolio (1). Shows the return achievable with derivatives at equivalent risk — indicative only, not always available (requires derivative frontier to overlap with Portfolio (1) risk level).</td>
</tr>
<tr>
  <td style="padding:.3rem .5rem;white-space:nowrap">➡️ <strong>White dotted arrow</strong></td>
  <td style="padding:.3rem .5rem">Return gap between Portfolio (1) and Portfolio (2) at the selected H and α constraint — illustrates the return uplift (or reduction) from adding derivatives.</td>
</tr>
</table>

<span style="color:#4a9eff;font-weight:700">MVT/MAT equivalence:</span><br>At the equivalence point (λ=3.795, H=-10%, α=5%), the purple and blue curves meet exactly — confirming the MVT/MAT equivalence proven in Das, Markowitz, Scheid & Statman (2010). Adding derivatives shifts the frontier upward (gold squares above blue dots), revealing what the behavioural approach with derivatives can unlock beyond mean-variance.

<span style="color:#4a9eff;font-weight:700">Note on discrete vs continuous frontiers:</span><br>The behavioural frontiers are plotted at discrete constraint levels (H = -2%, -5%, -8%, -10%, -12%, -15%, -18%, -20%, -25%, -30%, -35%, -40%). Each point is the optimal portfolio for that specific mental-account threshold. The MV frontier is continuous as it is computed by sweeping the risk-aversion parameter λ — each MV portfolio corresponds to one behavioural portfolio via the MVT/MAT equivalence, demonstrating that both approaches converge to the same solution when no derivatives are present.

<span style="color:#4a9eff;font-weight:700">Why some behavioural points may appear below the MV frontier:</span><br>When derivatives are present, or when the downside constraint is particularly binding at certain H values, some behavioural frontier points may fall below the MV frontier. This is mathematically correct — the behavioural approach optimises under an additional constraint (the shortfall threshold) which can restrict the feasible set. Without derivatives, both frontiers should coincide closely. With derivatives, the behavioural approach can outperform MV at higher risk levels while remaining protected at the threshold — this is the core insight of the framework. Use Standard or High precision resolution to reduce grid approximation errors.

</div>
""", unsafe_allow_html=True)
        # Sample chart
        import os
        if os.path.exists("sample_output.png"):
            st.markdown(
                '<div style="text-align:center;margin:1rem 0 .6rem 0">'
                '<span style="font-size:1.3rem;font-weight:700;color:#4a9eff">'
                '🖼️ Sample Output</span><br>'
                '<span style="font-size:.85rem;color:#c0c8d8">'
                'Safety collar — showing all three portfolio perspectives</span>'
                '</div>',
                unsafe_allow_html=True)
            _col_l2, _col_img, _col_r2 = st.columns([1, 4, 1])
            with _col_img:
                st.image("sample_output.png", use_container_width=True)

        pass  # welcome screen shown, About tab still renders

    if _run_active and not _needs_compute and _has_results:
        # Results already computed — just re-display from cache
        _cache = st.session_state['_cached_results']
        nd_res = _cache['nd_res']
        dr_res = _cache['dr_res']
        p3_return = _cache['p3_return']
        p3_std = _cache['p3_std']
        constraint_label = _cache['constraint_label']
        der_config = _cache['der_config']
        der_label_sel = _cache['der_label_sel']
        H_val = _cache['H_val']
        _alpha = _cache['_alpha']
        use_es = _cache['use_es']
        _L = _cache['_L']
        # Show simple placeholder for chart (cached fig)
        if st.session_state.get('_fig_plotly'):
            col_summary_c, col_chart_c = st.columns([1, 3.5])
            with col_chart_c:
                st.plotly_chart(st.session_state['_fig_plotly'],
                               use_container_width=True,
                               config={'editable': True, 'displayModeBar': True})

    if _run_active and _needs_compute:
        # Fresh computation needed
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
<div style="background:#ffffff;border:1px solid #1a3a5c;border-radius:8px;padding:.8rem 1rem;margin-bottom:.8rem;color:#111111;font-size:.82rem">
<b style="color:#1a6bbf">Up to three portfolios can be generated as output of the optimisation:</b><br><br>
<b style="color:#10b981">Portfolio (1)</b> — Optimum portfolio without derivatives: identical to the Markowitz MV optimum, derived through the mental accounting framework (reference portfolio)<br>
<b style="color:#f59e0b">Portfolio (2)</b> — Optimum portfolio with derivative, same mental-accounting &amp; risk-aversion constraint (H, α ↔ λ): may reach higher expected returns by exploiting asymmetric derivative payoffs<br>
<b style="color:#e76f51">Portfolio (3)</b> — Portfolio with derivative and with the same variance as Portfolio (1): interpolated from the derivative frontier at equivalent risk level (indicative only)
</div>
''', unsafe_allow_html=True)

        # ── Pre-compute nd_res — retry up to 3 times for robustness ────────────
        _nd_res_pre = None
        for _retry in range(3):
            try:
                _r, _ = run_opt(means_arr, sigs_arr, cov_mat, None, H_val, _alpha,
                                m_val, mp_val, constraint_type=_ctype, L=_L)
                if _r is not None:
                    _nd_res_pre = _r
                    break
            except Exception:
                pass

        with st.spinner("Rendering chart..."):
            # Compute Portfolio (3) point for chart overlay using pre-computed nd_res
            _p3_x, _p3_y = None, None
            if der_xs and len(der_xs) >= 2 and _nd_res_pre:
                try:
                    _target_std = _nd_res_pre['std_dev'] * 100
                    _fp = sorted(zip(der_xs, der_ys), key=lambda p: p[0])
                    _fx = [p[0] for p in _fp]
                    _fy = [p[1] for p in _fp]
                    if min(_fx) <= _target_std <= max(_fx):
                        _p3_x = _target_std
                        _p3_y = float(np.interp(_target_std, _fx, _fy))
                except Exception:
                    pass

            # Compute implied lambda for actual Portfolio (1) point
            _lam_actual = None
            try:
                from scipy.optimize import brentq as _brentq
                _cov_s = corr_to_cov(sigs_in, corr_in)
                _lam_actual = implied_lambda(H_val, alpha_val, means_in, _cov_s)
            except Exception:
                pass

            fig_plotly=plot_frontier_plotly(mv_x,mv_y,mv_eq,nd_xs,nd_ys,nd_lbls,
                                            der_xs,der_ys,der_lbls,der_label_sel,H_val,alpha_val,
                                            p3_x=_p3_x, p3_y=_p3_y,
                                            nd_res_actual=_nd_res_pre,
                                            lam_actual=_lam_actual)
            st.session_state['_fig_plotly'] = fig_plotly
            # Also store as PNG bytes for PDF export (more reliable than figure object)
            try:
                st.session_state['_fig_png'] = fig_plotly.to_image(
                    format='png', width=900, height=500, scale=2)
            except Exception:
                st.session_state['_fig_png'] = None

            # ── Simulation summary + chart side by side ───────────────────────
            col_summary, col_chart = st.columns([1, 3.5])

            with col_summary:
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
                st.plotly_chart(fig_plotly, use_container_width=True, config={'editable': True, 'displayModeBar': True})

        # ── Reading the chart — full width below columns ──────────────────────
        st.markdown(
            '<div style="background:#ffffff;border:1px solid #1a3a5c;border-radius:8px;'
            'padding:.8rem 1rem;margin-top:.5rem;color:#111111;font-size:.82rem">'
            '<b style="color:#1a6bbf">📐 Reading the chart</b> — '
            'Without derivatives, the blue behavioural frontier should closely track the purple MV frontier, '
            'confirming the MVT/MAT equivalence (Das, Markowitz, Scheid &amp; Statman, 2010). '
            'With derivatives, the frontiers may diverge — this is expected and is the core contribution of the framework: '
            'derivatives allow the behavioural approach to reach portfolios that mean-variance optimisation cannot. '
            'Some behavioural points may appear below the MV frontier at certain risk levels — this reflects the binding '
            'nature of the downside constraint at those H values, not a failure of the method. '
            '<b style="color:#d97706">Fast resolution</b> uses a coarse grid and may introduce small approximation errors — '
            'use Standard or High precision for publication-quality results.</div>',
            unsafe_allow_html=True)

        # ── Results ───────────────────────────────────────────────────────────────
        st.markdown("---")
        constraint_label = f"H={H_val:.0%}, α={_alpha:.0%}" if not use_es else f"H={H_val:.0%}, L={_L:.0%}"
        _lam_suffix = f" — implied λ = {lam_summary}" if lam_summary and lam_summary != "—" else ""
        st.markdown(
            f'<h3 style="color:#4a9eff;text-align:center">'
            f'Optimal portfolios with {constraint_label}{_lam_suffix}</h3>',
            unsafe_allow_html=True)

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
        # Use _nd_res_pre if available; if None, retry once more
        nd_res = _nd_res_pre
        if nd_res is None:
            for _retry2 in range(3):
                try:
                    _r2, _ = run_opt(means_arr, sigs_arr, cov_mat, None, H_val, _alpha,
                                     m_val, mp_val, constraint_type=_ctype, L=_L)
                    if _r2 is not None:
                        nd_res = _r2
                        break
                except Exception:
                    pass
        dr_res = None
        p3_return = None
        p3_std = None

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
                # Suggest a wider constraint based on current securities volatility
                _avg_sig = float(np.mean(sigs_arr)) * 100
                _suggested_H_val = min(40, max(15, int(_avg_sig * 1.5)))
                _suggested_H = f"-{_suggested_H_val}%"
                st.warning(
                    f"⚠️ No eligible portfolio found at H={H_val:.0%}, α={_alpha:.0%}. "
                    f"With live market data (avg volatility {_avg_sig:.1f}%), "
                    f"try a wider threshold such as H={_suggested_H} or switch to Standard resolution.")

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
                        f'<div style="background:#ffffff;border:1px solid #f59e0b;border-radius:6px;'
                        f'padding:.6rem 1rem;color:#111111;font-size:.82rem;margin-top:.6rem">'
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
                f'<div style="background:#ffffff;border:1px solid #e76f51;border-radius:6px;'
                f'padding:.6rem 1rem;color:#111111;font-size:.82rem;margin-top:.4rem">'
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

        # ── PDF Export — generate and store in session_state ────────────────────
        st.markdown("---")
        if nd_res:
            try:
                _lam_s = lam_summary if 'lam_summary' in dir() else "—"
                _nd_lbls_pdf = [names_in[i] if i<len(names_in) else f"Asset {i+1}" for i in range(len(nd_res["weights"]))]
                _nd_wts_pdf  = list(nd_res["weights"])
                _nd_cols_pdf = [DONUT_COLORS[i % len(DONUT_COLORS)] for i in range(len(_nd_wts_pdf))]
                _dr_lbls_pdf = [asset_labels[i] if i<len(asset_labels) else f"Asset {i+1}" for i in range(len(dr_res["weights"]))] if dr_res else []
                _dr_wts_pdf  = list(dr_res["weights"]) if dr_res else []
                _dr_cols_pdf = [DONUT_COLORS[i % len(DONUT_COLORS)] for i in range(len(_dr_wts_pdf))] if dr_res else []
                _p3r = p3_return if p3_return is not None else None
                _p3s = p3_std if p3_std is not None else None
                _pdf_bytes = generate_pdf_report(
                    constraint_label=constraint_label,
                    nd_res=nd_res, dr_res=dr_res,
                    p3_return=_p3r, p3_std=_p3s,
                    nd_labels=_nd_lbls_pdf, nd_weights=_nd_wts_pdf, nd_colors=_nd_cols_pdf,
                    dr_labels=_dr_lbls_pdf, dr_weights=_dr_wts_pdf, dr_colors=_dr_cols_pdf,
                    der_label_sel=der_label_sel,
                    H_val=H_val, _alpha=_alpha, use_es=use_es, _L=_L,
                    data_mode=data_mode, names_in=names_in,
                    grid_lbl=grid_lbl, lam_summary=_lam_s,
                    fig_png=st.session_state.get('_fig_png', None)
                )
                # Store PDF bytes in session_state so download button doesn't trigger rerun loss
                st.session_state['_pdf_bytes'] = _pdf_bytes
                st.session_state['_pdf_filename'] = f"portfolio_optimisation_{H_val:.0%}_{_alpha:.0%}.pdf"
            except Exception as _pdf_err:
                st.caption(f"PDF generation failed: {_pdf_err}")

        # Render download button from session_state — st.download_button does NOT trigger rerun
        if st.session_state.get('_pdf_bytes'):
            _col_l, _col_c, _col_r = st.columns([1, 2, 1])
            with _col_c:
                st.download_button(
                    label="📄 Export & Download PDF Report",
                    data=st.session_state['_pdf_bytes'],
                    file_name=st.session_state.get('_pdf_filename', 'report.pdf'),
                    mime="application/pdf",
                    type="primary",
                    key="pdf_download",
                    use_container_width=True
                )

        # ── Cache results in session_state ───────────────────────────────────
        st.session_state['_cached_results'] = {
            'nd_res': nd_res, 'dr_res': dr_res,
            'p3_return': p3_return, 'p3_std': p3_std,
            'constraint_label': constraint_label,
            'der_config': der_config, 'der_label_sel': der_label_sel,
            'H_val': H_val, '_alpha': _alpha, 'use_es': use_es, '_L': _L,
        }
        st.session_state['_needs_compute'] = False

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

    st.markdown(
        '<div style="background:#0d1a2e;border:1px solid #1a3a5c;border-radius:8px;padding:.8rem 1rem;margin-top:.6rem">'
        '<div style="color:#4a9eff;font-weight:600;font-size:.85rem;margin-bottom:.4rem">'
        'Figure 4.1 — Implied Risk-Aversion Coefficient Surface (Jeddou, 2012)</div>'
        '<div style="color:#c0c8d8;font-size:.8rem;margin-bottom:.6rem">'
        'The 3D surface shows the implied risk-aversion coefficient λ for different combinations '
        'of mental-account threshold H (x-axis) and shortfall probability α (y-axis). '
        'Each data point (sphere) corresponds to a feasible (H, α) pair where the MVT/MAT equivalence holds, '
        'confirming that the behavioral and mean-variance approaches are mathematically equivalent '
        'when no derivatives are present.</div>'
        f'<div style="text-align:center">'
        f'<img src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAISAr0DASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9U6KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAoooo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iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigBMCloooATApaKKACkwKWigAooooAKKKKACiiigApMClooAKKKKAE2j0paKKAEwK8x+OH/Er8PaP4gQlr3Q9as7m2VvuM8z/Yzv8A9nZdOeMcqPpXp9cr8SNEk8S+BNf063t0ubyazkNpG+P+PgDdCwLcKwkVCD/CQDQB1Cfdp1cv8ONcj8R+BNE1COdrnzbZFeWTO5pE+R85/wBtWrqKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAoopN1AC0UgOaWgAooooAKKKKACiiigAooooAKKKKACiiigAqN/mqSo2Xd260Aeb/As/2f4Tv/AA9nzD4f1S50xrj7onbIl34/h/12Mc/dr0sHNeZeFFGnfGjxvYk/Zbe6s7C+t7b7izP++W4mRe7f6lXYeqZ7V6XHjnFAD6KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKQ9KAEAqld30VrdQxyZHm8A/w596TUrg20JKOqSv8ke/1rDmaVTFZoTLvBaSV/mKr/F/31VJXM5S5Tp42yMggg08NXLWmo3KXbx26B7OBQjK/3uF6L+a9f8a2NO1SDUUV43wSFJjb7y9ev5H8qlqxSkpGnRUY6r0qSgoKKKgbKtQBPRVfd/8AXqZPu0bAOooooAKKKKACiiigAooooA8v8bt/YHxi+H+snMx1Nbzw55XTy/Mi+2ednvj7Ds2/9NM5+XDemRrtzXnHxztntPC9h4gto2bUNA1SzvbeULkW6NKLe5kYdNotprjJbhR83GM16HbSpcRLNG4likAKOrZDD1oAsUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQ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" '
        f'style="max-width:500px;width:100%;border-radius:6px;border:1px solid #1a3a5c"/>'
        f'</div></div>',
        unsafe_allow_html=True)

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
