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

At the equivalence point (λ=3.795, H=-10%, α=5%), the purple and blue curves meet exactly —
confirming the MVT/MAT equivalence proven in Das, Markowitz, Scheid & Statman (2010).
Adding derivatives shifts the frontier upward (gold squares above blue dots), revealing
what the behavioural approach with derivatives can unlock beyond mean-variance.

**Note on discrete vs continuous frontiers:** The behavioural frontiers are plotted at discrete constraint levels (H = -2%, -5%, -8%, -10%, -12%, -15%, -18%, -20%, -25%, -30%, -35%, -40%). Each point is the optimal portfolio for that specific mental-account threshold. The MV frontier is continuous as it is computed by sweeping the risk-aversion parameter λ — each MV portfolio corresponds to one behavioural portfolio via the MVT/MAT equivalence, demonstrating that both approaches converge to the same solution when no derivatives are present.

**Why some behavioural points may appear below the MV frontier:** When derivatives are present, or when the downside constraint is particularly binding at certain H values, some behavioural frontier points may fall below the MV frontier. This is mathematically correct — the behavioural approach optimises under an additional constraint (the shortfall threshold) which can restrict the feasible set. Without derivatives, both frontiers should coincide closely. With derivatives, the behavioural approach can outperform MV at higher risk levels while remaining protected at the threshold — this is the core insight of the framework. Use Standard or High precision resolution to reduce grid approximation errors.

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
<div style="background:#0d1a2e;border:1px solid #1a3a5c;border-radius:8px;padding:.8rem 1rem;margin-bottom:.8rem;color:#c0c8d8;font-size:.82rem">
<b style="color:#4a9eff">Up to three portfolios can be generated as output of the optimisation:</b><br><br>
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
                # ── Below chart: metrics scorecard left + reading note right ──────
                _cb_left, _cb_right = st.columns([1, 2.5])
                with _cb_left:
                    # Build metrics scorecard from results if available
                    _m1 = _nd_res_pre
                    _nd_ret_s  = f"{_m1['expected_return']*100:.2f}%" if _m1 else "—"
                    _nd_std_s  = f"{_m1['std_dev']*100:.2f}%"        if _m1 else "—"
                    _nd_skew_s = f"{_m1['skewness']:.3f}"            if _m1 else "—"
                    _nd_sf_s   = f"{_m1['shortfall_stat']*100:.2f}%" if _m1 else "—"
                    _lam_s2    = lam_summary if lam_summary != "—" else "—"
                    def _row(label, val, color="#c0c8d8"):
                        return (f'<tr><td style="padding:.25rem .5rem;color:#7fb3e8;font-size:.75rem;white-space:nowrap">{label}</td>'
                                f'<td style="padding:.25rem .5rem;color:{color};font-weight:600;font-size:.8rem;text-align:right">{val}</td></tr>')
                    _scorecard = (
                        '<div style="background:#0d1a2e;border:1px solid #1a3a5c;border-radius:6px;padding:.6rem .8rem;margin-top:.3rem">'                        '<div style="color:#4a9eff;font-weight:700;font-size:.8rem;margin-bottom:.4rem">📊 Key Metrics</div>'                        '<table style="width:100%;border-collapse:collapse">'                        + _row("Portfolio (1) return", _nd_ret_s, "#10b981")                        + _row("Portfolio (1) std dev", _nd_std_s)                        + _row("Portfolio (1) skewness", _nd_skew_s)                        + _row("Shortfall / ES", _nd_sf_s)                        + _row("Implied λ", _lam_s2, "#10b981")                        + _row("Constraint", constraint_str)                    )
                    if _nd_res_pre and dr_res:
                        _dr_ret_s = f"{dr_res['expected_return']*100:.2f}%"
                        _gain_s = f"{(dr_res['expected_return']-_nd_res_pre['expected_return'])*100:+.2f} pp"
                        _gain_col = "#10b981" if dr_res['expected_return'] > _nd_res_pre['expected_return'] else "#ef4444"
                        _scorecard += _row(f"Portfolio (2) return", _dr_ret_s, "#f59e0b")
                        _scorecard += _row("Return gain (2) vs (1)", _gain_s, _gain_col)
                    if p3_return is not None and _nd_res_pre:
                        _p3_gain_s = f"{p3_return - _nd_res_pre['expected_return']*100:+.2f} pp"
                        _p3_col = "#10b981" if p3_return > _nd_res_pre['expected_return']*100 else "#ef4444"
                        _scorecard += _row("Portfolio (3) return", f"{p3_return:.2f}%", "#e76f51")
                        _scorecard += _row("Return gain (3) vs (1)", _p3_gain_s, _p3_col)
                    _scorecard += '</table></div>'
                    st.markdown(_scorecard, unsafe_allow_html=True)
                with _cb_right:
                    st.markdown(
                    '<div style="background:#0d1a2e;border:1px solid #1a3a5c;border-radius:6px;'
                    'padding:.6rem 1rem;color:#c0c8d8;font-size:.78rem;margin-top:.3rem">'
                    '<b style="color:#4a9eff">📐 Reading the chart</b> — '
                    'Without derivatives, the blue behavioural frontier should closely track the purple MV frontier, '
                    'confirming the MVT/MAT equivalence (Das, Markowitz, Scheid &amp; Statman, 2010). '
                    'With derivatives, the frontiers may diverge — this is expected and is the core contribution of the framework: '
                    'derivatives allow the behavioural approach to reach portfolios that mean-variance optimisation cannot. '
                    'Some behavioural points may appear below the MV frontier at certain risk levels — this reflects the binding '
                    'nature of the downside constraint at those H values, not a failure of the method. '
                    '<b style="color:#f59e0b">Fast resolution</b> uses a coarse grid and may introduce small approximation errors — '
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
                f'<div style="background:#ffffff;border:1px solid #e76f51;border-radius:6px;'
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
        f'<img src="data:image/jpeg;base64,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oioL3CZS1iMrBk6vA6eHfg6wbjm0j/kVZlzuLi4jxd+6/EWnaVNXZw2kuqZCIiwUAkb7EjfkfnQkkAEnYdiIrLAJJ23JO3YhJIAJOw7PmREkAEg7jkURFAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBdffc6v+x62/4lH/AJTrkFdffc6v+x62/wCJR/5Trztf6b53mqO2iI/dBPhYwn5ij/XzLm9dIfdBPhYwn5ij/XzLm9as+yhVtYREWjIREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAF199zq/7Hrb/iUf+U65BXX33Or/ALHrb/iUf+U687X+m+d5qjtoiP3QT4WMJ+Yo/wBfMub10h90E+FjCfmKP9fMub1qz7KFW1kn6O60Fm/kjNBSmMOPkkiFsDqmvDm7F2/YOfasjlcRp+fr8o6dkVapBC24zGbPa6w8uG0Zcdg0AAnzd3nUPo3rVIWBVl6sWIXQS+SDxMO245jl2DmOa9sRl7+KMvgcrGsmAEsckbZGP2O43a4Ebg9hUqpbco+Tb6Hb1WtVrZ1xshTuwnPvjDvJDkNLY3EV79jJ37T2VrTa8QgiG8nHF1jSdz5PI8+3sUggoabxufvPrx2WyVcOJxvXje1nkM8tocTu/nvz5bkrX97M5O9BNDbtumZPY8Jk4mjd0nCW777b9h227PmXvLqPLyx8D7LD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3DvUwudNdfNyVG6l0jWuQ15HyARTN2a98JjLxHJG9jncbnybPDmku225bqu/05mzqXF5dmmOBmPyLrwgN4Fsm9VlcNO0QA24OLcN257bDtUmoYGtoNH6unqyW4NLZyWvHG2WSVmPlcxjHDdriQ3YAjmD2FVV9Hapltx136ey0HHK6IvkoTcLC3bj32aT5IILgASB5lsmTpvr2XV8ha0xL7q1JJbML4shw15LEtcQSOkj6vfg2AcGBw25jfbsuLn7oO5YxJpfezE2Uw12GfwznxtdvO/bg/8AmtAbtv5O2+7uxWasiYEIzHRRrvHxVZodPZHJRW5J44XUqM7y7qnljiWlgc0EgkbgEjnssHqrTV7TrMa+4QRerGYDhLXRPa90ckTweYex7S0hbAj6XcRLk6mUv6PszXqFrIWaUkWVaxsLrczpQ7hdA7ifGXeSTy3APCop0h57F5PFadxmJkszRUKssk0ll5fL108zpXsc4tbxlu4HEAATuUTe8EOREWgEREAREQBERAEREAREQBERAEREAREQBERAEREAXrW99L/wJf8AocvJetb30v8AwJf+hyoMSiIuYBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAXX/ANze/l7/AGd/qlyAuv8A7m9/L3+zv9UpVsKtpHfugnwsYT8xR/r5lzeukPugnwsYT8xR/r5lzeuyz7KOarazOaNr4izesR5R0HH1J8EZZmdFA+XiHJ7282jh4tuYG+25AUgdg8fRweQuXcRRgtxZB8LYLl6TyGCJrwGOYQHk8W4PPcEdqiGIylnFyvkrMqydY3heyxWjnYRvv717SP0q/wDvtzjmztmmqWWzzde9tihBMA/hDd2h7Dw+SAAG7AABdatKFQlGPgs/g5qrOt1ynh4l7kqmHxzqOIONfPZnr15pLjp3NIdK1r9mtHk8Ia7h58ydzuOxZfEaZw1jVeXx80DjBWztenC3rSCInzvY5u+/M8LRz7VFX6iykmKbjZZK8kLIxEx76sTpmRg7hjZC3jDd/MCrifWGoJpIJXW4WSw2GWRJHViY6SVm/C+Qhu8hG599v2nvK9NbY31U1gnMQtmX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9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2CeJv0nfLuj/AK3Z9giJeYgeJv0nfLuj/rdn2C3x+5L6G9T9En3zffHfw9v3V8E6jwCaR/D1XXcXFxxs2/2jdtt+w9iIjqbEH//Z" '
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
