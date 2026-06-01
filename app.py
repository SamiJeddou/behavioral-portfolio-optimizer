"""
Behavioral Portfolio Optimizer — Streamlit Dashboard
=====================================================
Interactive dashboard for the behavioral portfolio optimizer.
Based on Das, Markowitz, Scheid & Statman (2010) JFQA
and original MSc thesis by Sami Jeddou, USI Lugano 2012.
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from behavioral_portfolio_optimizer import (
    build_state_space,
    assign_probabilities,
    optimize_portfolio,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Behavioral Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #ffffff; font-size: 1.6rem; }
    h2, h3 { color: #c0c8d8; }
    .stMetric label { color: #8896a8 !important; font-size: 0.8rem; }
    .stMetric value { color: #ffffff !important; }
    .info-box {
        background: #1a1a2e; border: 1px solid #3a3a5a;
        border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 1rem;
    }
    .result-box {
        background: #0f1923; border: 1px solid #2a3a4a;
        border-radius: 8px; padding: 1rem 1.2rem;
    }
    .warning-box {
        background: #1a1200; border: 1px solid #f59e0b;
        border-radius: 6px; padding: 0.6rem 1rem; margin-top: 0.4rem;
        color: #f59e0b; font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Base parameters (Das & Statman base case) ─────────────────────────────────
MEANS      = np.array([0.05, 0.10, 0.25])
SIGS       = np.array([0.05, 0.20, 0.50])
COV_MATRIX = np.array([0.0025, 0, 0,
                        0, 0.04, 0.02,
                        0, 0.02, 0.25]).reshape(3, 3)

GRID_OPTIONS = {
    "⚡ Fast — preview  (m=21, m'=15)":          (21, 15),
    "⚖️ Standard  (m=35, m'=50)":                (35, 50),
    "🎯 High precision — thesis  (m=51, m'=99)": (51, 99),
}

DERIVATIVE_OPTIONS = {
    "None — primary securities only":       None,
    "Put option":                           "put",
    "Call option":                          "call",
    "Safety collar (long put + short call)":"safety_collar",
    "Aggressive collar (long call + short put)": "aggressive_collar",
    "Straddle (long call + long put)":      "straddle",
    "Strangle (long call + long put, diff strikes)": "strangle",
    "Capital-guaranteed note — uncapped":   "cgn_uncapped",
    "Capital-guaranteed note — capped":     "cgn_capped",
    "Barrier-M note":                       "barrier_m",
}

ASSET_LABELS = ["Sec 1 — Low risk", "Sec 2 — Mid risk",
                "Sec 3 — High risk", "Derivative"]


# ── Helper: build derivative config ──────────────────────────────────────────
def build_derivative_config(der_type, params):
    base = {"underlying_index": 2, "vol": SIGS[2],
            "S0": 1.0, "r": 0.03, "T": 1.0}
    if der_type == "put":
        return {**base, "type": "put", "strike": params["strike"]}
    elif der_type == "call":
        return {**base, "type": "call", "strike": params["strike"]}
    elif der_type == "safety_collar":
        return {**base, "type": "safety_collar",
                "strike_p": params["strike_p"], "strike_c": params["strike_c"]}
    elif der_type == "aggressive_collar":
        return {**base, "type": "aggressive_collar",
                "strike_p": params["strike_p"], "strike_c": params["strike_c"]}
    elif der_type == "straddle":
        return {**base, "type": "straddle", "strike": params["strike"]}
    elif der_type == "strangle":
        return {**base, "type": "strangle",
                "strike_kp": params["strike_kp"], "strike_kc": params["strike_kc"]}
    elif der_type == "cgn_uncapped":
        return {**base, "type": "cgn", "floor": params["floor"],
                "participation": params["participation"],
                "cap": None, "cgn_premium": params["premium"]}
    elif der_type == "cgn_capped":
        return {**base, "type": "cgn", "floor": params["floor"],
                "participation": params["participation"],
                "cap": params["cap"], "cgn_premium": params["premium"]}
    elif der_type == "barrier_m":
        return {**base, "type": "barrier_m",
                "M": params["M"], "premium_bm": params["premium_bm"]}
    return None


# ── Helper: mean-variance frontier ───────────────────────────────────────────
@st.cache_data
def compute_mv_frontier():
    def mv_opt(lam):
        def obj(w): return -(w @ MEANS - (lam / 2) * (w @ COV_MATRIX @ w))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1)] * 3
        best = None
        for x0 in [np.array([0.6, 0.2, 0.2]),
                   np.array([0.33, 0.33, 0.34]),
                   np.array([0.8, 0.1, 0.1])]:
            res = minimize(obj, x0, method="SLSQP",
                           bounds=bounds, constraints=constraints)
            if res.success and (best is None or res.fun < best.fun):
                best = res
        w = best.x
        return float(np.sqrt(w @ COV_MATRIX @ w)) * 100, float(w @ MEANS) * 100
    lambdas = np.linspace(0.5, 20, 120)
    pts = [mv_opt(l) for l in lambdas]
    return [p[0] for p in pts], [p[1] for p in pts], mv_opt(3.7950)


# ── Helper: run optimizer ─────────────────────────────────────────────────────
def run_optimizer(der_config, H, alpha, m, m_prime):
    U, dr = build_state_space(MEANS, SIGS, m=m, derivative_config=der_config)
    U = assign_probabilities(U, MEANS, SIGS, COV_MATRIX, dr)
    n_sec = U.shape[1] - 1
    return optimize_portfolio(U, n_sec, H=H, alpha=alpha, m_prime=m_prime), n_sec


# ── Helper: build frontier data (sweep H) ────────────────────────────────────
def build_behavioral_frontier(der_config, alpha, m, m_prime):
    H_values = [-0.05, -0.08, -0.10, -0.12, -0.15, -0.18, -0.20]
    xs, ys, labels = [], [], []
    for H in H_values:
        try:
            res, _ = run_optimizer(der_config, H, alpha, m, m_prime)
            xs.append(res["std_dev"] * 100)
            ys.append(res["expected_return"] * 100)
            labels.append(f"H={H:.0%}")
        except Exception:
            pass
    return xs, ys, labels


# ── Helper: plot ──────────────────────────────────────────────────────────────
def plot_frontier(mv_x, mv_y, mv_eq,
                  nd_x, nd_y, nd_labels,
                  cgn_x, cgn_y, cgn_labels,
                  der_label, H_selected, alpha):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.grid(True, color="#1e2130", linewidth=0.6, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)

    # MV frontier
    ax.plot(mv_x, mv_y, color="#6b7280", linewidth=2, linestyle="--",
            label="Mean-variance frontier (Markowitz)", zorder=2, alpha=0.9)

    # Behavioral — no derivative
    ax.plot(nd_x, nd_y, color="#4a9eff", linewidth=2.5, marker="o",
            markersize=7, markerfacecolor="#4a9eff",
            label="Behavioral — no derivative", zorder=3)
    for x, y, lbl in zip(nd_x, nd_y, nd_labels):
        ax.annotate(lbl, xy=(x, y), xytext=(x, y - 1.8),
                    color="#7fb3e8", fontsize=7.5, ha="center", zorder=4)

    # Behavioral — with derivative
    if cgn_x:
        ax.scatter(cgn_x, cgn_y, color="#f59e0b", s=65, marker="s", zorder=3,
                   label=f"Behavioral — {der_label}")
        # Label the H=-10% point if present
        for x, y, lbl in zip(cgn_x, cgn_y, cgn_labels):
            if lbl == f"H={H_selected:.0%}":
                ax.annotate(f"{lbl}, α={alpha:.0%}",
                            xy=(x, y), xytext=(x - 8, y + 2),
                            color="#f59e0b", fontsize=8,
                            arrowprops=dict(arrowstyle="->",
                                            color="#f59e0b", lw=1.2),
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="#0d1117",
                                      edgecolor="#f59e0b", alpha=0.85))
        # Gain arrow at selected H
        try:
            h_idx_nd  = nd_labels.index(f"H={H_selected:.0%}")
            h_idx_cgn = cgn_labels.index(f"H={H_selected:.0%}")
            x0, y0 = nd_x[h_idx_nd],  nd_y[h_idx_nd]
            x1, y1 = cgn_x[h_idx_cgn], cgn_y[h_idx_cgn]
            gain = y1 - y0
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color="#ffffff",
                                        lw=1.6, linestyle="dotted"))
            ax.text((x0 + x1) / 2 + 1, (y0 + y1) / 2,
                    f"+{gain:.1f} pp return\nsame constraint",
                    color="#e2e8f0", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#0d1117",
                              edgecolor="#ffffff", alpha=0.8))
        except (ValueError, IndexError):
            pass

    # Equivalence point
    ax.scatter(*mv_eq, color="#10b981", s=130, zorder=5, marker="D")
    ax.annotate(
        "Equivalence point\nλ=3.795 ↔ H=-10%, α=5%\nMV = Behavioral = 10.2%",
        xy=mv_eq, xytext=(mv_eq[0] + 3, mv_eq[1] - 5),
        color="#10b981", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#10b981", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                  edgecolor="#10b981", alpha=0.9), zorder=6)

    # MVT/MAT note
    ax.text(22, max(max(nd_y), max(cgn_y) if cgn_y else 0) + 2,
            "MV and behavioral frontiers converge without derivatives\n"
            "(MVT/MAT equivalence — Das, Markowitz, Scheid & Statman 2010)",
            color="#6b7280", fontsize=7.5, ha="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                      edgecolor="#3a3a5a", alpha=0.85))

    ax.set_xlabel("Portfolio Risk — Standard Deviation (%)",
                  color="#c0c8d8", fontsize=10, labelpad=6)
    ax.set_ylabel("Expected Return (%)",
                  color="#c0c8d8", fontsize=10, labelpad=6)
    ax.set_title("Mean-Variance vs Behavioral Portfolio Frontier",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#8896a8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a3a")
    ax.legend(loc="upper left", fontsize=9,
              facecolor="#1a1a2e", edgecolor="#3a3a5a",
              labelcolor="white", framealpha=0.9)
    fig.text(0.5, 0.001,
             "Das & Statman (2004)  |  "
             "Das, Markowitz, Scheid & Statman (2010) JFQA Vol.45 No.2  |  "
             "Sami Jeddou, MSc Finance USI Lugano 2012",
             ha="center", color="#4a5568", fontsize=7, style="italic")
    all_x = mv_x + nd_x + (cgn_x if cgn_x else [])
    all_y = mv_y + nd_y + (cgn_y if cgn_y else [])
    ax.set_xlim(0, max(all_x) * 1.15)
    ax.set_ylim(min(all_y) - 3, max(all_y) + 6)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown("---")

    # Derivative selector
    st.markdown("### Derivative / Structured product")
    der_label_selected = st.selectbox(
        "Type", list(DERIVATIVE_OPTIONS.keys()), index=0,
        label_visibility="collapsed")
    der_type = DERIVATIVE_OPTIONS[der_label_selected]

    # Derivative-specific parameters
    der_params = {}
    if der_type in ("put", "call", "straddle"):
        der_params["strike"] = st.slider(
            "Strike price", 0.5, 2.0,
            1.4 if der_type == "put" else (1.2 if der_type == "call" else 0.7),
            0.05)
    elif der_type in ("safety_collar", "aggressive_collar"):
        der_params["strike_p"] = st.slider("Put strike (Kp)", 0.5, 1.5, 1.2, 0.05)
        der_params["strike_c"] = st.slider("Call strike (Kc)", 1.0, 2.0, 1.6, 0.05)
    elif der_type == "strangle":
        der_params["strike_kp"] = st.slider("Put strike (Kp)", 0.5, 1.2, 0.8, 0.05)
        der_params["strike_kc"] = st.slider("Call strike (Kc)", 0.8, 1.5, 0.9, 0.05)
    elif der_type in ("cgn_uncapped", "cgn_capped"):
        der_params["floor"]         = st.slider("Floor (%)", 0.0, 10.0, 1.0, 0.5) / 100
        der_params["participation"] = st.slider("Participation (%)", 50, 150, 100, 10) / 100
        der_params["premium"]       = st.slider("Premium (%)", 0.0, 20.0, 0.0, 1.0) / 100
        if der_type == "cgn_capped":
            der_params["cap"] = st.slider("Cap (%)", 5.0, 50.0, 20.0, 5.0) / 100
    elif der_type == "barrier_m":
        der_params["M"]          = st.slider("Barrier M (%)", 10, 60, 40, 5) / 100
        der_params["premium_bm"] = st.slider("Premium (%)", 0.0, 20.0, 10.0, 1.0) / 100

    st.markdown("---")
    st.markdown("### Mental-account constraint")
    H_val     = st.slider("Threshold H (%)", -25, -3, -10, 1) / 100
    alpha_val = st.slider("Shortfall probability α (%)", 1, 15, 5, 1) / 100

    st.markdown("---")
    st.markdown("### Grid resolution")
    grid_label = st.selectbox("Resolution", list(GRID_OPTIONS.keys()), index=0,
                               label_visibility="collapsed")
    m_val, mp_val = GRID_OPTIONS[grid_label]
    if "High precision" in grid_label:
        st.markdown(
            '<div class="warning-box">⚠️ High precision may take 5–10 minutes. '
            'Recommended for final results only.</div>',
            unsafe_allow_html=True)
    elif "Standard" in grid_label:
        st.markdown(
            '<div class="warning-box">⏱️ Standard resolution takes ~1–2 minutes.</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    run_btn = st.button("▶ Run optimizer", type="primary", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═════════════════════════════════════════════════════════════════════════════
tab_optimizer, tab_about = st.tabs(["📊 Optimizer", "📖 About"])

with tab_optimizer:
    st.markdown("## Behavioral Portfolio Optimizer")
    st.markdown(
        "Extends Markowitz mean-variance theory to portfolios including **derivatives "
        "and structured products** using a **mental-accounting downside constraint**.")

    if not run_btn:
        st.info(
            "Configure parameters in the sidebar and click **▶ Run optimizer** to generate results.",
            icon="👈")
        st.stop()

    # ── Run ──────────────────────────────────────────────────────────────────
    der_config = build_derivative_config(der_type, der_params) if der_type else None

    with st.spinner("Step 1/4 — Building state space..."):
        mv_x, mv_y, mv_eq = compute_mv_frontier()

    with st.spinner("Step 2/4 — Running behavioral optimizer (no derivative)..."):
        try:
            nd_xs, nd_ys, nd_lbls = build_behavioral_frontier(
                None, alpha_val, m_val, mp_val)
        except Exception as e:
            st.error(f"Optimizer failed (no derivative): {e}")
            st.stop()

    cgn_xs, cgn_ys, cgn_lbls = [], [], []
    if der_config is not None:
        with st.spinner(f"Step 3/4 — Running behavioral optimizer ({der_label_selected})..."):
            try:
                cgn_xs, cgn_ys, cgn_lbls = build_behavioral_frontier(
                    der_config, alpha_val, m_val, mp_val)
            except Exception as e:
                st.warning(f"Could not compute derivative frontier: {e}")

    with st.spinner("Step 4/4 — Rendering chart..."):
        fig = plot_frontier(
            mv_x, mv_y, mv_eq,
            nd_xs, nd_ys, nd_lbls,
            cgn_xs, cgn_ys, cgn_lbls,
            der_label_selected, H_val, alpha_val)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Point results at selected H ──────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### Optimal portfolio at H = {H_val:.0%}, α = {alpha_val:.0%}")

    col1, col2 = st.columns(2)

    # No derivative
    with col1:
        st.markdown("**Without derivative**")
        try:
            h_idx = nd_lbls.index(f"H={H_val:.0%}")
            nd_res, nd_nsec = run_optimizer(None, H_val, alpha_val, m_val, mp_val)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Expected return", f"{nd_res['expected_return']*100:.2f}%")
            m2.metric("Std deviation",   f"{nd_res['std_dev']*100:.2f}%")
            m3.metric("Skewness",        f"{nd_res['skewness']:.3f}")
            m4.metric("Shortfall prob",  f"{nd_res['shortfall_stat']*100:.2f}%")
            st.markdown("**Weights**")
            for i, w in enumerate(nd_res["weights"]):
                lbl = ASSET_LABELS[i] if i < len(ASSET_LABELS) else f"Asset {i+1}"
                st.progress(float(w), text=f"{lbl}: {w*100:.1f}%")
        except Exception as e:
            st.warning(f"Could not retrieve result: {e}")

    # With derivative
    with col2:
        if der_config is not None:
            st.markdown(f"**With {der_label_selected}**")
            try:
                cgn_res, cgn_nsec = run_optimizer(
                    der_config, H_val, alpha_val, m_val, mp_val)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Expected return", f"{cgn_res['expected_return']*100:.2f}%",
                          delta=f"+{(cgn_res['expected_return']-nd_res['expected_return'])*100:.2f}pp")
                m2.metric("Std deviation",   f"{cgn_res['std_dev']*100:.2f}%")
                m3.metric("Skewness",        f"{cgn_res['skewness']:.3f}")
                m4.metric("Shortfall prob",  f"{cgn_res['shortfall_stat']*100:.2f}%")
                st.markdown("**Weights**")
                labels = ASSET_LABELS[:cgn_nsec]
                for i, w in enumerate(cgn_res["weights"]):
                    lbl = labels[i] if i < len(labels) else f"Asset {i+1}"
                    st.progress(float(w), text=f"{lbl}: {w*100:.1f}%")
            except Exception as e:
                st.warning(f"Could not retrieve result: {e}")
        else:
            st.info("Select a derivative in the sidebar to compare.")


with tab_about:
    st.markdown("## About this tool")
    st.markdown("""
<div class="info-box">

### Theoretical framework

This optimizer extends **Markowitz mean-variance theory** to portfolios that include
derivatives and structured products, using a **mental-accounting downside constraint**:

> *The probability of the portfolio return falling below a threshold H must not exceed α.*

This is equivalent to a **Value-at-Risk constraint** embedded directly in the objective function,
allowing the optimizer to allocate to derivatives whose asymmetric payoffs provide downside
protection while preserving upside participation.

### Algorithm — three steps

**Step 1 — State space construction**
A discrete grid of return scenarios is built for all primary securities (m grid steps each).
For each scenario, derivative returns are computed analytically using Black-Scholes pricing.

**Step 2 — Probability assignment**
Each state is assigned a probability using a **Gaussian copula**, correctly capturing the
dependence structure between assets and supporting non-normal marginal distributions.

**Step 3 — Two-stage optimization**
- *Grid search*: All weight combinations (m′ steps per weight) are evaluated against the
  mental-account constraint. The highest-return eligible portfolio becomes the starting point.
- *Gradient refinement*: A COBYLA nonlinear optimizer refines the solution with the constraint
  embedded as a penalty term.

### MVT / MAT equivalence (Chapter 4)

When **no derivatives** are present, the mean-variance and behavioral frontiers converge exactly.
For H = -10% and α = 5%, the implied risk-aversion coefficient is **λ = 3.795**, at which point
both methods yield the same optimal portfolio (return ≈ 10.2%, std ≈ 12.3%).

Adding derivatives **breaks this equivalence** — the behavioral approach can exploit asymmetric
payoff profiles that mean-variance optimization cannot capture.

</div>

### Academic references

- **Das, Sanjiv and Meir Statman (2004)** — *Beyond Mean-Variance: Portfolios with Derivatives
  and Non-Normal Returns in Mental Accounts*

- **Das, Sanjiv, Harry Markowitz, Jonathan Scheid and Meir Statman (2010)** —
  *Portfolio Optimization with Mental Accounts*, Journal of Financial and Quantitative Analysis,
  Vol. 45, No. 2, pp. 311–334

- **Sami Jeddou (2012)** — *Beyond Mean-Variance: Options and Structured Products in Behavioral
  Portfolios*, MSc Finance Thesis, Università della Svizzera italiana (USI Lugano),
  supervised by Prof. Enrico De Giorgi

### Author

**Sami Jeddou** — Senior Financial Services Transformation Leader
🔗 [LinkedIn](https://www.linkedin.com/in/sami-jeddou-25787a404)
🐙 [GitHub](https://github.com/SamiJeddou/behavioral-portfolio-optimizer)
""", unsafe_allow_html=True)
