"""
Behavioral Portfolio Optimizer — Streamlit Dashboard
=====================================================
Interactive dashboard for the behavioral portfolio optimizer.
Based on Das, Markowitz, Scheid & Statman (2010) JFQA
and original MSc thesis by Sami Jeddou, USI Lugano 2012.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from io import StringIO
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

st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #ffffff; font-size: 1.6rem; }
    h2, h3 { color: #c0c8d8; }
    .stMetric label { color: #8896a8 !important; font-size: 0.8rem; }
    .info-box {
        background: #1a1a2e; border: 1px solid #3a3a5a;
        border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 1rem;
    }
    .warning-box {
        background: #1a1200; border: 1px solid #f59e0b;
        border-radius: 6px; padding: 0.6rem 1rem; margin-top: 0.4rem;
        color: #f59e0b; font-size: 0.82rem;
    }
    .success-box {
        background: #001a0f; border: 1px solid #10b981;
        border-radius: 6px; padding: 0.6rem 1rem; margin-top: 0.4rem;
        color: #10b981; font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Defaults (Das & Statman base case) ───────────────────────────────────────
DEFAULT_MEANS  = [0.05, 0.10, 0.25]
DEFAULT_SIGS   = [0.05, 0.20, 0.50]
DEFAULT_CORR   = [[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.4],
                  [0.0, 0.4, 1.0]]
DEFAULT_NAMES  = ["Sec 1 — Low risk", "Sec 2 — Mid risk", "Sec 3 — High risk"]

GRID_OPTIONS = {
    "⚡ Fast — preview  (m=21, m'=15)":           (21, 15),
    "⚖️  Standard  (m=35, m'=50)":                (35, 50),
    "🎯 High precision — thesis  (m=51, m'=99)":  (51, 99),
}

DERIVATIVE_OPTIONS = {
    "None — primary securities only":                   None,
    "Put option":                                        "put",
    "Call option":                                       "call",
    "Safety collar (long put + short call)":             "safety_collar",
    "Aggressive collar (long call + short put)":         "aggressive_collar",
    "Straddle (long call + long put)":                   "straddle",
    "Strangle (long call + long put, diff strikes)":     "strangle",
    "Capital-guaranteed note — uncapped":                "cgn_uncapped",
    "Capital-guaranteed note — capped":                  "cgn_capped",
    "Barrier-M note":                                    "barrier_m",
}


# ── Helper: corr → cov ───────────────────────────────────────────────────────
def corr_to_cov(sigs, corr):
    sigs = np.array(sigs)
    corr = np.array(corr)
    return np.outer(sigs, sigs) * corr


# ── Helper: CSV → means, sigs, corr ─────────────────────────────────────────
def parse_csv(uploaded_file):
    """Parse CSV of historical prices (date column + one column per asset)."""
    df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    returns = df.pct_change().dropna()
    means  = returns.mean().tolist()
    sigs   = returns.std().tolist()
    corr   = returns.corr().values.tolist()
    names  = list(returns.columns)
    return means, sigs, corr, names


# ── Helper: build derivative config ──────────────────────────────────────────
def build_derivative_config(der_type, params, sigs):
    base = {"underlying_index": len(sigs) - 1,
            "vol": sigs[-1], "S0": 1.0, "r": 0.03, "T": 1.0}
    configs = {
        "put":               {**base, "type": "put",   "strike": params["strike"]},
        "call":              {**base, "type": "call",  "strike": params["strike"]},
        "straddle":          {**base, "type": "straddle", "strike": params["strike"]},
        "safety_collar":     {**base, "type": "safety_collar",
                              "strike_p": params["strike_p"], "strike_c": params["strike_c"]},
        "aggressive_collar": {**base, "type": "aggressive_collar",
                              "strike_p": params["strike_p"], "strike_c": params["strike_c"]},
        "strangle":          {**base, "type": "strangle",
                              "strike_kp": params["strike_kp"], "strike_kc": params["strike_kc"]},
        "cgn_uncapped":      {**base, "type": "cgn", "floor": params["floor"],
                              "participation": params["participation"],
                              "cap": None, "cgn_premium": params["premium"]},
        "cgn_capped":        {**base, "type": "cgn", "floor": params["floor"],
                              "participation": params["participation"],
                              "cap": params["cap"], "cgn_premium": params["premium"]},
        "barrier_m":         {**base, "type": "barrier_m",
                              "M": params["M"], "premium_bm": params["premium_bm"]},
    }
    return configs.get(der_type)


# ── Helper: mean-variance frontier ───────────────────────────────────────────
def compute_mv_frontier(means, cov_matrix):
    means = np.array(means)
    n = len(means)
    def mv_opt(lam):
        def obj(w): return -(w @ means - (lam / 2) * (w @ cov_matrix @ w))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1)] * n
        best = None
        x0_list = [np.ones(n) / n]
        x0_list += [np.eye(n)[i] * 0.6 + np.ones(n) * 0.4 / n for i in range(n)]
        for x0 in x0_list:
            res = minimize(obj, x0, method="SLSQP",
                           bounds=bounds, constraints=constraints)
            if res.success and (best is None or res.fun < best.fun):
                best = res
        if best is None:
            return None
        w = best.x
        return float(np.sqrt(w @ cov_matrix @ w)) * 100, float(w @ means) * 100
    lambdas = np.linspace(0.3, 25, 120)
    pts = [mv_opt(l) for l in lambdas if mv_opt(l) is not None]
    eq_pt = mv_opt(3.7950)
    return [p[0] for p in pts], [p[1] for p in pts], eq_pt


# ── Helper: run optimizer ─────────────────────────────────────────────────────
def run_optimizer(means, sigs, cov_matrix, der_config, H, alpha, m, m_prime):
    U, dr = build_state_space(means, sigs, m=m, derivative_config=der_config)
    U = assign_probabilities(U, means, sigs, cov_matrix, dr)
    n_sec = U.shape[1] - 1
    result = optimize_portfolio(U, n_sec, H=H, alpha=alpha, m_prime=m_prime)
    return result, n_sec


# ── Helper: sweep H for frontier ─────────────────────────────────────────────
def build_behavioral_frontier(means, sigs, cov_matrix, der_config, alpha, m, m_prime):
    H_values = [-0.05, -0.08, -0.10, -0.12, -0.15, -0.18, -0.20]
    xs, ys, labels = [], [], []
    for H in H_values:
        try:
            res, _ = run_optimizer(means, sigs, cov_matrix,
                                   der_config, H, alpha, m, m_prime)
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
                  der_label, H_selected, alpha,
                  asset_names):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.grid(True, color="#1e2130", linewidth=0.6, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)

    ax.plot(mv_x, mv_y, color="#6b7280", linewidth=2, linestyle="--",
            label="Mean-variance frontier (Markowitz)", zorder=2, alpha=0.9)

    ax.plot(nd_x, nd_y, color="#4a9eff", linewidth=2.5, marker="o",
            markersize=7, markerfacecolor="#4a9eff",
            label="Behavioral — no derivative", zorder=3)
    for x, y, lbl in zip(nd_x, nd_y, nd_labels):
        ax.annotate(lbl, xy=(x, y), xytext=(x, y - 1.8),
                    color="#7fb3e8", fontsize=7.5, ha="center", zorder=4)

    if cgn_x:
        ax.scatter(cgn_x, cgn_y, color="#f59e0b", s=65, marker="s", zorder=3,
                   label=f"Behavioral — {der_label}")
        for x, y, lbl in zip(cgn_x, cgn_y, cgn_labels):
            if lbl == f"H={H_selected:.0%}":
                ax.annotate(f"{lbl}, \u03b1={alpha:.0%}",
                            xy=(x, y), xytext=(x - 8, y + 2),
                            color="#f59e0b", fontsize=8,
                            arrowprops=dict(arrowstyle="->",
                                            color="#f59e0b", lw=1.2),
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="#0d1117",
                                      edgecolor="#f59e0b", alpha=0.85))
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

    if mv_eq:
        ax.scatter(*mv_eq, color="#10b981", s=130, zorder=5, marker="D")
        ax.annotate(
            f"Equivalence point\n\u03bb=3.795 \u2194 H=-10%, \u03b1=5%\nMV = Behavioral = {mv_eq[1]:.1f}%",
            xy=mv_eq, xytext=(mv_eq[0] + 3, mv_eq[1] - 5),
            color="#10b981", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#10b981", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                      edgecolor="#10b981", alpha=0.9), zorder=6)

    # Grey note — fixed at top centre, always visible
    ax.text(0.5, 0.97,
            "MV and behavioral frontiers converge without derivatives\n"
            "(MVT/MAT equivalence \u2014 Das, Markowitz, Scheid & Statman 2010)",
            transform=ax.transAxes,
            color="#6b7280", fontsize=7.5, ha="center", va="top", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                      edgecolor="#3a3a5a", alpha=0.95), zorder=10)

    ax.set_xlabel("Portfolio Risk \u2014 Standard Deviation (%)",
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

    # ── Section 1: Data input ─────────────────────────────────────────────────
    st.markdown("### 📂 Portfolio data")
    data_mode = st.radio(
        "Data source",
        ["Use default (Das & Statman base case)", "Enter manually", "Upload CSV"],
        index=0, label_visibility="collapsed")

    means_input = DEFAULT_MEANS.copy()
    sigs_input  = DEFAULT_SIGS.copy()
    corr_input  = [row[:] for row in DEFAULT_CORR]
    names_input = DEFAULT_NAMES.copy()
    data_valid  = True

    if data_mode == "Use default (Das & Statman base case)":
        st.markdown(
            '<div class="success-box">✓ Using Das & Statman (2010) base case:<br>'
            'Means: 5%, 10%, 25% &nbsp;|&nbsp; Std devs: 5%, 20%, 50%</div>',
            unsafe_allow_html=True)

    elif data_mode == "Enter manually":
        n_assets = st.number_input("Number of primary securities", 2, 6, 3, 1)
        names_input, means_input, sigs_input = [], [], []
        st.markdown("**Returns (annualised)**")
        for i in range(n_assets):
            c1, c2, c3 = st.columns([2, 1, 1])
            name = c1.text_input(f"Name {i+1}", value=f"Asset {i+1}", key=f"name_{i}")
            mean = c2.number_input("Mean %", value=DEFAULT_MEANS[i]*100 if i < 3 else 10.0,
                                   key=f"mean_{i}", format="%.1f") / 100
            sig  = c3.number_input("Std %", value=DEFAULT_SIGS[i]*100 if i < 3 else 20.0,
                                   key=f"sig_{i}", format="%.1f") / 100
            names_input.append(name)
            means_input.append(mean)
            sigs_input.append(sig)

        st.markdown("**Correlation matrix**")
        corr_input = [[1.0] * n_assets for _ in range(n_assets)]
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                default_corr = DEFAULT_CORR[i][j] if i < 3 and j < 3 else 0.0
                val = st.slider(
                    f"Corr({names_input[i]}, {names_input[j]})",
                    -1.0, 1.0, default_corr, 0.05,
                    key=f"corr_{i}_{j}")
                corr_input[i][j] = val
                corr_input[j][i] = val

        # Validate positive semi-definite
        cov_test = corr_to_cov(sigs_input, corr_input)
        if np.any(np.linalg.eigvalsh(cov_test) < -1e-8):
            st.error("⚠️ Correlation matrix is not positive semi-definite. Adjust correlations.")
            data_valid = False

    elif data_mode == "Upload CSV":
        st.markdown(
            '<div class="info-box" style="font-size:0.82rem">'
            '📋 <b>CSV format:</b> First column = dates, remaining columns = asset prices '
            '(one column per asset, header = asset name). '
            'Returns are computed automatically as daily % changes.</div>',
            unsafe_allow_html=True)

        # Download sample CSV button
        sample_csv = """Date,Low_Risk,Mid_Risk,High_Risk
2020-01-02,100.00,100.00,100.00
2020-01-03,100.05,100.15,100.40
2020-01-06,100.08,100.30,100.85
2020-01-07,100.12,100.10,101.20
2020-01-08,100.09,100.45,100.60
2020-01-09,100.15,100.55,101.80
2020-01-10,100.18,100.40,102.10"""
        st.download_button("⬇ Download sample CSV", sample_csv,
                           "sample_portfolio.csv", "text/csv")

        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            try:
                means_input, sigs_input, corr_input, names_input = parse_csv(uploaded)
                st.markdown(
                    f'<div class="success-box">✓ Loaded {len(means_input)} assets: '
                    f'{", ".join(names_input)}</div>',
                    unsafe_allow_html=True)
                with st.expander("Preview computed statistics"):
                    stats_df = pd.DataFrame({
                        "Asset": names_input,
                        "Mean return": [f"{m*100:.2f}%" for m in means_input],
                        "Std deviation": [f"{s*100:.2f}%" for s in sigs_input],
                    })
                    st.dataframe(stats_df, hide_index=True)
                    st.markdown("**Correlation matrix**")
                    corr_df = pd.DataFrame(corr_input,
                                           index=names_input,
                                           columns=names_input)
                    st.dataframe(corr_df.round(3))
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
                data_valid = False
        else:
            st.info("Upload a CSV file to continue.")
            data_valid = False

    st.markdown("---")

    # ── Section 2: Derivative ─────────────────────────────────────────────────
    st.markdown("### 📊 Derivative / Structured product")
    der_label_selected = st.selectbox(
        "Type", list(DERIVATIVE_OPTIONS.keys()), index=0,
        label_visibility="collapsed")
    der_type = DERIVATIVE_OPTIONS[der_label_selected]

    der_params = {}
    if der_type in ("put", "call", "straddle"):
        default_strike = 1.4 if der_type == "put" else (1.2 if der_type == "call" else 0.7)
        der_params["strike"] = st.slider("Strike price", 0.5, 2.0, default_strike, 0.05)
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

    # ── Section 3: Mental-account constraint ──────────────────────────────────
    st.markdown("### 🎯 Mental-account constraint")
    H_val     = st.slider("Threshold H (%)", -25, -3, -10, 1) / 100
    alpha_val = st.slider("Shortfall probability α (%)", 1, 15, 5, 1) / 100

    st.markdown("---")

    # ── Section 4: Grid resolution ────────────────────────────────────────────
    st.markdown("### ⚡ Grid resolution")
    grid_label = st.selectbox("Resolution", list(GRID_OPTIONS.keys()), index=0,
                               label_visibility="collapsed")
    m_val, mp_val = GRID_OPTIONS[grid_label]
    if "High precision" in grid_label:
        st.markdown(
            '<div class="warning-box">⚠️ High precision may take 5–10 min. '
            'Recommended for final results only.</div>',
            unsafe_allow_html=True)
    elif "Standard" in grid_label:
        st.markdown(
            '<div class="warning-box">⏱️ Standard resolution takes ~1–2 min.</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    run_btn = st.button("▶ Run optimizer", type="primary",
                        use_container_width=True, disabled=not data_valid)


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
            "Configure parameters in the sidebar and click **▶ Run optimizer** "
            "to generate results.", icon="👈")
        st.stop()

    # Resolve inputs
    means_arr  = np.array(means_input)
    sigs_arr   = np.array(sigs_input)
    cov_matrix = corr_to_cov(sigs_arr, corr_input)
    der_config = build_derivative_config(der_type, der_params, sigs_arr) \
                 if der_type else None
    asset_labels = names_input + (["Derivative"] if der_config else [])

    # ── Run ──────────────────────────────────────────────────────────────────
    with st.spinner("Computing mean-variance frontier..."):
        mv_x, mv_y, mv_eq = compute_mv_frontier(means_arr, cov_matrix)

    with st.spinner("Running behavioral optimizer — no derivative..."):
        try:
            nd_xs, nd_ys, nd_lbls = build_behavioral_frontier(
                means_arr, sigs_arr, cov_matrix,
                None, alpha_val, m_val, mp_val)
        except Exception as e:
            st.error(f"Optimizer failed (no derivative): {e}")
            st.stop()

    cgn_xs, cgn_ys, cgn_lbls = [], [], []
    if der_config is not None:
        with st.spinner(f"Running behavioral optimizer — {der_label_selected}..."):
            try:
                cgn_xs, cgn_ys, cgn_lbls = build_behavioral_frontier(
                    means_arr, sigs_arr, cov_matrix,
                    der_config, alpha_val, m_val, mp_val)
            except Exception as e:
                st.warning(f"Could not compute derivative frontier: {e}")

    with st.spinner("Rendering chart..."):
        fig = plot_frontier(
            mv_x, mv_y, mv_eq,
            nd_xs, nd_ys, nd_lbls,
            cgn_xs, cgn_ys, cgn_lbls,
            der_label_selected, H_val, alpha_val,
            names_input)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Results at selected H ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### Optimal portfolio — H = {H_val:.0%}, α = {alpha_val:.0%}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Without derivative**")
        try:
            nd_res, nd_nsec = run_optimizer(
                means_arr, sigs_arr, cov_matrix,
                None, H_val, alpha_val, m_val, mp_val)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Expected return", f"{nd_res['expected_return']*100:.2f}%")
            m2.metric("Std deviation",   f"{nd_res['std_dev']*100:.2f}%")
            m3.metric("Skewness",        f"{nd_res['skewness']:.3f}")
            m4.metric("Shortfall prob",  f"{nd_res['shortfall_stat']*100:.2f}%")
            st.markdown("**Weights**")
            for i, w in enumerate(nd_res["weights"]):
                lbl = names_input[i] if i < len(names_input) else f"Asset {i+1}"
                st.progress(float(w), text=f"{lbl}: {w*100:.1f}%")
        except Exception as e:
            st.warning(f"No eligible portfolio found: {e}")

    with col2:
        if der_config is not None:
            st.markdown(f"**With {der_label_selected}**")
            try:
                cgn_res, cgn_nsec = run_optimizer(
                    means_arr, sigs_arr, cov_matrix,
                    der_config, H_val, alpha_val, m_val, mp_val)
                delta = (cgn_res['expected_return'] - nd_res['expected_return']) * 100
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Expected return",
                          f"{cgn_res['expected_return']*100:.2f}%",
                          delta=f"+{delta:.2f}pp" if delta > 0 else f"{delta:.2f}pp")
                m2.metric("Std deviation",  f"{cgn_res['std_dev']*100:.2f}%")
                m3.metric("Skewness",       f"{cgn_res['skewness']:.3f}")
                m4.metric("Shortfall prob", f"{cgn_res['shortfall_stat']*100:.2f}%")
                st.markdown("**Weights**")
                for i, w in enumerate(cgn_res["weights"]):
                    lbl = asset_labels[i] if i < len(asset_labels) else f"Asset {i+1}"
                    st.progress(float(w), text=f"{lbl}: {w*100:.1f}%")
            except Exception as e:
                st.warning(f"No eligible portfolio found: {e}")
        else:
            st.info("Select a derivative in the sidebar to compare.")

    # ── Data summary ──────────────────────────────────────────────────────────
    with st.expander("📋 Portfolio data used in this run"):
        stats_df = pd.DataFrame({
            "Asset":         names_input,
            "Mean return":   [f"{m*100:.2f}%" for m in means_input],
            "Std deviation": [f"{s*100:.2f}%" for s in sigs_input],
        })
        st.dataframe(stats_df, hide_index=True)
        corr_df = pd.DataFrame(corr_input,
                               index=names_input, columns=names_input)
        st.markdown("**Correlation matrix**")
        st.dataframe(corr_df.round(3))


with tab_about:
    st.markdown("## About this tool")
    st.markdown("""
<div class="info-box">

### Theoretical framework

This optimizer extends **Markowitz mean-variance theory** to portfolios including
derivatives and structured products, using a **mental-accounting downside constraint**:

> *The probability of the portfolio return falling below a threshold H must not exceed α.*

### Algorithm — three steps

**Step 1 — State space construction**
A discrete grid of return scenarios is built for all primary securities (m grid steps each).
For each scenario, derivative returns are computed analytically using Black-Scholes pricing.

**Step 2 — Probability assignment**
Each state is assigned a probability using a **Gaussian copula**, correctly capturing the
dependence structure between assets and supporting non-normal marginal distributions.

**Step 3 — Two-stage optimization**
- *Grid search*: All weight combinations are evaluated against the mental-account constraint.
- *Gradient refinement*: COBYLA nonlinear optimizer refines the solution from the best grid point.

### MVT / MAT equivalence

When **no derivatives** are present, the mean-variance and behavioral frontiers converge exactly.
For H = -10% and α = 5%, the implied risk-aversion is **λ = 3.795** — at which point both methods
yield identical optimal portfolios. Adding derivatives breaks this equivalence.

### Data input

Three modes are supported:
- **Default**: Das & Statman (2010) base case — 3 securities, pre-calibrated parameters
- **Manual entry**: Enter your own means, standard deviations, and correlations
- **CSV upload**: Upload historical price data — means and covariances are computed automatically

</div>

### Academic references

- **Das & Statman (2004)** — *Beyond Mean-Variance: Portfolios with Derivatives and Non-Normal Returns in Mental Accounts*
- **Das, Markowitz, Scheid & Statman (2010)** — *Portfolio Optimization with Mental Accounts*, JFQA Vol. 45, No. 2, pp. 311–334
- **Sami Jeddou (2012)** — *Beyond Mean-Variance: Options and Structured Products in Behavioral Portfolios*, MSc Finance Thesis, USI Lugano

### Author

**Sami Jeddou** — Senior Financial Services Transformation Leader

🔗 [LinkedIn](https://www.linkedin.com/in/sami-jeddou-25787a404)
🐙 [GitHub](https://github.com/SamiJeddou/behavioral-portfolio-optimizer)
""", unsafe_allow_html=True)
