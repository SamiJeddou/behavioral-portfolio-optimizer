"""
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
from scipy.optimize import minimize
from io import StringIO
from datetime import date, timedelta
from behavioral_portfolio_optimizer import (
    build_state_space, assign_probabilities, optimize_portfolio,
    compute_structured_payoff, bs_call, bs_put
)
from scipy.stats import norm as _norm
from scipy.optimize import brentq as _brentq

def implied_lambda(H, alpha, means, cov_matrix, lam_lo=0.01, lam_hi=100):
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
        return _brentq(f, lam_lo, lam_hi)
    except Exception:
        return None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Behavioral Portfolio Optimizer",
    page_icon="📈", layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
.main{background:#0d1117}.block-container{padding-top:1.5rem}
h1{color:#fff;font-size:1.6rem}h2,h3{color:#c0c8d8}
.info-box{background:#1a1a2e;border:1px solid #4a9eff;border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem;color:#ffffff !important}
.warn-box{background:#1a1200;border:1px solid #f59e0b;border-radius:6px;padding:.5rem 1rem;color:#f59e0b;font-size:.82rem;margin-top:.3rem}
.ok-box{background:#001a0f;border:1px solid #10b981;border-radius:6px;padding:.5rem 1rem;color:#10b981;font-size:.82rem;margin-top:.3rem}
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MEANS = [0.05, 0.10, 0.25]
DEFAULT_SIGS  = [0.05, 0.20, 0.50]
DEFAULT_CORR  = [[1.0,0.0,0.0],[0.0,1.0,0.4],[0.0,0.4,1.0]]
DEFAULT_NAMES = ["Sec 1 — Low risk","Sec 2 — Mid risk","Sec 3 — High risk"]

GRID_OPTIONS = {
    "⚡ Fast (m=21, m'=15)":           (21,15),
    "⚖️  Standard (m=35, m'=50)":      (35,50),
    "🎯 High precision (m=51, m'=99)": (51,99),
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

def parse_csv(f):
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    rets = df.pct_change().dropna()
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
        factor = 252 if freq == "Daily" else 12
        means = (rets.mean() * factor).tolist()
        sigs  = (rets.std() * np.sqrt(factor)).tolist()
        corr  = rets.corr().values.tolist()
        names = list(rets.columns)
        last_prices = raw.iloc[-1].to_dict()
        return means, sigs, corr, names, last_prices, None
    except Exception as e:
        return None, None, None, None, None, str(e)

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
    pts=[mv_opt(l) for l in np.linspace(0.3,25,120)]
    pts=[p for p in pts if p]
    eq=mv_opt(3.7950)
    return [p[0] for p in pts],[p[1] for p in pts],eq

def run_opt(means,sigs,cov,der_config,H,alpha,m,mp):
    U,dr=build_state_space(means,sigs,m=m,derivative_config=der_config)
    U=assign_probabilities(U,means,sigs,cov,dr)
    n=U.shape[1]-1
    res=optimize_portfolio(U,n,H=H,alpha=alpha,m_prime=mp)
    return res,n

def build_frontier(means,sigs,cov,der_config,alpha,m,mp):
    H_vals=[-0.05,-0.08,-0.10,-0.12,-0.15,-0.18,-0.20]
    xs,ys,lbls=[],[],[]
    for H in H_vals:
        try:
            r,_=run_opt(means,sigs,cov,der_config,H,alpha,m,mp)
            xs.append(r["std_dev"]*100); ys.append(r["expected_return"]*100)
            lbls.append(f"H={H:.0%}")
        except: pass
    return xs,ys,lbls

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
    st.markdown("### 📂 Portfolio data")
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
                m,s,c,n,lp,err=fetch_tickers(tickers,d_start,d_end,freq)
            if err:
                st.error(f"Fetch failed: {err}"); data_valid=False
            else:
                st.session_state["live_data"]=(m,s,c,n,lp)
                st.markdown(f'<div class="ok-box">✓ Loaded: {", ".join(n)}</div>',
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
    st.markdown("### 📊 Derivative / Structured product")
    der_label_sel=st.selectbox("Type",list(PREDEFINED_DERIVATIVES.keys()),
                                index=0,label_visibility="collapsed")
    der_type=PREDEFINED_DERIVATIVES[der_label_sel]
    der_params={}

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
    st.markdown("### 🎯 Mental-account constraint")
    H_val    =st.slider("Threshold H (%)",-25,-3,-10,1)/100
    alpha_val=st.slider("Shortfall probability α (%)",1,15,5,1)/100

    # Implied lambda — displayed right after H and alpha
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
                    'λ could not be computed for these parameters</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── 4. Grid ───────────────────────────────────────────────────────────────
    st.markdown("### ⚡ Grid resolution")
    grid_lbl=st.selectbox("Resolution",list(GRID_OPTIONS.keys()),
                           index=0,label_visibility="collapsed")
    m_val,mp_val=GRID_OPTIONS[grid_lbl]
    if "High" in grid_lbl:
        st.markdown('<div class="warn-box">⚠️ May take 5–10 min.</div>',
                    unsafe_allow_html=True)
    elif "Standard" in grid_lbl:
        st.markdown('<div class="warn-box">⏱️ ~1–2 min.</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    run_btn=st.button("▶ Run optimizer",type="primary",
                       use_container_width=True,disabled=not data_valid)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='margin-top:2.5rem'></div>", unsafe_allow_html=True)
tab1,tab2=st.tabs(["📊 Optimizer","📖 About"])

with tab1:
    st.markdown("## Behavioral Portfolio Optimizer")
    st.markdown("Extends Markowitz mean-variance theory to portfolios including "
                "**derivatives and structured products** using a "
                "**mental-accounting downside constraint**.")

    if not run_btn:
        st.markdown("""
<div class="info-box" style="color:#ffffff !important">

### 👈 How to use this tool

Follow these steps in the sidebar:

| Step | Action |
|---|---|
| **1 — Portfolio data** | Choose a data source: default base case, live market tickers, manual entry, or CSV upload |
| **2 — Derivative** | Select a derivative or structured product type (or build a custom one) |
| **3 — Product parameters** | Set the strike, maturity, floor, participation, or other characteristics |
| **4 — Constraint** | Set the mental-account threshold H and shortfall probability α |
| **5 — Grid resolution** | Choose Fast for a quick preview, High precision for thesis-level accuracy |
| **6 — Run** | Click **▶ Run optimizer** |

The chart will show three curves:
- 🔘 **Grey dashed** — classical mean-variance efficient frontier (Markowitz)
- 🔵 **Blue** — behavioral optimizer frontier without derivatives
- 🟡 **Gold** — behavioral optimizer frontier including your selected derivative

At the equivalence point (λ=3.795, H=-10%, α=5%), the grey and blue curves meet exactly —
confirming the MVT/MAT equivalence proven in Das, Markowitz, Scheid & Statman (2010).
The gold curve shows what the behavioral approach unlocks beyond what mean-variance can achieve.

</div>
""", unsafe_allow_html=True)
        # Sample chart
        import os
        if os.path.exists("sample_output.png"):
            st.image("sample_output.png", caption="Sample output — default base case with Capital-Guaranteed Note (CGN)", use_container_width=True)

        st.markdown("---")

        # LinkedIn + contact
        st.markdown("""
<div style="background:#0f1923;border:1px solid #4a9eff;border-radius:8px;padding:1rem 1.4rem;color:#ffffff">

**👤 About the author**

**Sami Jeddou** — Senior Financial Services Transformation Leader | Risk, Capital Markets & Front-to-Back Delivery

🔗 [Connect on LinkedIn](https://www.linkedin.com/in/sami-jeddou-25787a404) &nbsp;&nbsp;|&nbsp;&nbsp;
🐙 [View source on GitHub](https://github.com/SamiJeddou/behavioral-portfolio-optimizer)

</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        with st.form("contact_form"):
            st.markdown("**💬 Send a comment or question**")
            sender_name  = st.text_input("Your name")
            sender_email = st.text_input("Your email")
            message      = st.text_area("Message", height=100,
                                         placeholder="Questions, feedback, collaboration ideas...")
            submitted = st.form_submit_button("Send message")
            if submitted:
                if sender_name and sender_email and message:
                    import urllib.parse
                    subject = urllib.parse.quote(f"Behavioral Portfolio Optimizer — message from {sender_name}")
                    body    = urllib.parse.quote(f"From: {sender_name}\nEmail: {sender_email}\n\n{message}")
                    mailto  = f"mailto:sami.jeddou@protonmail.com?subject={subject}&body={body}"
                    st.markdown(
                        f'<meta http-equiv="refresh" content="0;url={mailto}">',
                        unsafe_allow_html=True)
                    st.success("✓ Opening your email client to send the message.")
                else:
                    st.warning("Please fill in all fields before sending.")

        st.stop()

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

    with st.spinner("Behavioral optimizer — no derivative..."):
        try:
            nd_xs,nd_ys,nd_lbls=build_frontier(
                means_arr,sigs_arr,cov_mat,None,alpha_val,m_val,mp_val)
        except Exception as e:
            st.error(f"Optimizer failed: {e}"); st.stop()

    der_xs,der_ys,der_lbls=[],[],[]
    if der_config:
        with st.spinner(f"Behavioral optimizer — {der_label_sel}..."):
            try:
                der_xs,der_ys,der_lbls=build_frontier(
                    means_arr,sigs_arr,cov_mat,der_config,alpha_val,m_val,mp_val)
            except Exception as e:
                st.warning(f"Derivative frontier failed: {e}")

    with st.spinner("Rendering chart..."):
        fig=plot_frontier(mv_x,mv_y,mv_eq,nd_xs,nd_ys,nd_lbls,
                          der_xs,der_ys,der_lbls,der_label_sel,H_val,alpha_val)
        st.pyplot(fig,use_container_width=True); plt.close(fig)

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### Optimal portfolio — H={H_val:.0%}, α={alpha_val:.0%}")
    c1,c2=st.columns(2)

    nd_res=None
    with c1:
        st.markdown("**Without derivative**")
        try:
            nd_res,_=run_opt(means_arr,sigs_arr,cov_mat,None,H_val,alpha_val,m_val,mp_val)
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Return",f"{nd_res['expected_return']*100:.2f}%")
            m2.metric("Std dev",f"{nd_res['std_dev']*100:.2f}%")
            m3.metric("Skewness",f"{nd_res['skewness']:.3f}")
            m4.metric("Shortfall",f"{nd_res['shortfall_stat']*100:.2f}%")
            st.markdown("**Weights**")
            for i,w in enumerate(nd_res["weights"]):
                lbl=names_in[i] if i<len(names_in) else f"Asset {i+1}"
                st.progress(float(w),text=f"{lbl}: {w*100:.1f}%")
            st.caption(f"Method: {nd_res.get('method_used','—')}")
        except Exception as e:
            st.warning(f"No eligible portfolio: {e}")

    with c2:
        if der_config:
            st.markdown(f"**With {der_label_sel}**")
            try:
                dr_res,_=run_opt(means_arr,sigs_arr,cov_mat,der_config,
                                  H_val,alpha_val,m_val,mp_val)
                delta=(dr_res['expected_return']-(nd_res['expected_return'] if nd_res else 0))*100
                m1,m2,m3,m4=st.columns(4)
                m1.metric("Return",f"{dr_res['expected_return']*100:.2f}%",
                           delta=f"+{delta:.2f}pp" if delta>0 else f"{delta:.2f}pp")
                m2.metric("Std dev",f"{dr_res['std_dev']*100:.2f}%")
                m3.metric("Skewness",f"{dr_res['skewness']:.3f}")
                m4.metric("Shortfall",f"{dr_res['shortfall_stat']*100:.2f}%")
                st.markdown("**Weights**")
                for i,w in enumerate(dr_res["weights"]):
                    lbl=asset_labels[i] if i<len(asset_labels) else f"Asset {i+1}"
                    st.progress(float(w),text=f"{lbl}: {w*100:.1f}%")
                st.caption(f"Method: {dr_res.get('method_used','—')}")
            except Exception as e:
                st.warning(f"No eligible portfolio: {e}")
        else:
            st.info("Select a derivative to compare.")

    with st.expander("📋 Portfolio data used in this run"):
        st.dataframe(pd.DataFrame({
            "Asset":names_in,
            "Mean return":[f"{m*100:.2f}%" for m in means_in],
            "Std deviation":[f"{s*100:.2f}%" for s in sigs_in],
        }),hide_index=True)
        st.markdown("**Correlation matrix**")
        st.dataframe(pd.DataFrame(corr_in,index=names_in,
                                   columns=names_in).round(3))

with tab2:
    st.markdown("## About this tool")
    st.markdown("""
<div class="info-box">

### Theoretical framework

Extends **Markowitz mean-variance theory** to portfolios including derivatives and structured
products using a **mental-accounting downside constraint**:

> *P(portfolio return < H) ≤ α*

### Algorithm

**Step 1** — Discrete state space built via grid over primary security returns.
Derivative payoffs computed analytically using Black-Scholes.

**Step 2** — Probabilities assigned via Gaussian copula.

**Step 3** — Two-stage optimization:
- *≤ 4 securities*: exhaustive grid search + COBYLA refinement
- *≥ 5 securities*: differential evolution (global stochastic) + COBYLA refinement

### Data input modes
- **Default**: Das & Statman (2010) base case — reproduces thesis results exactly
- **Live market data**: fetch any global ticker from Yahoo Finance, daily or monthly returns
- **Manual entry**: enter your own means, std devs, correlations for 2–10 securities
- **CSV upload**: upload historical price data — statistics computed automatically

### Derivatives & structured products
- 9 predefined types (puts, calls, collars, straddles, strangles, CGNs, barrier notes)
- **Custom composer**: build any structured product from calls, puts, digitals, and zero-coupon bonds
  with live payoff diagram preview

</div>

### References
- **Das & Statman (2004)** — *Beyond Mean-Variance: Portfolios with Derivatives and Non-Normal Returns in Mental Accounts*
- **Das, Markowitz, Scheid & Statman (2010)** — *Portfolio Optimization with Mental Accounts*, JFQA Vol. 45, No. 2, pp. 311–334
- **Sami Jeddou (2012)** — *Beyond Mean-Variance: Options and Structured Products in Behavioral Portfolios*, MSc Thesis, USI Lugano

### Author
**Sami Jeddou** — Senior Financial Services Transformation Leader

🔗 [LinkedIn](https://www.linkedin.com/in/sami-jeddou-25787a404)
🐙 [GitHub](https://github.com/SamiJeddou/behavioral-portfolio-optimizer)
""", unsafe_allow_html=True)
