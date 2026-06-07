# Behavioral Portfolio Optimizer

> Extending Markowitz mean-variance theory to portfolios with derivatives and structured products using a mental-accounting framework.

![Portfolio Optimiser Output — annotated sample showing each section of a run (chart, summary of portfolios, and per-portfolio details); illustrative only, not live results](sample_output_annotated.png)

---

## Overview

This project implements a **behavioural portfolio optimisation algorithm** that goes beyond classical mean-variance theory by:

- Incorporating **derivatives and structured products** (puts, calls, collars, straddles, strangles, capital-guaranteed notes, barrier-M notes) directly into the optimisation
- Using a **mental-accounting framework** with a downside risk constraint: the probability of the portfolio return falling below a threshold H must not exceed α
- Handling **non-normal return distributions** via Gaussian and Student-t copulas
- Proving the **equivalence between mean-variance and mental-accounting** optimisation at a given implied risk-aversion coefficient λ
- Embedding a **hedging benefit** within the portfolio construction process itself — incorporating derivatives can simultaneously improve expected return and provide downside protection, with the optimal derivative weight endogenously determined by the downside constraint rather than imposed as an external hedge ratio

The optimisation produces up to four portfolios for comparison:

- **Portfolio (0) — Markowitz mean-variance optimum (no derivative)** — the minimum-variance portfolio at Portfolio (1)'s expected return. It coincides with Portfolio (1) when Portfolio (1) is mean-variance efficient, directly demonstrating the MVT/MAT equivalence (shown whenever Portfolio (1) exists)
- **Portfolio (1) — Behavioural optimum without derivatives at the chosen constraint (H, α)** — mean-variance efficient via the mental-accounting framework; coincides with Portfolio (0) when the implied λ equals 3.795 (the MVT/MAT equivalence)
- **Portfolio (2) — Behavioural optimum with derivative, same mental-accounting & risk-aversion constraint (H, α ↔ λ)** — may reach higher expected returns by exploiting asymmetric derivative payoffs
- **Portfolio (3) — Portfolio with derivative, same variance as Portfolio (1)** — interpolated from the derivative frontier at an equivalent risk level (indicative only)

Under the default base case (H = -10%, α = 5%), the no-derivative behavioural optimum returns **10.2%**. Adding a derivative gives a modest improvement under the same downside constraint: an uncapped Capital-Guaranteed Note reaches about **11.4%** (+1.2pp), while the largest gain among the available instruments — a straddle — reaches about **12.1%** (+1.9pp).

The threshold H ranges from -40% to -1%, making the framework applicable to highly volatile assets including **cryptocurrencies and digital assets**, emerging market equities, and other non-traditional instruments — extending the mental-accounting approach to today's broader investment universe.

Beyond the exact grid, the optimiser also includes a **scalable Monte-Carlo + CVaR engine** for institutional-size portfolios, and an **out-of-sample back-test** to check whether a chosen portfolio's modelled expectations hold on later data. Both are described under *Algorithm* below.

---

## Theoretical Background

This work is based on the mental-accounting portfolio theory introduced in:

- **Das, Sanjiv and Meir Statman (2009)** — *Beyond Mean-Variance: Portfolios with Derivatives and Non-Normal Returns in Mental Accounts*
- **Das, Sanjiv, Harry Markowitz, Jonathan Scheid and Meir Statman (2010)** — *"Portfolio Optimization with Mental Accounts"*, Journal of Financial and Quantitative Analysis, Vol. 45, No. 2, pp. 311–334

The MVT/MAT equivalence — first proven in Das, Markowitz, Scheid & Statman (2010) JFQA and applied in Chapter 4 of Jeddou (2012) — shows that for a given threshold H and shortfall probability α, there exists an implied risk-aversion coefficient λ such that the mean-variance optimal portfolio and the behavioural optimal portfolio are identical — **when no derivatives are present**. Adding derivatives breaks this equivalence and reveals the superiority of the behavioural approach.

This Python implementation is based on the original R program developed as part of:

> **Sami Jeddou** (2012) — *"Beyond Mean-Variance: Options and Structured Products in Behavioral Portfolios"*, Master in Finance Thesis, Università della Svizzera italiana (USI Lugano), supervised by Prof. Enrico De Giorgi. [PDF — USI institutional repository](https://thesis.bul.sbu.usi.ch/theses/1012-1112BenJeddou/pdf?1390987439)

The thesis extended the empirical analysis of Das & Statman (2009) through additional derivative simulations and broader parameter analysis. This app further develops that work with live market data connectivity, an expanded derivative library, an interactive optimisation interface, and PDF export.

---

## Algorithm

The optimiser runs in three steps:

**Step 1 — State space construction**
A discrete grid of return scenarios is built for all primary securities. For each scenario, derivative returns are computed analytically using Black-Scholes pricing. The result is a matrix U of all possible return vectors across m^n′ states.

**Step 2 — Probability assignment**
Each state is assigned a probability using a Gaussian (or Student-t) copula, correctly capturing the dependence structure between assets including non-normal marginals.

**Step 3 — Two-stage optimisation**
- *Grid search*: All weight combinations are evaluated. Those satisfying the mental-account constraint (VaR or ES) are kept as eligible. The highest-return eligible portfolio is selected as the starting point.
- *Gradient refinement*: A COBYLA nonlinear optimiser refines the solution from that starting point, with the constraint embedded as a penalty term.

### Scaling to large portfolios — Monte-Carlo + CVaR

The exact grid above is precise, but its state space grows as *m^n'* and becomes impractical beyond a handful of assets. A second, **scalable engine** is included for institutional-size portfolios:

- **Scenario generation** — joint return and derivative-payoff scenarios are sampled through a copula (Gaussian or Student-t). The Student-t copula captures tail dependence (assets crashing together).
- **CVaR linear program** — the goal is solved as a Rockafellar–Uryasev CVaR linear program, so cost grows *linearly* in the number of assets and several derivatives can be optimised at once, even on different underlyings.
- **Smooth frontier** — the frontier is swept with common random numbers so points are directly comparable.

This engine uses an **α-CVaR** objective; it is a scalable complement to the exact grid rather than a bit-for-bit reproduction of it.

### Out-of-sample back-test

To test the *efficiency* of each optimisation method — not just its in-sample fit — the app can build portfolio weights on a construction window and then **buy-and-hold** those weights through a later, out-of-sample window, with any derivative marked to market, comparing expected against realised outcomes. It also reports the realised **alpha, beta and R²** of each security and of the portfolio against a benchmark you select (S&P 500, global ACWI, a 60/40 SPY-AGG blend, or any ticker), with an optional expected-market-return input that adds a CAPM required return and an ex-ante alpha.

---

## Constraint Methods & Resolutions

There are two independent choices — the **constraint method** (what downside rule is enforced) and the **resolution / solver** (how the optimiser searches). Two routing conditions can override the resolution choice: the number of securities, and whether a derivative is present.

### The three constraint / objective methods

| Method | What it optimises | Best / recommended for |
|---|---|---|
| **VaR** (Method I) | max E[r] s.t. P(r < H) ≤ α — a probability-of-shortfall threshold | The thesis's primary method; most cases |
| **ES — thesis-faithful** (default Method II) | ES-eligible grid seed, but the COBYLA refinement still targets the **VaR** penalty — faithfully reproduces the original R thesis | Reproducing the thesis tables exactly |
| **Rigorous ES** | max E[r] s.t. ES ≥ L, with a genuinely **ES-aware** COBYLA penalty | Real decision-making — recovers up to ~2.4pp of E[r] the thesis method leaves unused (e.g. L = −15%: 15.5% vs 13.2%) |

### The four resolutions / solvers — and where each applies

| Resolution | VaR | ES (thesis) | Rigorous ES | Grid (m / m') | Speed / reliability | Best for |
|---|---|---|---|---|---|---|
| **Fast** | ✓ | ✓ | — | 21 / 15 | fastest; coarse, visible discretisation error | quick previews |
| **Standard** | ✓ | ✓ | — | 35 / 50 | moderate; safe with derivatives | daily work, derivative cases |
| **High precision** | ✓ | ✓ | — | 51 / 99 | slow (~15–30 min full frontier); thesis-grade | publication numbers, validation, derivative cases |
| **Turbo** | ✓ *(n ≤ 4, no-derivative)* | ✗ | — | 51, coarse-to-fine | ~seconds (~60× faster than High); **unreliable with a derivative** (up to 32% disagreement) | fast no-derivative VaR frontier exploration |
| **Rigorous-ES** (own mode, resolution fixed) | — | — | ✓ | 51 (fixed) | ~seconds; ES-aware | ES decision-making |

*Legend: ✓ available · ✗ deliberately disabled · — not applicable (separate fixed-resolution mode).*

**Routing rules that override the resolution choice.** Fast / Standard / High serve both **VaR** and **thesis-ES**; **Turbo** is **VaR-only** and live only for **≤ 4 total securities with no derivative** (it is hidden for ES and for 5+ securities); **Rigorous-ES** is a separate mode whose resolution is fixed at m = 51. The derivative counts toward the security total: **n ≤ 4 → exhaustive grid search**, **n ≥ 5 → differential evolution** (a stochastic global optimiser). Only Turbo's and Rigorous-ES's coarse-to-fine seeding is exposed to derivative basin-miss errors; the exhaustive-grid resolutions are immune to that and limited only by grid coarseness.

---

## Supported Derivatives & Structured Products

The library includes **16 predefined instruments plus a custom composer**:

| Type | Description |
|---|---|
| Put / Call | Standard European options |
| Safety collar | Long put + short call |
| Aggressive collar | Long call + short put |
| Straddle / Strangle | Long call + long put (same or different strikes) |
| Capital-guaranteed note | Uncapped or capped, with floor and participation rate |
| Barrier-M note | Corridor note with digital components |
| Bull call spread | Long call + short higher call — bullish, capped, lower cost than a call |
| Bear put spread | Long put + short lower put — cheaper bearish hedge, capped |
| Long butterfly (calls) | Long–short²–long calls — low-volatility "pin" bet, very cheap |
| Call condor | Four-strike range bet with a flat maximum payoff between the inner strikes |
| Reverse convertible | Zero-coupon bond − short put — high coupon, capped upside, principal at risk |
| Discount certificate | Synthetic underlying − short call — bought at a discount, upside capped |
| Outperformance certificate | Synthetic underlying + extra call — full downside, geared (>100%) upside |
| Custom composer | Build any payoff from calls, puts, digitals, and zero-coupon bonds |

---

## Project Structure

```
behavioral-portfolio-optimizer/
│
├── behavioral_portfolio_optimizer.py   # Core optimizer (Steps 1–3 + all derivative types)
├── app.py                              # Streamlit interactive dashboard
├── requirements.txt                    # Python dependencies
├── efficient_frontier_v2.png           # Frontier chart (used in README)
└── README.md
```

---


## Data Input

Three modes are supported for portfolio data:

| Mode | Description |
|---|---|
| **Default** | Das & Statman (2009) base case — 3 securities with pre-calibrated means, std devs, and correlations. Works out of the box, reproduces thesis results exactly. |
| **Live market data** | Fetch any global ticker from Yahoo Finance — stocks, ETFs, indices, and crypto (e.g. BTC-USD, ETH-USD). Select a date range and choose daily or monthly return frequency. Means and covariances are computed automatically. Data is automatically cleaned: stale price rows (zero returns) are removed and outliers beyond ±5 standard deviations are winsorised. |
| **Manual entry** | Enter your own means, standard deviations, and correlation matrix directly in the sidebar. Supports 2–10 primary securities. |
| **CSV upload** | Upload a CSV of historical prices (date column + one column per asset). Means and covariances are computed automatically. The same data cleaning (stale price removal, ±5σ winsorisation) is applied as for live market data. |

### CSV format

```
Date,Asset1,Asset2,Asset3
2020-01-02,100.00,100.00,100.00
2020-01-03,100.05,100.15,100.40
```

First column must be dates. Remaining columns are asset prices with the asset name as the header. A sample CSV is available for download directly in the app.

---

## Quickstart

### Run locally

```bash
# Install dependencies
pip install numpy scipy matplotlib streamlit fastapi uvicorn

# Run the optimiser directly
python behavioral_portfolio_optimizer.py

# Launch the interactive dashboard
streamlit run app.py

```

### Interactive dashboard

The Streamlit dashboard allows you to:
- Select derivative type from a dropdown
- Adjust the mental-account threshold H and shortfall probability α via sliders
- Visualise the three-curve efficient frontier (MV / Behavioral / Behavioral + derivative) in real time
- Read optimal portfolio weights and statistics for the selected parameters

🔗 **Live app**: [sami-jeddou-behavioral-portfolio-optimizer.streamlit.app](https://sami-jeddou-behavioral-portfolio-optimizer.streamlit.app/?view=home)

📄 **[User Guide (PDF)](https://raw.githubusercontent.com/SamiJeddou/behavioral-portfolio-optimizer/main/Beyond_Mean_Variance_Portfolio_Optimiser_User_Guide.pdf)** — step-by-step guide to using the app

### API

The FastAPI endpoint exposes the optimiser as a REST service:

```bash
POST /optimize
{
  "derivative_type": "cgn",
  "H": -0.10,
  "alpha": 0.05,
  "floor": 0.01,
  "participation": 1.0,
  "cap": null
}
```

---

## Key Results

| Configuration | Expected Return | Std Dev | Skewness |
|---|---|---|---|
| No derivative (H=-10%, α=5%) | 10.21% | 12.29% | 0.00 |
| With CGN — floor=0%, uncapped (H=-10%, α=5%) | 11.36% | 20.50% | — |
| Best derivative — straddle (H=-10%, α=5%) | 12.12% | 15.90% | — |
| Equivalence point: λ=3.795 ↔ H=-10%, α=5% | 10.23% | 12.30% | — |

The baseline result (10.21%) matches the thesis mean-variance result (10.23%) to within **2 basis points**, confirming correct algorithm calibration.

---

## ✅ Recently added

| Feature | Status |
|---|---|
| Scalable Monte-Carlo + CVaR engine (copula scenarios → CVaR linear program) for institutional-size portfolios | ✔ Live |
| Out-of-sample back-test of optimised portfolios, with realised alpha / beta vs a chosen benchmark | ✔ Live |
| Rigorous-ES mode (genuinely ES-aware optimisation) | ✔ Live |
| Expanded library — 16 instruments + custom structured-product composer | ✔ Live |

## 🔮 Coming Soon

| Feature | Status |
|---|---|
| Multi-period / multi-horizon optimisation | 🔜 Planned |
| Productionised REST API & async job handling for institutional workflows | 🔜 Planned |
| Further structured-product templates | 🔜 Planned |

## Author

<img src="profile.jpeg" width="80" align="left" style="border-radius:50%;margin-right:16px;margin-bottom:8px"/>

**Sami Jeddou**
Senior Financial Services Executive — Transformation, Risk & Capital Markets | Risk · Capital Markets · Post-Trade & Clearing · High-Value Payments · Quantitative Finance · Front-to-Back Delivery · Regulatory Programs

- 🔗 [LinkedIn](https://www.linkedin.com/in/sami-jeddou-25787a404)
- 📧 sami.jeddou@protonmail.com

---

## ⚠️ Disclaimer

This application is based on the mental accounts portfolio optimisation framework of Das & Statman (2009) and Das, Markowitz, Scheid & Statman (2010), as extended in Jeddou (2012) through additional derivative simulations and parameter analysis. The app further develops this work with live market data connectivity, an expanded derivative library, and an interactive optimisation interface.

It is provided for **educational and research purposes only** and does not constitute financial advice, investment recommendations, or a solicitation to buy or sell any financial instrument. Results are purely illustrative and should not be used as the basis for any investment decision. Past performance and modelled outputs are not indicative of future results.

The framework is designed to be extensible — future versions may incorporate additional derivative structures, alternative risk measures, and API connectivity for institutional workflows.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
