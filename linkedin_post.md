🚀 **Sharing a project I've been building — an interactive app that goes beyond classical mean-variance portfolio theory, grounded in academic research.**

**Beyond Mean-Variance: Portfolio Optimisation with Derivatives & Structured Products — A Mental Accounting Framework** extends the groundwork of Markowitz (1952) on mean-variance optimisation and the mental-accounting portfolio theory of Das & Statman (2009) and Das, Markowitz, Scheid & Statman (2010) — applying them to settings where closed-form solutions don't exist and numerical optimisation is required.

It's built on the research I first developed in my MSc Finance thesis at USI Lugano (2012), originally implemented in R, and now fully reimplemented and extended in Python — with interactive, AI-powered capabilities, including a glossary that explains derivatives, risk measures and portfolio-theory concepts in plain language.

**What it does**

📊 Optimises portfolios that mix traditional assets, crypto (BTC, ETH…) and a library of **16 derivatives & structured products** — puts, calls, collars, straddles, spreads, capital-guaranteed and barrier notes, certificates — plus a builder to configure your own.

🎯 Optimises under a downside-risk constraint you choose, across **four risk-based methods** — Value-at-Risk, thesis Expected Shortfall, a rigorous-ES mode, and a scalable Monte-Carlo CVaR — grounded in the mental-accounting idea that investors set safety-first thresholds rather than simply minimising variance.

🔗 Shows, **analytically and visually**, the **MVT/MAT equivalence** (Mean-Variance ↔ Mental Accounts) proven by Das, Markowitz, Scheid & Statman (2010): for a chosen loss threshold H and shortfall probability α there is an implied risk-aversion λ at which both approaches give the same optimal portfolio — when no derivatives are present. Add derivatives and the behavioural approach can reach portfolios mean-variance cannot — with the hedge embedded *inside* the optimisation rather than bolted on as a separate overlay.

📈 Runs on live market data with automatic cleaning, and explains every result in plain language.

**What's new**

⚡ A **scalable Monte-Carlo + CVaR engine** that takes the method to institutional scale. The exact grid is precise but its state space grows as mᴺ — impractical past a handful of assets. The new engine samples joint scenarios through a copula and solves the goal as a linear program: cost linear in the number of assets, several derivatives at once (even on different underlyings), Student-t copulas for tail dependence (assets crashing together), and a coherent Expected-Shortfall floor via Rockafellar–Uryasev.

🔬 An **out-of-sample back-test**: build the weights on a construction window, then buy-and-hold them through a later window with the derivative marked to market — to see whether what the model expected actually held, and to read off the realised **alpha and beta** of each holding and of the portfolio against a benchmark of your choice.

🔌 **Callable beyond the app** — the optimisation **and** back-testing engines are exposed through a REST API and an MCP server, so external portfolio, risk and trading systems — or an AI agent — can call them directly, not just through this interface.

The optimiser returns **up to four portfolios** side by side — the Markowitz mean-variance optimum, the behavioural optimum without derivatives, the optimum with derivatives, and a same-risk comparison — so you can see exactly what the framework adds.

🔗 Live app, the 2012 USI Lugano thesis, and a step-by-step user guide — links in the first comment 👇

I'd genuinely value your thoughts — happy to connect and discuss with anyone interested in quantitative finance, risk, derivatives or portfolio construction. Feel free to reach out.

#QuantitativeFinance #PortfolioOptimisation #RiskManagement #Derivatives #StructuredProducts #Python #MentalAccounting #CVaR #MCP #FinTech
