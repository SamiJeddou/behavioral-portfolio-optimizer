# © 2026 Sami Jeddou. All rights reserved.
# Published publicly for demonstration and evaluation only — no license is granted.
"""Ticker analytics — CFA-style ratio catalogue, formatting and neutral explanations.

UI-free. Pulls fields from a yfinance ``Ticker(...).info`` dict and turns them into a
grouped set of valuation / profitability / leverage / risk ratios, each with a
plain-language, *ticker-specific but strictly neutral* explanation.

IMPORTANT — educational only. Nothing here scores, ranks or recommends an instrument.
Explanations describe what a ratio measures and how to read it; they never give a
buy/sell signal, price target or suitability judgement.
"""

CATEGORIES = ["Valuation", "Profitability", "Leverage & solvency", "Risk & market"]


def _money(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    a = abs(v)
    for thr, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= thr:
            return f"${v / thr:.2f}{suf}"
    return f"${v:,.0f}"


def _fmt(kind, v):
    """Format a raw value; return '—' when missing/unusable."""
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if kind == "mult":
        return f"{v:.1f}×"
    if kind == "pct":          # raw is a fraction (0.45 -> 45.0%)
        return f"{v * 100:.1f}%"
    if kind == "num":
        return f"{v:.2f}"
    if kind == "d2e":          # yfinance debtToEquity is a percent-like number (150 -> 1.50)
        return f"{v / 100:.2f}"
    if kind == "money":
        return _money(v)
    return f"{v:g}"


# Each ratio: category, label, yfinance .info key, format kind, short gauge, and a
# neutral explanation template ({t} = ticker, {v} = formatted value).
RATIOS = [
    # ── Valuation ──────────────────────────────────────────────────────────────
    {"cat": "Valuation", "label": "Trailing P/E", "key": "trailingPE", "fmt": "mult",
     "gauge": "Price vs. past earnings",
     "explain": "{t}'s trailing P/E of {v} means the market pays about that many dollars for each "
                "$1 of the company's earnings over the last twelve months. A higher multiple usually "
                "reflects higher growth expectations or a richer price; it is only meaningful next to "
                "the company's own history and its sector peers, and is not in itself a buy or sell signal."},
    {"cat": "Valuation", "label": "Forward P/E", "key": "forwardPE", "fmt": "mult",
     "gauge": "Price vs. expected earnings",
     "explain": "{t}'s forward P/E of {v} uses analysts' expected earnings for the year ahead rather "
                "than past earnings. Comparing it with the trailing P/E indicates whether profits are "
                "expected to rise (forward below trailing) or fall (forward above trailing)."},
    {"cat": "Valuation", "label": "Price / Book", "key": "priceToBook", "fmt": "mult",
     "gauge": "Price vs. net assets",
     "explain": "A price-to-book of {v} compares {t}'s market value with its accounting net assets "
                "(book value). Asset-heavy businesses often trade near 1–3×, while asset-light or "
                "high-return businesses trade higher; on its own it says nothing about quality."},
    {"cat": "Valuation", "label": "Price / Sales", "key": "priceToSalesTrailing12Months", "fmt": "mult",
     "gauge": "Price vs. revenue",
     "explain": "{t}'s price-to-sales of {v} values the company against its revenue rather than profit, "
                "which is useful when earnings are thin or volatile. Because margins differ widely, it "
                "is only comparable within the same industry."},
    {"cat": "Valuation", "label": "EV / EBITDA", "key": "enterpriseToEbitda", "fmt": "mult",
     "gauge": "Whole-firm value vs. cash earnings",
     "explain": "Enterprise-value-to-EBITDA of {v} values {t} including its debt against pre-tax "
                "operating cash earnings. By including debt it makes companies with different capital "
                "structures easier to compare than P/E alone."},
    {"cat": "Valuation", "label": "Dividend yield", "key": "trailingAnnualDividendYield", "fmt": "pct",
     "gauge": "Income vs. price",
     "explain": "{t}'s dividend yield of {v} is the last twelve months' dividends as a share of the "
                "current price. A higher yield can reflect a steady income return or simply a depressed "
                "price; it is one income characteristic, not a measure of quality."},
    # ── Profitability ──────────────────────────────────────────────────────────
    {"cat": "Profitability", "label": "Return on equity (ROE)", "key": "returnOnEquity", "fmt": "pct",
     "gauge": "Profit per $ of equity",
     "explain": "Return on equity of {v} shows the profit {t} generates for each dollar of "
                "shareholders' equity. Higher is generally more efficient, but leverage can inflate it, "
                "so it is best read alongside the debt ratios."},
    {"cat": "Profitability", "label": "Return on assets (ROA)", "key": "returnOnAssets", "fmt": "pct",
     "gauge": "Profit per $ of assets",
     "explain": "Return on assets of {v} measures profit per dollar of total assets — a leverage-neutral "
                "view of how productively {t} uses its asset base."},
    {"cat": "Profitability", "label": "Gross margin", "key": "grossMargins", "fmt": "pct",
     "gauge": "Revenue kept after direct costs",
     "explain": "A gross margin of {v} is the share of revenue {t} keeps after the direct cost of its "
                "products or services — a gauge of pricing power and unit economics."},
    {"cat": "Profitability", "label": "Operating margin", "key": "operatingMargins", "fmt": "pct",
     "gauge": "Core-business profitability",
     "explain": "Operating margin of {v} is the share of revenue left after operating costs but before "
                "interest and tax, isolating the profitability of {t}'s core business."},
    {"cat": "Profitability", "label": "Net profit margin", "key": "profitMargins", "fmt": "pct",
     "gauge": "Bottom-line profitability",
     "explain": "Net margin of {v} is the share of revenue {t} keeps as bottom-line profit after all "
                "costs, interest and tax."},
    # ── Leverage & solvency ────────────────────────────────────────────────────
    {"cat": "Leverage & solvency", "label": "Debt / Equity", "key": "debtToEquity", "fmt": "d2e",
     "gauge": "Borrowing vs. equity",
     "explain": "A debt-to-equity of {v} compares {t}'s total debt with shareholders' equity. Higher "
                "leverage can amplify returns in good times and deepen losses in bad ones; what counts "
                "as high varies a lot by industry."},
    {"cat": "Leverage & solvency", "label": "Current ratio", "key": "currentRatio", "fmt": "num",
     "gauge": "Short-term assets vs. liabilities",
     "explain": "A current ratio of {v} compares {t}'s short-term assets with its short-term "
                "liabilities. Around 1 or above suggests it can cover near-term obligations; a very high "
                "figure can signal idle cash."},
    {"cat": "Leverage & solvency", "label": "Quick ratio", "key": "quickRatio", "fmt": "num",
     "gauge": "Liquidity excluding inventory",
     "explain": "The quick ratio of {v} is a stricter liquidity test than the current ratio: it "
                "excludes inventory to focus on the assets {t} could turn to cash most quickly."},
    # ── Risk & market ──────────────────────────────────────────────────────────
    {"cat": "Risk & market", "label": "Beta", "key": "beta", "fmt": "num",
     "gauge": "Sensitivity to the market",
     "explain": "A beta of {v} measures how {t}'s price has moved relative to the overall market: 1 "
                "moves with the market, above 1 is more volatile and below 1 less. It describes past "
                "co-movement, not a guarantee of future risk."},
    {"cat": "Risk & market", "label": "52-week range position", "key": "_range52", "fmt": "pct",
     "gauge": "Where price sits in its year",
     "explain": "{t} is trading at {v} of its 52-week high–low range, showing where today's price sits "
                "between the year's extremes. It is a context gauge, not a signal."},
    {"cat": "Risk & market", "label": "Market cap", "key": "marketCap", "fmt": "money",
     "gauge": "Company size",
     "explain": "{t}'s market capitalisation of {v} is the total market value of its shares — a measure "
                "of company size, which tends to track liquidity and volatility characteristics."},
]


# Plain-language, neutral explanations for the figures shown in the charts
# (revenue, cash-flow and balance-sheet items) — used by the "Explain" selector.
# Margins are already covered by the ratio catalogue above. {t} = ticker.
CHART_TERMS = {
    "Revenue": "{t}'s revenue (sales) is the total income it earns from its products and services "
               "before any costs are subtracted. The chart shows the annual trend over recent fiscal years.",
    "Net income": "Net income is {t}'s bottom-line profit after all costs, interest and tax. Set against "
                  "revenue, it shows how much of each sales dollar is kept as profit.",
    "Operating cash flow": "Operating cash flow is the cash {t} generates from its core operations. It "
                           "strips out non-cash accounting items, so it is often a cleaner read on cash "
                           "generation than reported profit.",
    "Free cash flow": "Free cash flow is {t}'s operating cash flow minus capital expenditure — the cash "
                      "left after maintaining and growing its asset base, available for debt, dividends or buybacks.",
    "Total assets": "Total assets is everything {t} owns — cash, receivables, inventory, property and "
                    "intangibles. By the accounting identity it always equals total liabilities plus equity.",
    "Total debt": "Total debt is {t}'s interest-bearing borrowings, short- and long-term. It is one input "
                  "to leverage; what counts as high varies a lot by industry.",
    "Cash": "Cash and equivalents is {t}'s most liquid asset — funds readily available. It sits within "
            "total assets and is often compared with debt to gauge net debt.",
    "Shareholders' equity": "Shareholders' equity is the residual value left for {t}'s owners after "
                            "subtracting all liabilities from total assets — its net worth on the books.",
}


def _range52(info):
    """Position of the current price within the 52-week low-high band, as a fraction [0,1]."""
    lo = info.get("fiftyTwoWeekLow")
    hi = info.get("fiftyTwoWeekHigh")
    px = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    try:
        lo, hi, px = float(lo), float(hi), float(px)
        if hi <= lo:
            return None
        return max(0.0, min(1.0, (px - lo) / (hi - lo)))
    except (TypeError, ValueError):
        return None


def company_header(ticker, info):
    """Compact header facts for the top of the analytics view."""
    import datetime as _dt
    px = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    cur = info.get("currency", "")
    # Quote timestamp ('as of' date/time) when Yahoo provides it; else retrieval date.
    _ts = info.get("regularMarketTime")
    asof, asof_is_market = None, False
    if _ts:
        try:
            asof = _dt.datetime.fromtimestamp(int(_ts)).strftime("%d %b %Y, %H:%M")
            asof_is_market = True
        except Exception:
            asof = None
    if not asof:
        asof = _dt.date.today().strftime("%d %b %Y")
    return {
        "ticker": ticker.upper(),
        "name": info.get("longName") or info.get("shortName") or ticker.upper(),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "exchange": info.get("fullExchangeName") or info.get("exchange", ""),
        "price": (f"{float(px):,.2f} {cur}".strip() if px is not None else "—"),
        "market_cap": _money(info.get("marketCap")),
        "low52": (f"{float(info['fiftyTwoWeekLow']):,.2f}" if info.get("fiftyTwoWeekLow") else "—"),
        "high52": (f"{float(info['fiftyTwoWeekHigh']):,.2f}" if info.get("fiftyTwoWeekHigh") else "—"),
        "currency": cur,
        "price_asof": asof,
        "asof_is_market": asof_is_market,
        "range_pos": _range52(info),   # raw fraction [0,1] for the 52-week range bar
    }


def build_ratios(ticker, info):
    """Return {category: [row, ...]} where row = {label, value, gauge, explain, available}.
    `explain` is the neutral template already filled with the ticker and formatted value."""
    t = ticker.upper()
    out = {c: [] for c in CATEGORIES}
    for r in RATIOS:
        if r["key"] == "_range52":
            raw = _range52(info)
        else:
            raw = info.get(r["key"])
        value = _fmt(r["fmt"], raw)
        available = value != "—"
        out[r["cat"]].append({
            "label": r["label"],
            "value": value,
            "gauge": r["gauge"],
            "available": available,
            "explain": r["explain"].format(t=t, v=value),
        })
    return out
