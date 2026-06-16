# © 2026 Sami Jeddou. All rights reserved.
# Published publicly for demonstration and evaluation only — no license is granted.
# Copying, modification, redistribution, or reuse (in whole or in part) without the
# author's prior written permission is prohibited.
"""Data layer — fetch prices, clean returns, derive an AssetUniverse.

The moved functions are unchanged. On top of them sits a swappable DataSource: any
feed (yfinance, a CSV, a vendor API) becomes one more class implementing .prices(),
and universe_from_prices() lifts a price frame into the typed boundary. No Streamlit.
"""
from __future__ import annotations
from typing import Protocol
import numpy as np
import pandas as pd
from core.types import AssetUniverse

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
                               group_by='column', threads=False)

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

def fetch_close_prices(tickers, start, end):
    """Return (clean Close-price DataFrame in requested ticker order, error_or_None)."""
    try:
        import yfinance as yf
        raw_full = yf.download(tickers, start=str(start), end=str(end),
                               auto_adjust=True, progress=False, group_by='column', threads=False)
        if raw_full is None or raw_full.empty:
            return None, "No data returned — check tickers and date range."
        if isinstance(raw_full.columns, pd.MultiIndex):
            lvl0 = raw_full.columns.get_level_values(0)
            if 'Close' in lvl0:
                raw = raw_full['Close'].copy()
            elif 'Adj Close' in lvl0:
                raw = raw_full['Adj Close'].copy()
            else:
                raw = raw_full.iloc[:, :len(tickers)]
        else:
            if 'Close' in raw_full.columns:
                raw = raw_full[['Close']].copy(); raw.columns = [tickers[0]]
            elif 'Adj Close' in raw_full.columns:
                raw = raw_full[['Adj Close']].copy(); raw.columns = [tickers[0]]
            else:
                raw = raw_full.copy()
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(tickers[0])
        available = [t for t in tickers if t in raw.columns]
        if not available:
            return None, f"No Close price data found for tickers: {tickers}"
        raw = raw[available].dropna(how='all').dropna(axis=1, how='all')
        if raw.empty or len(raw) < 5:
            return None, "Insufficient data — try a wider date range."
        return raw, None
    except Exception as e:
        return None, str(e)

def fetch_ticker_info(symbol):
    """Return (info dict, error_or_None) for a single ticker via yfinance.
    `info` is the raw fundamentals/profile dict; used by the ticker-analytics view."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return None, "Enter a ticker symbol."
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
        # yfinance returns a near-empty dict for an unknown symbol
        if not info or (info.get("quoteType") is None and info.get("marketCap") is None
                        and info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
            return None, f"No data found for '{symbol}'. Check the symbol (e.g. AAPL, MSFT, MC.PA, BTC-USD)."
        return info, None
    except Exception as e:
        return None, str(e)

def fetch_ticker_history(symbol, period="1y"):
    """Return (Close-price DataFrame, error_or_None) for one ticker over a period
    ('6mo' / '1y' / '5y' / 'max'). Used by the ticker-analytics price chart."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return None, "Enter a ticker symbol."
    try:
        import yfinance as yf
        h = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if h is None or h.empty or "Close" not in h.columns:
            return None, "No price history available."
        _cols = [c for c in ["Open", "High", "Low", "Close"] if c in h.columns]
        return h[_cols].dropna(), None
    except Exception as e:
        return None, str(e)


def fetch_ticker_financials(symbol):
    """Return (rows, error) where rows is a list of {year, revenue, net_income}
    (ascending by year) from the annual income statement. Empty list when a ticker
    has no statements (ETFs, indices, crypto) — never raises for that."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return [], None
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        df = None
        for attr in ("income_stmt", "financials"):
            try:
                cand = getattr(t, attr)
            except Exception:
                cand = None
            if cand is not None and not cand.empty:
                df = cand
                break
        if df is None or df.empty:
            return [], None

        def _row(*names):
            for n in names:
                if n in df.index:
                    return df.loc[n]
            return None

        rev = _row("Total Revenue", "TotalRevenue", "Revenue", "Operating Revenue")
        ni = _row("Net Income", "NetIncome", "Net Income Common Stockholders",
                  "Net Income Continuous Operations")
        gp = _row("Gross Profit")
        oi = _row("Operating Income", "Operating Income Or Loss", "Total Operating Income As Reported")
        rows = []
        for col in df.columns:
            yr = str(getattr(col, "year", "") or str(col)[:4])
            r, n = _stmt_val(rev, col), _stmt_val(ni, col)
            if r is not None or n is not None:
                rows.append({"year": yr, "revenue": r, "net_income": n,
                             "gross_profit": _stmt_val(gp, col),
                             "operating_income": _stmt_val(oi, col)})
        rows.sort(key=lambda d: d["year"])
        return rows, None
    except Exception as e:
        return [], str(e)


def _stmt_val(series, col):
    """Safely read a financial-statement cell as float, or None (drops NaN/missing)."""
    if series is None:
        return None
    try:
        x = float(series[col])
        return x if x == x else None
    except Exception:
        return None


def _stmt_df(t, *attrs):
    """Return the first non-empty statement DataFrame among the given Ticker attrs."""
    for attr in attrs:
        try:
            cand = getattr(t, attr)
        except Exception:
            cand = None
        if cand is not None and not cand.empty:
            return cand
    return None


def fetch_ticker_cashflow(symbol):
    """Return (rows, error) — rows = [{year, ocf, fcf}] ascending (operating &
    free cash flow). Free CF falls back to OCF + CapEx when not reported directly.
    Empty list for instruments without a cash-flow statement."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return [], None
    try:
        import yfinance as yf
        df = _stmt_df(yf.Ticker(symbol), "cashflow", "cash_flow")
        if df is None:
            return [], None

        def _row(*names):
            for n in names:
                if n in df.index:
                    return df.loc[n]
            return None

        ocf = _row("Operating Cash Flow", "Total Cash From Operating Activities",
                   "Cash Flow From Continuing Operating Activities")
        fcf = _row("Free Cash Flow")
        capex = _row("Capital Expenditure", "Capital Expenditures")
        rows = []
        for col in df.columns:
            yr = str(getattr(col, "year", "") or str(col)[:4])
            o, f = _stmt_val(ocf, col), _stmt_val(fcf, col)
            if f is None and o is not None:
                c = _stmt_val(capex, col)
                if c is not None:
                    f = o + c                 # CapEx is reported negative
            if o is not None or f is not None:
                rows.append({"year": yr, "ocf": o, "fcf": f})
        rows.sort(key=lambda d: d["year"])
        return rows, None
    except Exception as e:
        return [], str(e)


def fetch_ticker_balance(symbol):
    """Return (dict, error) — latest balance-sheet snapshot
    {year, total_assets, total_debt, cash, equity}. Empty dict when unavailable."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return {}, None
    try:
        import yfinance as yf
        df = _stmt_df(yf.Ticker(symbol), "balance_sheet", "balancesheet")
        if df is None:
            return {}, None
        col = df.columns[0]                    # most recent period

        def _g(*names):
            for n in names:
                if n in df.index:
                    v = _stmt_val(df.loc[n], col)
                    if v is not None:
                        return v
            return None

        return {
            "year": str(getattr(col, "year", "") or str(col)[:4]),
            "total_assets": _g("Total Assets"),
            "total_debt": _g("Total Debt", "Total Debt And Capital Lease Obligation"),
            "cash": _g("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments",
                       "Cash And Short Term Investments"),
            "equity": _g("Stockholders Equity", "Total Equity Gross Minority Interest",
                         "Common Stock Equity"),
        }, None
    except Exception as e:
        return {}, str(e)


def stats_from_prices(prices, freq):
    """Annualised means, sigs, corr, names, factor from a Close-price DataFrame."""
    raw = prices.copy()
    if freq == "Monthly":
        raw = raw.resample('ME').last()
    rets = raw.pct_change().dropna()
    rets, _ = clean_returns(rets.copy())
    factor = 252 if freq == "Daily" else 12
    means = (rets.mean() * factor).tolist()
    sigs  = (rets.std()  * np.sqrt(factor)).tolist()
    corr  = rets.corr().values.tolist()
    names = list(rets.columns)
    return means, sigs, corr, names, factor


# ── Pluggable data sources ────────────────────────────────────────────────────
class DataSource(Protocol):
    """Anything that can return a Close-price DataFrame (tickers as columns)."""
    def prices(self, tickers: list[str], start, end) -> pd.DataFrame: ...


class YFinanceSource:
    """Default source: Yahoo Finance via the existing fetch_close_prices()."""
    def prices(self, tickers: list[str], start, end) -> pd.DataFrame:
        df, err = fetch_close_prices(tickers, start, end)
        if err:
            raise RuntimeError(err)
        return df


class CSVSource:
    """Read a Close-price CSV (date index, one column per ticker)."""
    def __init__(self, path_or_buffer):
        self.src = path_or_buffer
    def prices(self, tickers=None, start=None, end=None) -> pd.DataFrame:
        df = pd.read_csv(self.src, index_col=0, parse_dates=True)
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        return df[tickers] if tickers else df


def universe_from_prices(prices: pd.DataFrame, freq: str = "Daily") -> AssetUniverse:
    """Lift a Close-price frame into the typed AssetUniverse (annualised stats)."""
    means, sigs, corr, names, _factor = stats_from_prices(prices, freq)
    return AssetUniverse(names=list(names), means=np.array(means),
                         sigmas=np.array(sigs), corr=np.array(corr))


__all__ = [
    "corr_to_cov", "clean_returns", "parse_csv", "fetch_tickers",
    "fetch_close_prices", "stats_from_prices",
    "DataSource", "YFinanceSource", "CSVSource", "universe_from_prices",
]
