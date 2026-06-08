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
