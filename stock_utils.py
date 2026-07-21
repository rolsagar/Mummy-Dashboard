"""
stock_utils.py
==============
Shared data-fetching and indicator utilities used by both `app.py` (the main
dashboard) and any pages under `pages/` (e.g. a per-stock chart view).

Keeping this logic in one module guarantees the dashboard and the chart page
always compute RSI/MACD the exact same way.
"""

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
RSI_PERIOD = 14


# --------------------------------------------------------------------------
# Data fetching
# --------------------------------------------------------------------------
@st.cache_data(ttl=300)  # cache for 5 minutes
def fetch_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a ticker from Yahoo Finance.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume,
    indexed by date. Returns an empty DataFrame on failure.
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance sometimes returns MultiIndex columns (esp. for single-ticker
    # downloads on newer versions) — flatten them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Close"])
    return df


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------
def resample_ohlc(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample a daily OHLCV DataFrame to a coarser timeframe.
    rule examples: "W-FRI" (weekly, week ending Friday), "ME" (month end).
    """
    if daily.empty:
        return daily

    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }
    if "Volume" in daily.columns:
        agg["Volume"] = "sum"

    resampled = daily.resample(rule).agg(agg).dropna(subset=["Close"])
    return resampled


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def rsi_wilder(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Classic Wilder's RSI using an exponentially-weighted moving average
    of gains/losses (alpha = 1/period), which matches how RSI is computed
    on most charting platforms (TradingView, etc.).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def macd_lines(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Standard MACD(12, 26, 9). Returns (macd_line, signal_line) as Series.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line
