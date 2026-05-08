# ================================================
# DATA_CLIENT.PY - Alpaca Market Data Client
# Singleton wrapper for StockHistoricalDataClient
# Free tier = IEX feed (15-min delayed intraday, real-time daily)
# ================================================

from datetime import datetime, timedelta
import pandas as pd

_client = None


def _get_client():
    global _client
    if _client is None:
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
        from alpaca.data.historical import StockHistoricalDataClient
        _client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    return _client


def _fetch(ticker: str, timeframe, start: datetime, end: datetime = None) -> pd.DataFrame:
    """Core fetch — returns clean OHLCV DataFrame or empty DataFrame on any error."""
    from alpaca.data.requests import StockBarsRequest
    from config import DATA_FEED

    if end is None:
        end = datetime.now()

    try:
        client = _get_client()
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=DATA_FEED,
        )
        bars = client.get_stock_bars(req)
        if not bars or not bars.data:
            return pd.DataFrame()

        df = bars.df
        if df is None or df.empty:
            return pd.DataFrame()

        # Drop symbol level from MultiIndex (symbol, timestamp) → (timestamp,)
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)

        # Strip timezone info
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df.index = pd.to_datetime(df.index)

        # Normalize column names to Title Case
        rename = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume',
        }
        df = df.rename(columns=rename)

        keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        return df[keep]

    except Exception:
        return pd.DataFrame()


# ── Public helpers ──────────────────────────────────────────────────────────────

def get_daily(ticker: str, days: int = 730) -> pd.DataFrame:
    from alpaca.data.timeframe import TimeFrame
    return _fetch(ticker, TimeFrame.Day, datetime.now() - timedelta(days=days))


def get_weekly(ticker: str, days: int = 365) -> pd.DataFrame:
    from alpaca.data.timeframe import TimeFrame
    return _fetch(ticker, TimeFrame.Week, datetime.now() - timedelta(days=days))


def get_monthly(ticker: str, months: int = 24) -> pd.DataFrame:
    from alpaca.data.timeframe import TimeFrame
    return _fetch(ticker, TimeFrame.Month, datetime.now() - timedelta(days=months * 31))


def get_4h(ticker: str, days: int = 60) -> pd.DataFrame:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    return _fetch(ticker, TimeFrame(4, TimeFrameUnit.Hour), datetime.now() - timedelta(days=days))


def get_65min(ticker: str, days: int = 30) -> pd.DataFrame:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    # 65-min is non-standard; fall back to hourly if rejected
    try:
        return _fetch(ticker, TimeFrame(65, TimeFrameUnit.Minute), datetime.now() - timedelta(days=days))
    except Exception:
        return _fetch(ticker, TimeFrame.Hour, datetime.now() - timedelta(days=days))


def get_45min(ticker: str, days: int = 30) -> pd.DataFrame:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    try:
        return _fetch(ticker, TimeFrame(45, TimeFrameUnit.Minute), datetime.now() - timedelta(days=days))
    except Exception:
        return _fetch(ticker, TimeFrame(30, TimeFrameUnit.Minute), datetime.now() - timedelta(days=days))


def get_15min(ticker: str, days: int = 5) -> pd.DataFrame:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    return _fetch(ticker, TimeFrame(15, TimeFrameUnit.Minute), datetime.now() - timedelta(days=days))


def get_yearly(ticker: str) -> pd.DataFrame:
    return get_daily(ticker, days=365)


def get_3month(ticker: str) -> pd.DataFrame:
    return get_daily(ticker, days=90)
