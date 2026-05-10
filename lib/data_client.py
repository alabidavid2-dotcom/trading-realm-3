# ================================================
# DATA_CLIENT.PY - Alpaca Market Data Client
# Singleton wrapper for StockHistoricalDataClient
# Free tier = IEX feed (15-min delayed intraday, real-time daily)
# ================================================

from __future__ import annotations   # Python 3.9 compat for X | Y type hints
from datetime import datetime, timedelta
import pandas as pd

# ── Singleton state ──────────────────────────────────────────────────────────────
_client = None

# ── Auth debug info (populated on first client init, read by app.py sidebar) ────
_auth_debug: dict = {}


# ── Internal helpers ─────────────────────────────────────────────────────────────

def _get_client():
    """
    Return the shared StockHistoricalDataClient, initialising it on first call.

    Key decisions made here:
    - Keys are fetched via config.get_alpaca_keys() which strips whitespace.
    - The DATA base URL is always https://data.alpaca.markets — it is the
      same for both paper and live accounts.  Paper vs live only matters for
      the TradingClient (broker API), not for market-data requests.
    - url_override is set explicitly so the correct endpoint is crystal-clear
      and immune to any future alpaca-py default changes.
    """
    global _client, _auth_debug

    if _client is not None:
        return _client

    from config import get_alpaca_keys, ALPACA_DATA_URL, ALPACA_BASE_URL

    key, sec = get_alpaca_keys()   # always stripped

    # ── Auth Validation ──────────────────────────────────────────────────────
    _problems = []
    if not key or not sec:
        _problems.append("one or both keys are empty — check .env file")
    if 'YOUR_KEY_HERE' in key or 'YOUR_KEY_HERE' in sec:
        _problems.append("placeholder 'YOUR_KEY_HERE' found — replace with real keys")
    if key and not key.startswith(('PK', 'AK')):
        _problems.append(
            f"API key prefix '{key[:4]}…' unrecognised — paper keys start 'PK', live keys 'AK'"
        )
    if len(key) < 20:
        _problems.append(f"API key too short ({len(key)} chars, expected ≥ 20)")
    if len(sec) < 40:
        _problems.append(f"Secret key too short ({len(sec)} chars, expected ≥ 40)")

    if _problems:
        msg = "Cybersecurity Check: Keys not found in .env file. " + " | ".join(_problems)
        try:
            import streamlit as st
            st.error(msg)
        except Exception:
            import sys; print(f"[AUTH ERROR] {msg}", file=sys.stderr)
        raise ValueError(msg)

    # ── Populate debug info (masked — last 4 chars only) ────────────────────
    env_label = 'PAPER' if key.startswith('PK') else 'LIVE'
    _auth_debug = {
        'key_last4':      key[-4:],
        'sec_last4':      sec[-4:],
        'key_len':        len(key),
        'sec_len':        len(sec),
        'environment':    env_label,
        'trading_url':    ALPACA_BASE_URL,      # broker endpoint (paper vs live)
        'data_url':       ALPACA_DATA_URL,       # data endpoint (always the same)
    }

    # ── Build client ─────────────────────────────────────────────────────────
    from alpaca.data.historical import StockHistoricalDataClient
    _client = StockHistoricalDataClient(
        api_key=key,
        secret_key=sec,
        url_override=ALPACA_DATA_URL,   # explicit — always data.alpaca.markets
    )
    return _client


def get_auth_debug_info() -> dict:
    """
    Return masked auth debug info for the sidebar.
    Triggers client init if not already done.
    Safe to call at any time — never raises.
    """
    global _auth_debug
    try:
        _get_client()
    except Exception:
        pass
    return dict(_auth_debug)


def _fetch(ticker: str, timeframe, start: datetime, end: datetime = None) -> pd.DataFrame:
    """Core fetch — returns clean OHLCV DataFrame; empty DataFrame on any error.
    Retries up to 3 times on 429 rate-limit responses with exponential backoff."""
    import time as _time
    from alpaca.data.requests import StockBarsRequest
    from config import DATA_FEED

    if end is None:
        end = datetime.now() + timedelta(days=1)

    for _attempt in range(3):
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

            # Normalise column names to Title Case
            df = df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume',
            })

            keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            return df[keep]

        except Exception as _e:
            _fetch._last_error = str(_e)
            # Retry on rate-limit (429); surface all other errors immediately
            if '429' in str(_e) or 'rate limit' in str(_e).lower() or 'too many' in str(_e).lower():
                _time.sleep(2 ** _attempt)   # 1s → 2s → 4s backoff
                continue
            return pd.DataFrame()

    return pd.DataFrame()


# ── Public helpers ───────────────────────────────────────────────────────────────

def get_daily(ticker: str, days: int = 730) -> pd.DataFrame:
    from alpaca.data.timeframe import TimeFrame
    lookback = max(days, 14)   # floor at 14 days — weekend always captures Friday
    end = datetime.now() + timedelta(days=1)
    return _fetch(ticker, TimeFrame.Day, datetime.now() - timedelta(days=lookback), end)


def get_weekly(ticker: str, days: int = 365) -> pd.DataFrame:
    from alpaca.data.timeframe import TimeFrame
    return _fetch(ticker, TimeFrame.Week, datetime.now() - timedelta(days=days))


def get_monthly(ticker: str, months: int = 24) -> pd.DataFrame:
    from alpaca.data.timeframe import TimeFrame
    return _fetch(ticker, TimeFrame.Month, datetime.now() - timedelta(days=months * 31))


def get_4h(ticker: str, days: int = 60) -> pd.DataFrame:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    return _fetch(ticker, TimeFrame(4, TimeFrameUnit.Hour), datetime.now() - timedelta(days=days))


def get_60min(ticker: str, days: int = 14) -> pd.DataFrame:
    """Standard 1-hour bars — The Strat's 60m timeframe."""
    from alpaca.data.timeframe import TimeFrame
    return _fetch(ticker, TimeFrame.Hour, datetime.now() - timedelta(days=days))


def get_5min(ticker: str, days: int = 3) -> pd.DataFrame:
    """5-minute bars — The Strat's intraday precision timeframe."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    return _fetch(ticker, TimeFrame(5, TimeFrameUnit.Minute), datetime.now() - timedelta(days=days))


# Keep legacy names so any existing callers outside FTFC don't break
def get_65min(ticker: str, days: int = 30) -> pd.DataFrame:
    return get_60min(ticker, days=days)


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


def get_ftfc_snapshot(ticker: str, mode: str = 'intraday') -> list[dict]:
    """
    Fetch the most recent completed bar for each FTFC timeframe via Alpaca.

    mode='intraday' → 7 TFs: Monthly, Weekly, Daily, 4H, 65min, 45min, 15min
    mode='swing'    → 4 TFs: Monthly, Weekly, Daily, 4H

    Each item: {tf, open, close, direction, change_pct}
    direction is 'up', 'down', or 'neutral' (no data).  Never raises.
    """
    # Canonical Strat timeframe stack: M → W → D → 4H → 60m → 15m → 5m
    # Lookbacks are generous so weekends/holidays never produce empty results.
    intraday_specs = [
        ('Monthly', lambda: get_monthly(ticker, months=6)),
        ('Weekly',  lambda: get_weekly(ticker,  days=120)),
        ('Daily',   lambda: get_daily(ticker,   days=30)),
        ('4H',      lambda: get_4h(ticker,      days=30)),
        ('60min',   lambda: get_60min(ticker,   days=14)),
        ('15min',   lambda: get_15min(ticker,   days=5)),
        ('5min',    lambda: get_5min(ticker,    days=3)),
    ]
    swing_specs = intraday_specs[:4]   # swing uses M/W/D/4H only
    specs = intraday_specs if mode == 'intraday' else swing_specs

    import time as _time
    result = []
    errors = []
    for name, fetch_fn in specs:
        _time.sleep(0.05)   # 50ms stagger between timeframe calls — keeps burst ≤ 20 req/sec per worker
        try:
            df = fetch_fn()
            if df.empty or 'Open' not in df.columns or 'Close' not in df.columns:
                raise ValueError(f"{name}: fetch returned empty DataFrame")
            last = df.iloc[-1]
            o   = round(float(last['Open']),  2)
            c   = round(float(last['Close']), 2)
            chg = round((c - o) / o * 100, 2) if o else None
            result.append({
                'tf': name, 'open': o, 'close': c,
                'direction': 'up' if c > o else 'down',
                'change_pct': chg,
            })
        except Exception as _e:
            errors.append(f"{name}: {_e}")
            result.append({'tf': name, 'direction': 'neutral',
                           'open': None, 'close': None, 'change_pct': None})

    get_ftfc_snapshot._last_errors = errors
    return result


# ── Pattern / Gap helpers ────────────────────────────────────────────────────────

def detect_daily_gap(df: pd.DataFrame) -> dict:
    """Compare yesterday's close to today's open.  Delegates to lib.indicators."""
    try:
        from lib.indicators import detect_gap
        return detect_gap(df)
    except Exception:
        return {'gap_type': 'none', 'gap_pct': 0.0}


def detect_candle_patterns(df: pd.DataFrame) -> list:
    """Identify classic candle patterns on the last 3 bars.  Delegates to lib.indicators."""
    try:
        from lib.indicators import detect_candlestick_patterns
        return detect_candlestick_patterns(df)
    except Exception:
        return []


# ── Live quote (snapshot API) ────────────────────────────────────────────────────

def get_live_quote(ticker: str) -> dict | None:
    """
    Fetch latest price and daily change via Alpaca snapshot API.
    Uses the SAME singleton client as all other data calls — same credentials,
    same data endpoint (https://data.alpaca.markets).

    Returns None on invalid ticker or network failure — never raises.
    Keys: ticker, price, change_pct, change_dollar, prev_close, last_trade_time
    """
    try:
        from alpaca.data.requests import StockSnapshotRequest
        from config import DATA_FEED
        client = _get_client()   # same client, same auth — guaranteed
        snaps = client.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=ticker.upper(), feed=DATA_FEED)
        )
        snap = (snaps or {}).get(ticker.upper())
        if snap is None:
            return None

        # Price chain: latest_trade → quote mid → daily bar close
        price = None
        last_trade_time = None
        if snap.latest_trade and snap.latest_trade.price:
            price           = float(snap.latest_trade.price)
            last_trade_time = snap.latest_trade.timestamp
        if price is None and snap.latest_quote:
            ask   = float(snap.latest_quote.ask_price or 0)
            bid   = float(snap.latest_quote.bid_price or 0)
            price = (ask + bid) / 2 if ask and bid else (ask or bid or None)
        if price is None and snap.daily_bar and snap.daily_bar.close:
            price = float(snap.daily_bar.close)
        if price is None:
            return None

        prev_close    = (
            float(snap.previous_daily_bar.close)
            if snap.previous_daily_bar and snap.previous_daily_bar.close
            else None
        )
        change_dollar = round(price - prev_close, 2)          if prev_close else None
        change_pct    = round((price - prev_close) / prev_close * 100, 2) if prev_close else None

        return {
            "ticker":          ticker.upper(),
            "price":           round(price, 2),
            "change_pct":      change_pct,
            "change_dollar":   change_dollar,
            "prev_close":      round(prev_close, 2) if prev_close else None,
            "last_trade_time": last_trade_time,
        }
    except Exception as _e:
        get_live_quote._last_error = str(_e)
        return None


def get_batch_snapshots(symbols: list) -> dict:
    """
    Fetch live snapshots for multiple symbols in one API call.
    Returns {TICKER: {price, prev_close, change_pct, direction}} — empty dict on error.
    One call for the whole universe is ~10x faster than per-ticker get_live_quote().
    """
    try:
        from alpaca.data.requests import StockSnapshotRequest
        from config import DATA_FEED
        client = _get_client()
        upper = [s.upper() for s in symbols if s]
        if not upper:
            return {}
        snaps = client.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=upper, feed=DATA_FEED)
        )
        if not snaps:
            return {}
        result = {}
        for ticker, snap in snaps.items():
            try:
                price = None
                if snap.latest_trade and snap.latest_trade.price:
                    price = float(snap.latest_trade.price)
                if price is None and snap.latest_quote:
                    ask = float(snap.latest_quote.ask_price or 0)
                    bid = float(snap.latest_quote.bid_price or 0)
                    price = (ask + bid) / 2 if ask and bid else (ask or bid or None)
                if price is None and snap.daily_bar and snap.daily_bar.close:
                    price = float(snap.daily_bar.close)
                if price is None:
                    continue
                prev_bar   = snap.previous_daily_bar
                prev_close = float(prev_bar.close) if prev_bar and prev_bar.close else None
                prev_high  = float(prev_bar.high)  if prev_bar and prev_bar.high  else None
                prev_low   = float(prev_bar.low)   if prev_bar and prev_bar.low   else None
                change_pct = round((price - prev_close) / prev_close * 100, 2) if prev_close else None
                direction  = 'up' if (prev_close and price > prev_close) else ('down' if prev_close else 'neutral')
                result[ticker] = {
                    'price':      round(price, 2),
                    'prev_close': round(prev_close, 2) if prev_close else None,
                    'prev_high':  round(prev_high, 2)  if prev_high  else None,
                    'prev_low':   round(prev_low, 2)   if prev_low   else None,
                    'change_pct': change_pct,
                    'direction':  direction,
                }
            except Exception:
                continue
        return result
    except Exception as _e:
        get_batch_snapshots._last_error = str(_e)
        return {}
