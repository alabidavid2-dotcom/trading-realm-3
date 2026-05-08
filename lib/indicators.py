# ================================================
# INDICATORS.PY - Technical Indicator Library
# All indicators computed here, gated by regime in signal_engine.py
# ================================================

import numpy as np
import pandas as pd
from config import (
    RSI_PERIOD, ADX_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    MOMENTUM_PERIOD, BB_PERIOD, BB_STD, VOLUME_MA_PERIOD
)


def add_all_indicators(df):
    """
    Compute all technical indicators and attach as columns.
    Input df must have OHLCV columns.
    """
    df = df.copy()
    df = add_rsi(df)
    df = add_adx(df)
    df = add_macd(df)
    df = add_momentum(df)
    df = add_bollinger_bands(df)
    df = add_volume_analysis(df)
    df = add_atr(df)
    return df


# --------------------------------------------------
# RSI
# --------------------------------------------------
def add_rsi(df, period=RSI_PERIOD):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    avg_loss_safe = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss_safe
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50).clip(0, 100)  # 50 when avg_loss=0 (all gains), clamp to valid range
    return df


# --------------------------------------------------
# ADX (Average Directional Index)
# --------------------------------------------------
def add_adx(df, period=ADX_PERIOD):
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan))

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df['adx'] = dx.ewm(span=period, adjust=False).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    df['adx_rising'] = df['adx'] > df['adx'].shift(1)
    return df


# --------------------------------------------------
# MACD
# --------------------------------------------------
def add_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    return df


# --------------------------------------------------
# Momentum
# --------------------------------------------------
def add_momentum(df, period=MOMENTUM_PERIOD):
    df['momentum'] = df['Close'].pct_change(period) * 100
    df['momentum_positive'] = df['momentum'] > 0
    return df


# --------------------------------------------------
# Bollinger Bands
# --------------------------------------------------
def add_bollinger_bands(df, period=BB_PERIOD, std=BB_STD):
    df['bb_mid'] = df['Close'].rolling(period).mean()
    rolling_std = df['Close'].rolling(period).std()
    df['bb_upper'] = df['bb_mid'] + (rolling_std * std)
    df['bb_lower'] = df['bb_mid'] - (rolling_std * std)
    df['bb_pct'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    df['bb_squeeze'] = rolling_std < rolling_std.rolling(50).mean()  # Tightening bands
    return df


# --------------------------------------------------
# Volume Analysis
# --------------------------------------------------
def add_volume_analysis(df, period=VOLUME_MA_PERIOD):
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['vol_ma'] = df['Volume'].rolling(period).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_ma'].replace(0, np.nan)
        df['high_volume'] = df['vol_ratio'] > 1.5
    else:
        df['vol_ma'] = 0
        df['vol_ratio'] = 1.0
        df['high_volume'] = False
    return df


# --------------------------------------------------
# ATR (for stop loss / position sizing)
# --------------------------------------------------
def add_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(period).mean()
    df['atr_pct'] = df['atr'] / df['Close'] * 100
    return df


# --------------------------------------------------
# Gap Detection
# --------------------------------------------------
def detect_gap(df) -> dict:
    """
    Detect opening gap vs prior close.
    Returns: gap_type ('strong_up','moderate_up','none','moderate_down','strong_down') and gap_pct.
    Thresholds: ignore <0.5%, moderate 0.5–1.5%, strong >1.5%.
    """
    if len(df) < 2:
        return {'gap_type': 'none', 'gap_pct': 0.0}

    prev_close = float(df['Close'].iloc[-2])
    curr_open  = float(df['Open'].iloc[-1])

    if prev_close == 0:
        return {'gap_type': 'none', 'gap_pct': 0.0}

    gap_pct = (curr_open - prev_close) / prev_close * 100

    if gap_pct > 1.5:
        gap_type = 'strong_up'
    elif gap_pct > 0.5:
        gap_type = 'moderate_up'
    elif gap_pct < -1.5:
        gap_type = 'strong_down'
    elif gap_pct < -0.5:
        gap_type = 'moderate_down'
    else:
        gap_type = 'none'

    return {'gap_type': gap_type, 'gap_pct': round(gap_pct, 2)}


# --------------------------------------------------
# Classic Candlestick Patterns
# --------------------------------------------------
def detect_candlestick_patterns(df) -> list:
    """
    Detect classic candlestick patterns on the last 3 bars.
    Returns list of pattern dicts with name, direction, grade, description.
    """
    patterns = []
    if len(df) < 3:
        return patterns

    c1 = df.iloc[-3]
    c2 = df.iloc[-2]
    c3 = df.iloc[-1]

    o1, h1, l1, cl1 = float(c1['Open']), float(c1['High']), float(c1['Low']), float(c1['Close'])
    o2, h2, l2, cl2 = float(c2['Open']), float(c2['High']), float(c2['Low']), float(c2['Close'])
    o3, h3, l3, cl3 = float(c3['Open']), float(c3['High']), float(c3['Low']), float(c3['Close'])

    body2   = abs(cl2 - o2)
    body3   = abs(cl3 - o3)
    range2  = (h2 - l2) or 0.001
    range3  = (h3 - l3) or 0.001

    # Bullish Engulfing: prior red, current green wraps it
    if cl2 < o2 and cl3 > o3 and cl3 >= o2 and o3 <= cl2:
        patterns.append({
            'name': 'Bullish Engulfing', 'direction': 'up', 'grade': 'A',
            'description': 'Green candle fully engulfs prior red — bullish reversal.',
        })

    # Bearish Engulfing
    if cl2 > o2 and cl3 < o3 and cl3 <= o2 and o3 >= cl2:
        patterns.append({
            'name': 'Bearish Engulfing', 'direction': 'down', 'grade': 'A',
            'description': 'Red candle fully engulfs prior green — bearish reversal.',
        })

    # Hammer (long lower wick, small body near top)
    lower_wick = (min(o2, cl2) - l2)
    upper_wick = (h2 - max(o2, cl2))
    if body2 > 0 and lower_wick >= 2 * body2 and upper_wick <= 0.15 * range2:
        patterns.append({
            'name': 'Hammer', 'direction': 'up', 'grade': 'B+',
            'description': 'Long lower wick — buyers absorbed selling pressure.',
        })

    # Shooting Star (long upper wick, small body near bottom)
    upper_wick = (h2 - max(o2, cl2))
    lower_wick = (min(o2, cl2) - l2)
    if body2 > 0 and upper_wick >= 2 * body2 and lower_wick <= 0.15 * range2:
        patterns.append({
            'name': 'Shooting Star', 'direction': 'down', 'grade': 'B+',
            'description': 'Long upper wick — sellers rejected price at highs.',
        })

    # Morning Doji Star (3-candle: bear → doji → bull)
    doji2 = body2 / range2 < 0.15
    if cl1 < o1 and doji2 and cl3 > o3 and cl3 > (o1 + cl1) / 2:
        patterns.append({
            'name': 'Morning Doji Star', 'direction': 'up', 'grade': 'A+',
            'description': '3-candle bullish reversal: bear → doji indecision → bull breakout.',
        })

    # Dark Cloud Cover (3-candle: bull → opens above, closes below midpoint)
    if cl1 > o1 and cl2 < o2 and o2 > cl1 and cl2 < (o1 + cl1) / 2:
        patterns.append({
            'name': 'Dark Cloud Cover', 'direction': 'down', 'grade': 'A',
            'description': 'Opens above prior high, closes below midpoint — bearish reversal.',
        })

    return patterns


# --------------------------------------------------
# Summary snapshot for latest bar
# --------------------------------------------------
def indicator_snapshot(df):
    """Return a dict summarizing the latest indicator values."""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    return {
        'rsi': round(float(latest.get('rsi', 50)), 1),
        'adx': round(float(latest.get('adx', 0)), 1),
        'adx_rising': bool(latest.get('adx_rising', False)),
        'plus_di': round(float(latest.get('plus_di', 0)), 1),
        'minus_di': round(float(latest.get('minus_di', 0)), 1),
        'macd_hist': round(float(latest.get('macd_hist', 0)), 4),
        'macd_cross_up': bool(latest.get('macd_cross_up', False)),
        'macd_cross_down': bool(latest.get('macd_cross_down', False)),
        'momentum': round(float(latest.get('momentum', 0)), 2),
        'bb_pct': round(float(latest.get('bb_pct', 0.5)), 2),
        'bb_squeeze': bool(latest.get('bb_squeeze', False)),
        'vol_ratio': round(float(latest.get('vol_ratio', 1.0)), 2),
        'high_volume': bool(latest.get('high_volume', False)),
        'atr': round(float(latest.get('atr', 0)), 2),
        'atr_pct': round(float(latest.get('atr_pct', 0)), 2),
    }
