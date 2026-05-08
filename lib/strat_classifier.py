# ================================================
# STRAT_CLASSIFIER.PY - The Strat Candle System
# Inside (1) / Directional (2) / Outside (3) classification
# + pattern recognition for actionable setups
# ================================================

import pandas as pd
import numpy as np
from config import STRAT_LOOKBACK


def classify_candle(prev, curr):
    """
    Classify a candle relative to its predecessor using The Strat system.
      1 = Inside Bar (range contained within previous bar)
      2 = Directional (2-Up breaks high only, 2-Down breaks low only)
      3 = Outside Bar (breaks both high and low of previous bar)
    Also returns direction for type 2 bars.
    """
    inside = curr['Low'] >= prev['Low'] and curr['High'] <= prev['High']
    breaks_high = curr['High'] > prev['High']
    breaks_low = curr['Low'] < prev['Low']

    if inside:
        return 1, 'neutral'
    elif breaks_high and breaks_low:
        # Outside bar - direction based on close
        direction = 'up' if curr['Close'] > curr['Open'] else 'down'
        return 3, direction
    elif breaks_high:
        return 2, 'up'
    elif breaks_low:
        return 2, 'down'
    else:
        return 1, 'neutral'  # Edge case: exact same range


def add_strat_columns(df):
    """
    Add Strat classification columns to the entire dataframe.
    """
    df = df.copy()
    df['strat_type'] = 0
    df['strat_dir'] = 'neutral'

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        stype, sdir = classify_candle(prev, curr)
        df.iloc[i, df.columns.get_loc('strat_type')] = stype
        df.iloc[i, df.columns.get_loc('strat_dir')] = sdir

    return df


def detect_strat_patterns(df, lookback=STRAT_LOOKBACK):
    """
    Scan the last N candles for actionable Strat patterns.
    Returns a list of pattern dicts with name, direction, and grade.

    Key patterns:
      2-1-2 Reversal: Directional → Inside → Directional (opposite). HIGH PROB.
      2-1-2 Continuation: Directional → Inside → Directional (same). SOLID.
      3-1 Reversal: Outside → Inside → breakout opposite the 3. GOOD.
      3-2 Continuation: Outside followed by Directional in same direction.
      1-1 (Hammer Time): Multiple inside bars = coiling energy. WATCH.
      2-2 Reversal: Two directional bars reversing direction. FAST MOVE.
    """
    if len(df) < lookback:
        return []

    recent = df.tail(lookback)
    types = recent['strat_type'].tolist()
    dirs = recent['strat_dir'].tolist()
    patterns = []

    # Scan windows of 2-3 candles at the end
    if len(types) >= 3:
        t3, t2, t1 = types[-3], types[-2], types[-1]
        d3, d2, d1 = dirs[-3], dirs[-2], dirs[-1]

        # --- 2-1-2 Reversal (Grade A+) ---
        if t3 == 2 and t2 == 1 and t1 == 2 and d3 != d1 and d1 != 'neutral':
            patterns.append({
                'name': '2-1-2 Reversal',
                'direction': d1,
                'grade': 'A+',
                'description': f'Reversal {d3}→inside→{d1}. High probability setup.',
            })

        # --- 2-1-2 Continuation (Grade A) ---
        if t3 == 2 and t2 == 1 and t1 == 2 and d3 == d1 and d1 != 'neutral':
            patterns.append({
                'name': '2-1-2 Continuation',
                'direction': d1,
                'grade': 'A',
                'description': f'Continuation {d1}. Inside bar breakout in trend direction.',
            })

        # --- 3-2 Continuation (Grade A) ---
        if t2 == 3 and t1 == 2 and d1 != 'neutral':
            patterns.append({
                'name': '3-2 Continuation',
                'direction': d1,
                'grade': 'A',
                'description': f'Outside bar followed by directional {d1}.',
            })

        # --- 3-1 Setup (Grade B+ watch) ---
        if t2 == 3 and t1 == 1:
            patterns.append({
                'name': '3-1 Compression',
                'direction': 'pending',
                'grade': 'B+',
                'description': 'Outside bar → Inside. Breakout imminent. Wait for direction.',
            })

    if len(types) >= 2:
        t2, t1 = types[-2], types[-1]
        d2, d1 = dirs[-2], dirs[-1]

        # --- 2-2 Reversal (Grade A) ---
        if t2 == 2 and t1 == 2 and d2 != d1 and d1 != 'neutral':
            patterns.append({
                'name': '2-2 Reversal',
                'direction': d1,
                'grade': 'A',
                'description': f'Back-to-back directional reversal. Fast move {d1}.',
            })

    # --- Inside bar compression (multiple 1s) ---
    inside_count = sum(1 for t in types[-3:] if t == 1)
    if inside_count >= 2:
        patterns.append({
            'name': 'Inside Bar Compression',
            'direction': 'pending',
            'grade': 'B',
            'description': f'{inside_count} inside bars in last 3. Energy coiling for breakout.',
        })

    # --- Classic candlestick patterns (with +10 confirmation bonus if aligned) ---
    try:
        from lib.indicators import detect_candlestick_patterns
        classic = detect_candlestick_patterns(df)
        strat_dirs = {p['direction'] for p in patterns if p.get('direction') not in (None, 'pending', 'neutral')}
        for cp in classic:
            cp = cp.copy()
            if cp['direction'] in strat_dirs:
                cp['description'] += ' (+10 confirmed by Strat alignment)'
            patterns.append(cp)
    except Exception:
        pass

    return patterns


def ftfc_check(daily_dir, weekly_dir=None, monthly_dir=None):
    """
    Full Timeframe Continuity (FTFC) check.
    All timeframes must agree on direction for highest probability trades.
    Returns: 'bullish', 'bearish', or 'mixed'
    """
    directions = [d for d in [daily_dir, weekly_dir, monthly_dir] if d is not None]

    if not directions:
        return 'mixed'

    if all(d == 'up' for d in directions):
        return 'bullish'
    elif all(d == 'down' for d in directions):
        return 'bearish'
    else:
        return 'mixed'


# Alias so both ftfc_check and ftc_check resolve to the same function
ftc_check = ftfc_check


def strat_analysis(df, timeframe_name="Daily"):
    """
    Full Strat analysis on a dataframe. Returns a summary dict.
    """
    df = add_strat_columns(df)
    patterns = detect_strat_patterns(df)
    latest = df.iloc[-1]

    return {
        'timeframe': timeframe_name,
        'last_type': int(latest['strat_type']),
        'last_dir': latest['strat_dir'],
        'patterns': patterns,
        'close': round(float(latest['Close']), 2),
        'recent_types': df['strat_type'].tail(STRAT_LOOKBACK).tolist(),
        'recent_dirs': df['strat_dir'].tail(STRAT_LOOKBACK).tolist(),
    }
