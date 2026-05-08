# ================================================
# SIGNAL_ENGINE.PY - Regime-Gated Signal Generation
# Combines HMM regime + indicators + The Strat into trade signals
# This is the brain of the system
# ================================================

import pandas as pd
import numpy as np
from config import (
    RSI_OVERBOUGHT, RSI_OVERSOLD, ADX_TREND_THRESHOLD,
    REGIME_RISK_MULTIPLIER
)


# --------------------------------------------------
# Signal Scores: Each component contributes to a composite score
# Range: -100 (max bearish) to +100 (max bullish)
# --------------------------------------------------

def score_regime(regime):
    """
    Score based on current HMM regime.
    This gates everything else — wrong regime = low score.
    """
    regime_scores = {
        "Bull_Quiet":     40,
        "Bull_Volatile":  25,
        "Chop":            0,
        "Bear_Quiet":    -40,
        "Bear_Volatile": -25,
    }
    return regime_scores.get(regime, 0)


def score_indicators(ind_snapshot, regime):
    """
    Score indicators CONDITIONALLY based on regime.
    Different regimes activate different indicator logic.
    """
    score = 0
    rsi = ind_snapshot['rsi']
    adx = ind_snapshot['adx']
    adx_rising = ind_snapshot['adx_rising']
    macd_hist = ind_snapshot['macd_hist']
    macd_cross_up = ind_snapshot['macd_cross_up']
    macd_cross_down = ind_snapshot['macd_cross_down']
    momentum = ind_snapshot['momentum']
    bb_pct = ind_snapshot.get('bb_pct', 0.5)
    bb_squeeze = ind_snapshot.get('bb_squeeze', False)
    vol_ratio = ind_snapshot.get('vol_ratio', 1.0)

    # === BULL REGIMES: Look for long entries ===
    if regime in ["Bull_Quiet", "Bull_Volatile"]:
        # RSI: Want above 50 (momentum confirmation), not overbought
        if 50 < rsi < RSI_OVERBOUGHT:
            score += 15
        elif rsi >= RSI_OVERBOUGHT:
            score -= 10  # Overextended, caution
        elif rsi < 40:
            score -= 5   # Weakening in a bull regime = red flag

        # ADX: Trending + rising = strong signal
        if adx > ADX_TREND_THRESHOLD and adx_rising:
            score += 15
        elif adx > ADX_TREND_THRESHOLD:
            score += 8

        # MACD: Cross up is a trigger
        if macd_cross_up:
            score += 12
        elif macd_hist > 0:
            score += 5
        elif macd_cross_down:
            score -= 15

        # Momentum positive
        if momentum > 0:
            score += 5

        # Volume confirmation
        if vol_ratio > 1.5:
            score += 5

    # === BEAR REGIMES: Look for short/put entries ===
    elif regime in ["Bear_Quiet", "Bear_Volatile"]:
        # RSI: Want below 50, not oversold
        if RSI_OVERSOLD < rsi < 50:
            score -= 15  # Good short signal (negative = bearish)
        elif rsi <= RSI_OVERSOLD:
            score += 10  # Oversold bounce risk
        elif rsi > 60:
            score += 5   # Strengthening in bear = might be turning

        # ADX trending + falling DI confirms bear
        if adx > ADX_TREND_THRESHOLD and adx_rising:
            score -= 15
        elif adx > ADX_TREND_THRESHOLD:
            score -= 8

        # MACD cross down = bear trigger
        if macd_cross_down:
            score -= 12
        elif macd_hist < 0:
            score -= 5
        elif macd_cross_up:
            score += 15

        # Momentum negative
        if momentum < 0:
            score -= 5

        # High volume on down moves
        if vol_ratio > 1.5:
            score -= 5

    # === CHOP: Minimal signals, mean reversion only ===
    elif regime == "Chop":
        # In chop, only trade extremes
        if rsi > RSI_OVERBOUGHT and bb_pct > 0.95:
            score -= 10  # Fade the top
        elif rsi < RSI_OVERSOLD and bb_pct < 0.05:
            score += 10  # Fade the bottom
        # Bollinger squeeze = breakout coming
        if bb_squeeze:
            score += 3  # Small bonus, wait for direction

        # ADX low = stay out
        if adx < 20:
            score *= 0.5  # Dampen all signals in no-trend

    return round(score)


def score_strat_patterns(patterns, regime):
    """
    Score The Strat patterns, boosted when aligned with regime direction.
    """
    score = 0
    for p in patterns:
        grade_weight = {'A+': 20, 'A': 15, 'B+': 8, 'B': 5}.get(p['grade'], 3)

        if p['direction'] == 'pending':
            score += grade_weight * 0.3  # Compression = potential, not action yet
            continue

        is_bullish_pattern = p['direction'] == 'up'
        is_bearish_pattern = p['direction'] == 'down'

        # Alignment bonus: pattern direction matches regime
        if is_bullish_pattern and regime in ["Bull_Quiet", "Bull_Volatile"]:
            score += grade_weight
        elif is_bearish_pattern and regime in ["Bear_Quiet", "Bear_Volatile"]:
            score -= grade_weight  # Negative = bearish signal
        elif is_bullish_pattern and regime in ["Bear_Quiet", "Bear_Volatile"]:
            score += grade_weight * 0.3  # Counter-trend, low weight
        elif is_bearish_pattern and regime in ["Bull_Quiet", "Bull_Volatile"]:
            score -= grade_weight * 0.3
        else:
            # Chop regime: half weight
            modifier = 1 if is_bullish_pattern else -1
            score += modifier * grade_weight * 0.5

    return round(score)


def generate_signal(regime, ind_snapshot, strat_patterns, regime_probs=None):
    """
    Master signal generator. Combines all components into a final signal.

    Returns dict:
      - composite_score: -100 to +100
      - direction: 'LONG', 'SHORT', 'FLAT'
      - strength: 'STRONG', 'MODERATE', 'WEAK', 'NO_TRADE'
      - trade_type: '0DTE_CALL', '0DTE_PUT', 'SWING_LONG', 'SWING_SHORT', 'NO_TRADE'
      - confidence: 0-100%
      - reasoning: list of contributing factors
    """
    reasoning = []

    # 1. Regime score
    r_score = score_regime(regime)
    reasoning.append(f"Regime ({regime}): {r_score:+d}")

    # 2. Indicator score (regime-gated)
    i_score = score_indicators(ind_snapshot, regime)
    reasoning.append(f"Indicators (regime-gated): {i_score:+d}")

    # 3. Strat pattern score
    s_score = score_strat_patterns(strat_patterns, regime)
    reasoning.append(f"Strat patterns: {s_score:+d}")

    # 4. Composite
    composite = r_score + i_score + s_score
    composite = max(-100, min(100, composite))  # Clamp

    # 5. Confidence from regime probabilities
    confidence = 50  # Default
    if regime_probs:
        top_prob = max(regime_probs.values())
        confidence = min(95, top_prob)  # Cap at 95
        if top_prob < 50:
            composite = int(composite * 0.7)  # Reduce signal when regime is uncertain
            reasoning.append(f"Regime confidence LOW ({top_prob}%) — signal dampened")

    # 6. Direction + Strength
    abs_score = abs(composite)
    if abs_score >= 50:
        strength = 'STRONG'
    elif abs_score >= 30:
        strength = 'MODERATE'
    elif abs_score >= 15:
        strength = 'WEAK'
    else:
        strength = 'NO_TRADE'

    if composite > 15:
        direction = 'LONG'
    elif composite < -15:
        direction = 'SHORT'
    else:
        direction = 'FLAT'

    # 7. Trade type
    if strength == 'NO_TRADE' or direction == 'FLAT':
        trade_type = 'NO_TRADE'
    elif direction == 'LONG' and strength == 'STRONG':
        trade_type = '0DTE_CALL'  # Strong = intraday scalp
    elif direction == 'LONG':
        trade_type = 'SWING_LONG'
    elif direction == 'SHORT' and strength == 'STRONG':
        trade_type = '0DTE_PUT'
    else:
        trade_type = 'SWING_SHORT'

    return {
        'composite_score': composite,
        'direction': direction,
        'strength': strength,
        'trade_type': trade_type,
        'confidence': confidence,
        'regime': regime,
        'risk_multiplier': REGIME_RISK_MULTIPLIER.get(regime, 0.5),
        'regime_score': r_score,
        'indicator_score': i_score,
        'strat_score': s_score,
        'reasoning': reasoning,
    }


def filter_grade_a_only(signal):
    """
    Final gate: Only pass through Grade A+ and A setups.
    Returns True if this signal is worth trading.
    """
    if signal['strength'] in ['NO_TRADE', 'WEAK']:
        return False
    if signal['confidence'] < 40:
        return False
    return True
