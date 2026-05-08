# ================================================
# TRADE_GRADER.PY - Setup Grading Engine
# Built from Blackhero's actual trading playbook
# Grades: A+ (Explosive) / A (High) / B (Solid) / C (Speculative) / No Trade
# Produces separate INTRADAY and SWING signals
# ================================================

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# SECTOR MAPPING
# ─────────────────────────────────────────────

SECTOR_MAP = {
    # Tech / Electronic Technology
    'XLK': ['AAPL','MSFT','NVDA','AMD','MU','AVGO','QCOM','INTC','ADBE','CRM','ORCL','TXN',
            'AMAT','LRCX','KLAC','SNPS','CDNS','ADI','PANW','INTU'],
    # Healthcare
    'XLV': ['UNH','JNJ','ABBV','LLY','MRK','TMO','ABT','DHR','AMGN','GILD','VRTX','REGN',
            'SYK','BDX','ISRG','ZTS','CI','MCO'],
    # Consumer Discretionary
    'XLY': ['TSLA','AMZN','HD','MCD','NKE','LOW','TGT','CMG','BKNG','GM','F','ELF','WMT'],
    # Financials
    'XLF': ['JPM','V','MA','GS','MS','BLK','AXP','SCHW','PNC','USB','BRK-B','SPGI','ICE',
            'CME','CB','MMC','FI'],
    # Energy
    'XLE': ['XOM','CVX','EOG','SLB','CL'],
    # Industrials
    'XLI': ['CAT','BA','DE','UPS','GE','RTX','HON'],
    # Utilities
    'XLU': ['NEE','SO','DUK'],
    # Consumer Staples
    'XLP': ['PG','KO','PEP','COST','MDLZ','PM','MO'],
    # Real Estate
    'XLRE': ['PLD'],
    # Communication
    'XLC': ['GOOGL','GOOG','META'],
}

# Reverse lookup: stock → sector ETF
STOCK_TO_SECTOR = {}
for etf, stocks in SECTOR_MAP.items():
    for s in stocks:
        STOCK_TO_SECTOR[s] = etf

# Tickers with high intraday move potential ($1-3+)
HIGH_MOVE_TICKERS = {
    'NVDA','TSLA','AAPL','AMD','AMZN','GOOGL','META','MSFT','MU','NFLX',
    'SPY','QQQ','IWM','BA','GS','BKNG','CMG','AVGO','LLY','UNH',
}

# Core indices
INDICES = ['SPY', 'QQQ', 'IWM']


# ─────────────────────────────────────────────
# SESSION TIMING
# ─────────────────────────────────────────────

def get_session_context():
    """
    Returns current session window and whether entries are allowed.
    Based on Blackhero's rules:
      9:30-9:45  = Observe only
      9:45-10:30 = Prime entries
      10:30-12:00 = Okay, momentum fading
      12:00-13:00 = Lunch chop — avoid
      13:00-15:00 = Power hour zone
      15:00-15:45 = Final push (caution, theta)
      15:45-16:00 = Exit all 0DTE
    """
    now = datetime.now()
    h, m = now.hour, now.minute
    t = h * 60 + m  # Minutes since midnight

    if t < 9 * 60 + 30:
        return {'session': 'Pre-Market', 'allow_0dte': False, 'allow_swing': True,
                'quality': 'prep', 'note': 'Pre-market prep window'}
    elif t < 9 * 60 + 45:
        return {'session': 'Open Observe', 'allow_0dte': False, 'allow_swing': False,
                'quality': 'observe', 'note': 'Observe only — no entries first 15 min'}
    elif t < 10 * 60 + 30:
        return {'session': 'Prime Entry', 'allow_0dte': True, 'allow_swing': True,
                'quality': 'prime', 'note': '🟢 Prime entry window — best setups here'}
    elif t < 12 * 60:
        return {'session': 'Mid-Morning', 'allow_0dte': True, 'allow_swing': True,
                'quality': 'good', 'note': 'Momentum may be fading — be selective'}
    elif t < 13 * 60:
        return {'session': 'Lunch Chop', 'allow_0dte': False, 'allow_swing': False,
                'quality': 'avoid', 'note': '⚠️ Lunch chop — no new entries'}
    elif t < 15 * 60:
        return {'session': 'Power Hour', 'allow_0dte': True, 'allow_swing': True,
                'quality': 'prime', 'note': '🟢 Power hour — prime for continuation'}
    elif t < 15 * 60 + 45:
        return {'session': 'Final Push', 'allow_0dte': True, 'allow_swing': False,
                'quality': 'caution', 'note': '⚠️ Final push — theta accelerating'}
    elif t < 16 * 60:
        return {'session': 'Exit Window', 'allow_0dte': False, 'allow_swing': False,
                'quality': 'exit', 'note': '🔴 Exit all 0DTE positions'}
    else:
        return {'session': 'After Hours', 'allow_0dte': False, 'allow_swing': True,
                'quality': 'prep', 'note': 'After hours — prep for tomorrow'}


# ─────────────────────────────────────────────
# MULTI-TIMEFRAME CONTINUITY (FTC)
# ─────────────────────────────────────────────

def get_candle_direction(row):
    """Simple candle direction: up, down, or neutral."""
    if row['Close'] > row['Open']:
        return 'up'
    elif row['Close'] < row['Open']:
        return 'down'
    return 'neutral'


def check_ftc(df_daily, df_weekly=None, df_4h=None, df_1h=None, df_15m=None):
    """
    Check Full Timeframe Continuity.
    Returns: dict with alignment count, direction, and whether FTC is confirmed.

    Intraday FTC: Weekly → Daily → 4H → 1H → 15m
    Swing FTC: Monthly → Weekly → Daily → 4H
    """
    directions = {}

    if df_daily is not None and len(df_daily) > 1:
        directions['daily'] = get_candle_direction(df_daily.iloc[-1])

    if df_weekly is not None and len(df_weekly) > 1:
        directions['weekly'] = get_candle_direction(df_weekly.iloc[-1])

    if df_4h is not None and len(df_4h) > 1:
        directions['4h'] = get_candle_direction(df_4h.iloc[-1])

    if df_1h is not None and len(df_1h) > 1:
        directions['1h'] = get_candle_direction(df_1h.iloc[-1])

    if df_15m is not None and len(df_15m) > 1:
        directions['15m'] = get_candle_direction(df_15m.iloc[-1])

    if not directions:
        return {'aligned': 0, 'total': 0, 'direction': 'neutral', 'ftc_confirmed': False, 'details': {}}

    # Count alignment
    up_count = sum(1 for d in directions.values() if d == 'up')
    down_count = sum(1 for d in directions.values() if d == 'down')
    total = len(directions)

    if up_count >= total * 0.8:
        direction = 'bullish'
        aligned = up_count
    elif down_count >= total * 0.8:
        direction = 'bearish'
        aligned = down_count
    elif up_count > down_count:
        direction = 'leaning_bull'
        aligned = up_count
    elif down_count > up_count:
        direction = 'leaning_bear'
        aligned = down_count
    else:
        direction = 'mixed'
        aligned = max(up_count, down_count)

    ftc_confirmed = aligned >= max(3, total - 1)  # Need almost all aligned

    return {
        'aligned': aligned,
        'total': total,
        'direction': direction,
        'ftc_confirmed': ftc_confirmed,
        'details': directions,
    }


# ─────────────────────────────────────────────
# SECTOR CORRELATION CHECK
# ─────────────────────────────────────────────

def check_sector_correlation(ticker, ticker_direction):
    """
    Verify sector alignment: Index → Sector ETF → Stock.
    Returns correlation score and details.
    """
    sector_etf = STOCK_TO_SECTOR.get(ticker)
    if not sector_etf:
        return {'correlated': False, 'score': 0, 'sector': None, 'note': 'No sector mapping'}

    try:
        from lib.data_client import get_daily
        df_spy = get_daily('SPY', days=10)
        df_sector = get_daily(sector_etf, days=10)

        if df_spy.empty or df_sector.empty:
            return {'correlated': False, 'score': 0, 'sector': sector_etf, 'note': 'Data unavailable'}

        spy_dir = 'up' if df_spy['Close'].iloc[-1] > df_spy['Open'].iloc[-1] else 'down'
        sector_dir = 'up' if df_sector['Close'].iloc[-1] > df_sector['Open'].iloc[-1] else 'down'

        # Check 3-day trend
        spy_trend = 'up' if len(df_spy) >= 3 and df_spy['Close'].iloc[-1] > df_spy['Close'].iloc[-3] else 'down'
        sector_trend = 'up' if len(df_sector) >= 3 and df_sector['Close'].iloc[-1] > df_sector['Close'].iloc[-3] else 'down'

        score = 0
        notes = []

        # SPY alignment
        if (ticker_direction == 'LONG' and spy_dir == 'up') or \
           (ticker_direction == 'SHORT' and spy_dir == 'down'):
            score += 1
            notes.append('SPY aligned')
        else:
            notes.append('SPY conflict')

        # Sector alignment
        if (ticker_direction == 'LONG' and sector_dir == 'up') or \
           (ticker_direction == 'SHORT' and sector_dir == 'down'):
            score += 1
            notes.append(f'{sector_etf} aligned')
        else:
            notes.append(f'{sector_etf} conflict')

        # Trend alignment
        if (ticker_direction == 'LONG' and sector_trend == 'up') or \
           (ticker_direction == 'SHORT' and sector_trend == 'down'):
            score += 1
            notes.append('Sector trending')

        return {
            'correlated': score >= 2,
            'score': score,
            'max_score': 3,
            'sector': sector_etf,
            'spy_dir': spy_dir,
            'sector_dir': sector_dir,
            'note': ' | '.join(notes),
        }

    except Exception:
        return {'correlated': False, 'score': 0, 'sector': sector_etf, 'note': 'Check failed'}


# ─────────────────────────────────────────────
# ATR MOVE POTENTIAL
# ─────────────────────────────────────────────

def check_atr_room(df, direction):
    """
    Check if there's ATR room left for the move.
    For 0DTE, need $1-3+ potential.
    Returns move potential in $ and whether it's sufficient.
    """
    if 'atr' not in df.columns or len(df) < 2:
        return {'atr': 0, 'room': 0, 'sufficient': False}

    atr = float(df['atr'].iloc[-1])
    close = float(df['Close'].iloc[-1])
    high = float(df['High'].iloc[-1])
    low = float(df['Low'].iloc[-1])
    day_range = high - low

    # How much of ATR has been used today
    used = day_range
    remaining = max(0, atr - used)

    # Dollar move potential
    move_pct = (atr / close) * 100

    return {
        'atr': round(atr, 2),
        'atr_pct': round(move_pct, 2),
        'used': round(used, 2),
        'remaining': round(remaining, 2),
        'sufficient_0dte': remaining > 0.5 or atr > 1.0,  # Need meaningful move left
        'sufficient_swing': atr > 0.5,
        'move_potential': f"${atr:.2f} ({move_pct:.1f}%)",
    }


# ─────────────────────────────────────────────
# SETUP GRADING ENGINE
# ─────────────────────────────────────────────

def grade_setup(ticker, regime, regime_confidence, composite_score, direction,
                ftc, strat_patterns, indicators, sector_corr, atr_info,
                session=None):
    """
    Grade a setup using Blackhero's probability system.

    A+ (Explosive): Full FTC + Strat pattern at major level + order block +
                     ATR room + sector confirmation
    A  (High):      4+ TFs aligned + clear trigger at S/R + moderate confirmation
    B  (Solid):     3 TFs aligned + pattern at minor level + some confirmation
    C  (Speculative): 2 TFs aligned + weak pattern. Skip unless perfect.
    No Trade:       Inside bars on 0DTE, FTC conflicts, low volume, lunch chop

    Returns separate grades for INTRADAY and SWING.
    """
    if session is None:
        session = get_session_context()

    # ─── SCORE COMPONENTS ───
    points_intraday = 0
    points_swing = 0
    reasons_intraday = []
    reasons_swing = []
    flags = []  # Hard disqualifiers

    # 1. REGIME (both)
    if regime_confidence >= 90:
        points_intraday += 20; points_swing += 20
        reasons_intraday.append(f"Regime {regime} ({regime_confidence}%) — strong")
        reasons_swing.append(f"Regime {regime} ({regime_confidence}%) — strong")
    elif regime_confidence >= 85:
        points_intraday += 15; points_swing += 15
        reasons_intraday.append(f"Regime {regime} ({regime_confidence}%)")
        reasons_swing.append(f"Regime {regime} ({regime_confidence}%)")
    elif regime_confidence >= 70:
        points_intraday += 8; points_swing += 10
    else:
        points_intraday += 0; points_swing += 3

    # Regime-direction alignment
    bull_regime = regime in ['Bull_Quiet', 'Bull_Volatile']
    bear_regime = regime in ['Bear_Quiet', 'Bear_Volatile']

    if (direction == 'LONG' and bull_regime) or (direction == 'SHORT' and bear_regime):
        points_intraday += 10; points_swing += 10
        reasons_intraday.append("Direction aligned with regime")
        reasons_swing.append("Direction aligned with regime")
    elif regime == 'Chop':
        points_intraday -= 15
        points_swing -= 5
        flags.append("Chop regime — reduced confidence")

    # 2. FTC
    if ftc['ftc_confirmed']:
        aligned = ftc['aligned']
        if aligned >= 5:
            points_intraday += 25; points_swing += 25
            reasons_intraday.append(f"Full FTC ({aligned}/{ftc['total']} aligned) ✅")
            reasons_swing.append(f"Full FTC ({aligned}/{ftc['total']} aligned) ✅")
        elif aligned >= 4:
            points_intraday += 20; points_swing += 20
            reasons_intraday.append(f"Strong FTC ({aligned}/{ftc['total']})")
            reasons_swing.append(f"Strong FTC ({aligned}/{ftc['total']})")
        elif aligned >= 3:
            points_intraday += 12; points_swing += 15
            reasons_intraday.append(f"Partial FTC ({aligned}/{ftc['total']})")
            reasons_swing.append(f"Partial FTC ({aligned}/{ftc['total']})")
    else:
        if ftc['direction'] == 'mixed':
            points_intraday -= 10; points_swing -= 5
            flags.append("FTC conflict — mixed timeframes")

    # 3. STRAT PATTERNS
    best_grade = None
    for p in strat_patterns:
        grade = p.get('grade', 'C')
        pat_dir = p.get('direction', 'pending')

        # Pattern-direction alignment
        aligned_with_trade = (
            (direction == 'LONG' and pat_dir == 'up') or
            (direction == 'SHORT' and pat_dir == 'down')
        )

        if grade == 'A+':
            if aligned_with_trade:
                points_intraday += 20; points_swing += 15
                reasons_intraday.append(f"🎯 {p['name']} [A+] → {pat_dir}")
                reasons_swing.append(f"🎯 {p['name']} [A+] → {pat_dir}")
                best_grade = 'A+'
            else:
                flags.append(f"A+ pattern opposite direction ({p['name']})")
        elif grade == 'A':
            if aligned_with_trade:
                points_intraday += 15; points_swing += 12
                reasons_intraday.append(f"🎯 {p['name']} [A] → {pat_dir}")
                reasons_swing.append(f"🎯 {p['name']} [A] → {pat_dir}")
                if best_grade not in ['A+']: best_grade = 'A'
        elif grade == 'B+':
            points_intraday += 8; points_swing += 8
            if best_grade not in ['A+', 'A']: best_grade = 'B+'
        elif grade == 'B':
            points_intraday += 4; points_swing += 5
            if best_grade not in ['A+', 'A', 'B+']: best_grade = 'B'

        # Inside bar on 0DTE = theta decay risk
        if p['name'] in ['Inside Compression', '3-1 Compression'] and pat_dir == 'pending':
            points_intraday -= 10
            flags.append("Inside bar compression — 0DTE theta risk")

    if not strat_patterns:
        points_intraday -= 5; points_swing -= 3
        flags.append("No active Strat patterns")

    # 4. SECTOR CORRELATION
    if sector_corr.get('correlated'):
        sc = sector_corr['score']
        points_intraday += sc * 5; points_swing += sc * 4
        reasons_intraday.append(f"Sector confirmed ({sector_corr['note']})")
        reasons_swing.append(f"Sector confirmed ({sector_corr['note']})")
    else:
        points_intraday -= 5; points_swing -= 3
        if sector_corr.get('note'):
            flags.append(f"Sector: {sector_corr['note']}")

    # 5. INDICATORS (regime-gated, already in composite)
    if abs(composite_score) >= 50:
        points_intraday += 10; points_swing += 10
        reasons_intraday.append(f"Strong indicator confluence ({composite_score:+d})")
        reasons_swing.append(f"Strong indicator confluence ({composite_score:+d})")
    elif abs(composite_score) >= 30:
        points_intraday += 5; points_swing += 7

    # RSI extreme check
    rsi = indicators.get('rsi', 50)
    if direction == 'LONG' and rsi > 75:
        points_intraday -= 5
        flags.append(f"RSI overbought ({rsi}) — pullback risk")
    elif direction == 'SHORT' and rsi < 25:
        points_intraday -= 5
        flags.append(f"RSI oversold ({rsi}) — bounce risk")

    # Volume
    if indicators.get('high_volume'):
        points_intraday += 5; points_swing += 3
        reasons_intraday.append("High volume confirmation")

    # ADX trending
    if indicators.get('adx', 0) > 25 and indicators.get('adx_rising'):
        points_intraday += 5; points_swing += 5
        reasons_intraday.append(f"ADX trending ({indicators['adx']:.0f}) ↑")

    # 6. ATR ROOM
    if atr_info.get('sufficient_0dte'):
        points_intraday += 5
        reasons_intraday.append(f"ATR room: {atr_info['move_potential']}")
    else:
        points_intraday -= 10
        flags.append(f"ATR exhausted — limited 0DTE move potential")

    if atr_info.get('sufficient_swing'):
        points_swing += 3

    # 7. HIGH MOVE TICKER BONUS
    if ticker in HIGH_MOVE_TICKERS:
        points_intraday += 5
        reasons_intraday.append("High-move ticker ($1-3+ potential)")

    # 8. SESSION TIMING (intraday only)
    if session['quality'] == 'prime':
        points_intraday += 5
        reasons_intraday.append(f"🟢 {session['session']} window")
    elif session['quality'] == 'avoid':
        points_intraday -= 20
        flags.append(f"⚠️ {session['session']} — no 0DTE entries")
    elif session['quality'] == 'exit':
        points_intraday -= 30
        flags.append(f"🔴 {session['session']} — exit all 0DTE")
    elif session['quality'] == 'observe':
        points_intraday -= 15
        flags.append("Observe only — first 15 min")

    # ─── FINAL GRADING ───

    def assign_grade(points, flags_list, is_intraday=True):
        # Hard disqualifiers
        if is_intraday and not session.get('allow_0dte', True):
            return 'NO_TRADE', points
        if not is_intraday and not session.get('allow_swing', True):
            return 'NO_TRADE', points

        if points >= 75:
            return 'A+', points
        elif points >= 55:
            return 'A', points
        elif points >= 35:
            return 'B', points
        elif points >= 20:
            return 'C', points
        else:
            return 'NO_TRADE', points

    grade_intraday, pts_intra = assign_grade(points_intraday, flags, is_intraday=True)
    grade_swing, pts_swing = assign_grade(points_swing, flags, is_intraday=False)

    # ─── TRADE TYPE MAPPING ───

    def map_trade_type(grade, direction, is_intraday):
        if grade in ['NO_TRADE', 'C']:
            return 'SKIP'
        if is_intraday:
            if direction == 'LONG':
                return '0DTE CALL' if grade in ['A+', 'A'] else 'WATCH CALL'
            else:
                return '0DTE PUT' if grade in ['A+', 'A'] else 'WATCH PUT'
        else:
            if direction == 'LONG':
                return 'SWING LONG' if grade in ['A+', 'A'] else 'WATCH LONG'
            else:
                return 'SWING SHORT' if grade in ['A+', 'A'] else 'WATCH SHORT'

    trade_intraday = map_trade_type(grade_intraday, direction, True)
    trade_swing = map_trade_type(grade_swing, direction, False)

    # ─── RISK SIZING ───

    risk_mult = {"Bull_Quiet":1.0, "Bull_Volatile":0.75, "Chop":0.4, "Bear_Quiet":1.0, "Bear_Volatile":0.5}.get(regime, 0.5)

    risk_0dte = round(75 * risk_mult)   # Base $75
    risk_swing = round(150 * risk_mult)  # Base $150

    if grade_intraday == 'A+':
        risk_0dte = round(risk_0dte * 1.2)  # Slight bump for explosive setups
    elif grade_intraday == 'B':
        risk_0dte = round(risk_0dte * 0.6)

    if grade_swing == 'A+':
        risk_swing = round(risk_swing * 1.1)
    elif grade_swing == 'B':
        risk_swing = round(risk_swing * 0.6)

    # Contracts
    contracts_0dte = 2 if risk_0dte >= 50 else 1
    contracts_swing = 2 if risk_swing >= 80 else 1

    # ─── STRIKE GUIDANCE ───
    if trade_intraday != 'SKIP':
        strike_note = "ATM or 1 strike OTM | 2 contracts: exit 1st at +150%, run 2nd"
    else:
        strike_note = ""

    if trade_swing != 'SKIP':
        swing_strike_note = "ATM or 1 OTM, 2-3 week expiry | Exit 1st at +30%, runner until reversal"
    else:
        swing_strike_note = ""

    return {
        # Intraday
        'grade_intraday': grade_intraday,
        'points_intraday': pts_intra,
        'trade_intraday': trade_intraday,
        'risk_0dte': risk_0dte,
        'contracts_0dte': contracts_0dte,
        'strike_note': strike_note,
        'reasons_intraday': reasons_intraday,

        # Swing
        'grade_swing': grade_swing,
        'points_swing': pts_swing,
        'trade_swing': trade_swing,
        'risk_swing': risk_swing,
        'contracts_swing': contracts_swing,
        'swing_strike_note': swing_strike_note,
        'reasons_swing': reasons_swing,

        # Shared
        'direction': direction,
        'flags': flags,
        'session': session,
        'sector_corr': sector_corr,
        'ftc': ftc,
        'atr': atr_info,
        'best_strat_grade': best_grade,
        'regime_mult': risk_mult,
    }


# ─────────────────────────────────────────────
# 7-LEVEL FTFC STACK BUILDER
# ─────────────────────────────────────────────

def build_ftfc_stack(ticker: str, mode: str = 'intraday') -> dict:
    """
    Build a multi-timeframe Full Timeframe Continuity stack.
    mode='intraday': Monthly→Weekly→Daily→4H→65min→45min→15min (7 levels)
    mode='swing':    Monthly→Weekly→Daily→4H (4 levels)
    Returns dict with stack list, consensus, confirmed, up/dn counts.
    """
    from lib.data_client import get_daily, get_weekly, get_monthly, get_4h, get_65min, get_45min, get_15min

    if mode == 'intraday':
        timeframe_fetchers = [
            ('Monthly', lambda: get_monthly(ticker)),
            ('Weekly',  lambda: get_weekly(ticker, days=90)),
            ('Daily',   lambda: get_daily(ticker, days=30)),
            ('4H',      lambda: get_4h(ticker, days=30)),
            ('65min',   lambda: get_65min(ticker, days=10)),
            ('45min',   lambda: get_45min(ticker, days=7)),
            ('15min',   lambda: get_15min(ticker, days=3)),
        ]
    else:
        timeframe_fetchers = [
            ('Monthly', lambda: get_monthly(ticker)),
            ('Weekly',  lambda: get_weekly(ticker, days=180)),
            ('Daily',   lambda: get_daily(ticker, days=90)),
            ('4H',      lambda: get_4h(ticker, days=60)),
        ]

    stack = []
    for name, fetch_fn in timeframe_fetchers:
        try:
            df = fetch_fn()
            if df.empty or len(df) < 2:
                stack.append({'tf': name, 'direction': 'neutral', 'close': None})
                continue
            last = df.iloc[-1]
            direction = 'up' if float(last['Close']) > float(last['Open']) else 'down'
            stack.append({
                'tf':        name,
                'direction': direction,
                'close':     round(float(last['Close']), 2),
                'open':      round(float(last['Open']), 2),
            })
        except Exception:
            stack.append({'tf': name, 'direction': 'neutral', 'close': None})

    dirs   = [s['direction'] for s in stack if s['direction'] != 'neutral']
    up_cnt = sum(1 for d in dirs if d == 'up')
    dn_cnt = sum(1 for d in dirs if d == 'down')
    total  = len(dirs)

    if total == 0:
        consensus = 'neutral'
    elif up_cnt == total:
        consensus = 'bullish'
    elif dn_cnt == total:
        consensus = 'bearish'
    elif up_cnt >= total * 0.75:
        consensus = 'leaning_bull'
    elif dn_cnt >= total * 0.75:
        consensus = 'leaning_bear'
    else:
        consensus = 'mixed'

    return {
        'stack':     stack,
        'consensus': consensus,
        'confirmed': (up_cnt == total or dn_cnt == total) and total >= 3,
        'up_count':  up_cnt,
        'dn_count':  dn_cnt,
        'total':     total,
        'mode':      mode,
    }


# ─────────────────────────────────────────────
# FULL GRADED ANALYSIS (called by scanner)
# ─────────────────────────────────────────────

def grade_ticker_full(ticker, regime, regime_confidence, composite_score,
                      direction, indicators, strat_patterns, df):
    """
    Run the complete grading pipeline for a ticker.
    Fetches additional timeframes and sector data.
    Returns the full grade dict.
    """
    session = get_session_context()

    # FTC check (use daily data we already have + try weekly)
    try:
        from lib.data_client import get_weekly
        df_weekly = get_weekly(ticker, days=180)
        if df_weekly.empty:
            df_weekly = None
    except Exception:
        df_weekly = None

    ftc = check_ftc(df_daily=df, df_weekly=df_weekly)

    # Sector correlation
    sector_corr = check_sector_correlation(ticker, direction)

    # ATR room
    atr_info = check_atr_room(df, direction)

    # Grade it
    grade = grade_setup(
        ticker=ticker,
        regime=regime,
        regime_confidence=regime_confidence,
        composite_score=composite_score,
        direction=direction,
        ftc=ftc,
        strat_patterns=strat_patterns,
        indicators=indicators,
        sector_corr=sector_corr,
        atr_info=atr_info,
        session=session,
    )

    # Attach FTFC stack (intraday mode)
    try:
        grade['ftfc_stack'] = build_ftfc_stack(ticker, mode='intraday')
    except Exception:
        grade['ftfc_stack'] = None

    return grade
