"""
universe.py — Refined Trading Universe Manager
PTR Score: PTR = (grade_accuracy * capital_efficiency) / (regime_complexity * data_penalty)
"""

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UNIVERSE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "universe.json")
MAX_UNIVERSE_SIZE = 15

SECTOR_GROUPS = {
    'Semiconductors': ['NVDA', 'AMD', 'MU', 'AVGO', 'INTC', 'QCOM'],
    'Mega Cap Tech':  ['AAPL', 'MSFT', 'META', 'GOOGL', 'AMZN'],
    'Financials':     ['JPM', 'BAC', 'GS', 'MS', 'WFC'],
    'Energy':         ['XOM', 'CVX', 'OXY', 'SLB'],
    'Consumer':       ['WMT', 'COST', 'TGT'],
    'Core Indices':   ['SPY', 'QQQ', 'IWM'],
}

REGIME_COMPLEXITY = {
    'Bull_Quiet':    0.3,
    'Bull_Volatile': 0.6,
    'Bear_Quiet':    0.4,
    'Bear_Volatile': 0.8,
    'Chop':          1.0,
}

DATA_PENALTY = {
    'iex': 0.85,   # free tier - 15min delayed
    'sip': 1.0,    # real-time - full score
}


def get_sector_for_ticker(ticker: str) -> str:
    for sector, tickers in SECTOR_GROUPS.items():
        if ticker in tickers:
            return sector
    return 'Other'


def load_universe() -> list[dict]:
    if not os.path.exists(UNIVERSE_FILE):
        return []
    try:
        with open(UNIVERSE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return []


def save_universe(data: list[dict]):
    try:
        with open(UNIVERSE_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass


def calculate_ptr_score(
    grade_accuracy: float,
    avg_confidence: float,
    regime: str = 'Chop',
    data_feed: str = 'iex',
    buying_power: float = 100000,
) -> float:
    """
    PTR = (η · E) / (k · D)
    η = grade_accuracy (0-1)
    E = capital_efficiency (normalized buying power)
    k = regime_complexity (0.3 to 1.0)
    D = data_quality (0.85 free / 1.0 real-time)
    """
    eta = grade_accuracy
    E   = min(buying_power / 100000, 1.0)
    k   = REGIME_COMPLEXITY.get(regime, 1.0)
    D   = DATA_PENALTY.get(data_feed, 0.85)

    if k == 0 or D == 0:
        return 0.0

    ptr = (eta * E) / (k * D)
    return round(ptr * 100, 2)


def calculate_universe_score(ticker_history: list[dict]) -> dict:
    """
    Score a ticker's suitability for the permanent universe.
    Combines PTR formula with appearance frequency.
    """
    if not ticker_history:
        return {'universe_score': 0, 'in_universe': False}

    total        = len(ticker_history)
    a_grades     = sum(1 for t in ticker_history if t.get('grade') in ['A+', 'A'])
    avg_conf     = sum(t.get('confidence', 0) for t in ticker_history) / total
    grade_acc    = a_grades / total
    last_regime  = ticker_history[0].get('regime', 'Chop')

    ptr = calculate_ptr_score(
        grade_accuracy=grade_acc,
        avg_confidence=avg_conf,
        regime=last_regime,
        data_feed='iex',
    )

    # Base score: PTR weighted + frequency bonus
    base_score = (ptr * 0.5) + (avg_conf * 0.3) + min(total * 5, 30)

    return {
        'ticker':         ticker_history[0].get('ticker', ''),
        'universe_score': round(base_score, 1),
        'ptr_score':      ptr,
        'appearances':    total,
        'a_grade_rate':   round(grade_acc * 100, 1),
        'avg_confidence': round(avg_conf, 1),
        'last_regime':    last_regime,
        'sector':         get_sector_for_ticker(ticker_history[0].get('ticker', '')),
        'in_universe':    base_score >= 60,
        'last_seen':      ticker_history[0].get('scan_time', ''),
    }


def add_to_universe(ticker: str, metadata: dict = None) -> tuple[bool, str]:
    universe = load_universe()
    if len(universe) >= MAX_UNIVERSE_SIZE:
        return False, f"Universe full ({MAX_UNIVERSE_SIZE} max). Remove a ticker first."
    if any(u['ticker'] == ticker for u in universe):
        return False, f"{ticker} already in universe."
    entry = {
        'ticker':    ticker,
        'sector':    get_sector_for_ticker(ticker),
        'added_at':  datetime.now(ET).isoformat(),
        'score':     metadata.get('universe_score', 0) if metadata else 0,
        'ptr_score': metadata.get('ptr_score', 0) if metadata else 0,
        **(metadata or {}),
    }
    universe.insert(0, entry)
    save_universe(universe)
    return True, f"{ticker} added to universe."


def remove_from_universe(ticker: str) -> tuple[bool, str]:
    universe = load_universe()
    updated  = [u for u in universe if u['ticker'] != ticker]
    if len(updated) == len(universe):
        return False, f"{ticker} not found in universe."
    save_universe(updated)
    return True, f"{ticker} removed from universe."


def auto_build_universe(scan_history: list[dict]) -> list[dict]:
    """
    Score all tickers from scan history.
    Keep top 15 by universe_score.
    """
    from collections import defaultdict
    ticker_map = defaultdict(list)
    for result in scan_history:
        t = result.get('ticker')
        if t:
            ticker_map[t].append(result)

    scored = []
    for ticker, history in ticker_map.items():
        score_data = calculate_universe_score(history)
        if score_data['in_universe']:
            scored.append(score_data)

    scored.sort(key=lambda x: x['universe_score'], reverse=True)
    top15 = scored[:MAX_UNIVERSE_SIZE]

    universe = []
    for s in top15:
        ok, _ = add_to_universe(s['ticker'], s)
        if ok:
            universe.append(s)

    return load_universe()


def get_universe_summary() -> dict:
    universe = load_universe()
    by_sector = {}
    for u in universe:
        sec = u.get('sector', 'Other')
        by_sector.setdefault(sec, []).append(u['ticker'])
    return {
        'total':     len(universe),
        'tickers':   [u['ticker'] for u in universe],
        'by_sector': by_sector,
        'avg_score': round(sum(u.get('score', 0) for u in universe) / max(len(universe), 1), 1),
    }
