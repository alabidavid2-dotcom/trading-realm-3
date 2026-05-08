"""
tracker.py — Watchlist Trade Tracker
Persists tracked tickers + entry data to Supabase.
Survives restarts. Manual clear available.
"""

from datetime import datetime
from zoneinfo import ZoneInfo
from config import supabase

ET = ZoneInfo("America/New_York")


# ── Load ───────────────────────────────────────────────────────────────────────

def load_tracked() -> list[dict]:
    try:
        res = supabase.table('tracker_positions').select('*').order('added_at', desc=True).execute()
        return res.data or []
    except Exception:
        return []


# ── Add / Remove ───────────────────────────────────────────────────────────────

def add_tracked(
    ticker: str,
    grade: str,
    direction: str,
    regime: str,
    signal_score: int,
    entry_price: float,
    trade_type: str,
    patterns: list = None,
    note: str = "",
) -> list[dict]:
    """Add a ticker to the watchlist tracker. Returns updated list."""
    try:
        existing = supabase.table('tracker_positions').select('ticker').eq('ticker', ticker).eq('active', True).execute()
        if existing.data:
            return load_tracked()
    except Exception:
        pass

    now = datetime.now(ET)
    entry = {
        "ticker":        ticker,
        "grade":         grade,
        "direction":     direction,
        "regime":        regime,
        "signal_score":  signal_score,
        "entry_price":   entry_price,
        "trade_type":    trade_type,
        "patterns":      patterns or [],
        "note":          note,
        "added_at":      now.isoformat(),
        "active":        True,
        "current_price": entry_price,
        "pnl_dollars":   0.0,
        "pnl_pct":       0.0,
        "last_updated":  now.isoformat(),
    }
    try:
        supabase.table('tracker_positions').insert(entry).execute()
    except Exception:
        pass
    return load_tracked()


def remove_tracked(ticker: str) -> list[dict]:
    """Remove a ticker from the active tracker."""
    try:
        supabase.table('tracker_positions').delete().eq('ticker', ticker).execute()
    except Exception:
        pass
    return load_tracked()


def clear_all_tracked() -> list[dict]:
    """Clear all tracked tickers."""
    try:
        supabase.table('tracker_positions').delete().neq('ticker', '').execute()
    except Exception:
        pass
    return []


# ── Update prices ──────────────────────────────────────────────────────────────

def update_tracked_prices(tracked: list[dict]) -> list[dict]:
    """Fetch latest prices, recalculate P&L, and persist each update to Supabase."""
    if not tracked:
        return tracked

    try:
        from lib.data_client import get_daily
        for t in tracked:
            if not t.get("active", True):
                continue
            sym = t["ticker"]
            try:
                df = get_daily(sym, days=5)
                if df.empty:
                    continue
                price = float(df['Close'].dropna().iloc[-1])
                entry = float(t["entry_price"])
                direction = t.get("direction", "LONG")
                if direction == "LONG":
                    pnl_pct     = (price - entry) / entry * 100
                    pnl_dollars = price - entry
                else:
                    pnl_pct     = (entry - price) / entry * 100
                    pnl_dollars = entry - price
                t["current_price"] = round(price, 2)
                t["pnl_pct"]       = round(pnl_pct, 2)
                t["pnl_dollars"]   = round(pnl_dollars, 4)
                t["last_updated"]  = datetime.now(ET).isoformat()

                supabase.table('tracker_positions').update({
                    "current_price": t["current_price"],
                    "pnl_pct":       t["pnl_pct"],
                    "pnl_dollars":   t["pnl_dollars"],
                    "last_updated":  t["last_updated"],
                }).eq('ticker', sym).execute()
            except Exception:
                pass
    except Exception:
        pass

    return tracked


# ── Options formula estimate ────────────────────────────────────────────────────

def estimate_option_contract(
    ticker: str,
    direction: str,
    current_price: float,
    trade_type: str,
    atr: float = None,
    grade: str = None,
) -> dict:
    """
    Grade-based option contract estimate.
    A+ → ATM (delta ~0.50), A → 1-strike OTM (delta ~0.35), B → 2-strike OTM (delta ~0.20).
    Returns: strike, c_or_p, expiry, est_premium, delta_estimate, theta_warning, moneyness.
    """
    import math
    from datetime import date, timedelta

    today  = date.today()
    c_or_p = "CALL" if direction == "LONG" else "PUT"

    # Grade-based moneyness
    if grade == 'A+':
        moneyness, otm_strikes = 'ATM',  0
    elif grade == 'A':
        moneyness, otm_strikes = 'OTM1', 1
    elif grade == 'B':
        moneyness, otm_strikes = 'OTM2', 2
    else:
        moneyness, otm_strikes = 'ATM',  0

    # Strike increment by price
    if current_price < 50:
        increment = 1
    elif current_price < 200:
        increment = 5
    else:
        increment = 10

    base_strike = round(current_price / increment) * increment
    if direction == "LONG":
        strike = base_strike + otm_strikes * increment
    else:
        strike = base_strike - otm_strikes * increment

    # Expiration
    if trade_type == "0DTE":
        expiry, dte = today, 0
    else:
        days_ahead = (4 - today.weekday()) % 7 or 7
        expiry = today + timedelta(days=days_ahead)
        dte    = days_ahead

    if atr is None:
        atr = current_price * 0.015

    est_premium = round(0.4 * atr * math.sqrt(dte + 1), 2)
    est_premium = max(est_premium, 0.05)

    # Delta estimate by moneyness
    delta_abs = {'ATM': 0.50, 'OTM1': 0.35, 'OTM2': 0.20}.get(moneyness, 0.50)
    delta_estimate = delta_abs if c_or_p == "CALL" else -delta_abs

    # Theta warning
    if dte == 0:
        theta_warning = True
        theta_message = "0DTE: theta decay accelerates after 2 PM ET — exit before 3:45 PM"
    elif dte <= 3:
        theta_warning = True
        theta_message = f"Short DTE ({dte}d): significant theta decay — use limit orders"
    else:
        theta_warning = False
        theta_message = ""

    expiry_str      = expiry.strftime("%y%m%d")
    contract_symbol = f"{ticker}{expiry_str}{'C' if c_or_p=='CALL' else 'P'}{int(strike*1000):08d}"

    return {
        "ticker":          ticker,
        "strike":          strike,
        "c_or_p":          c_or_p,
        "expiration":      expiry.strftime("%Y-%m-%d"),
        "dte":             dte,
        "est_premium":     est_premium,
        "est_cost_1x":     round(est_premium * 100, 2),
        "contract_symbol": contract_symbol,
        "note":            "⚠️ Formula estimate — wire live options API for real pricing",
        "current_price":   current_price,
        "moneyness":       moneyness,
        "delta_estimate":  delta_estimate,
        "theta_warning":   theta_warning,
        "theta_message":   theta_message,
        "grade_used":      grade or "none",
    }
