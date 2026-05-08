"""
pnl_tracker.py — Persistent P&L Statistics
Survives restarts. Tracks all time periods: Today / Week / Month / All Time.
Stats: win rate, avg win/loss, profit factor, regime breakdown,
       best/worst ticker, time-of-day, streak tracker.
"""

from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict
from config import supabase

ET = ZoneInfo("America/New_York")


# ── Load / Clear ───────────────────────────────────────────────────────────────

def load_pnl_history() -> list[dict]:
    try:
        res = supabase.table('pnl_history').select('*').order('timestamp', desc=True).execute()
        return res.data or []
    except Exception:
        return []


def clear_pnl_history():
    try:
        supabase.table('pnl_history').delete().gte('timestamp', '2000-01-01T00:00:00+00:00').execute()
    except Exception:
        pass


# ── Log a trade result ─────────────────────────────────────────────────────────

def log_trade_result(
    ticker: str,
    direction: str,
    trade_type: str,
    grade: str,
    regime: str,
    entry_price: float,
    exit_price: float,
    qty: int = 1,
    note: str = "",
):
    """
    Call this when a trade closes.
    Calculates P&L and inserts directly to Supabase.
    """
    now = datetime.now(ET)

    if direction == "LONG":
        pnl_pct     = (exit_price - entry_price) / entry_price * 100
        pnl_dollars = (exit_price - entry_price) * qty
    else:
        pnl_pct     = (entry_price - exit_price) / entry_price * 100
        pnl_dollars = (entry_price - exit_price) * qty

    record = {
        "ticker":       ticker,
        "direction":    direction,
        "trade_type":   trade_type,
        "grade":        grade,
        "regime":       regime,
        "entry_price":  entry_price,
        "exit_price":   exit_price,
        "qty":          qty,
        "pnl_pct":      round(pnl_pct, 3),
        "pnl_dollars":  round(pnl_dollars, 2),
        "win":          pnl_pct > 0,
        "note":         note,
        "timestamp":    now.isoformat(),
        "date":         now.date().isoformat(),
        "hour":         now.hour,
        "day_of_week":  now.strftime("%A"),
    }

    try:
        supabase.table('pnl_history').insert(record).execute()
    except Exception:
        pass

    return record


# ── Filter by period ───────────────────────────────────────────────────────────

def filter_by_period(history: list[dict], period: str) -> list[dict]:
    """
    period: "today" | "week" | "month" | "all"
    """
    if period == "all" or not history:
        return history

    now   = datetime.now(ET)
    today = now.date()

    if period == "today":
        cutoff = today
        return [h for h in history if h.get("date") == str(cutoff)]

    elif period == "week":
        start_of_week = today - timedelta(days=today.weekday())
        return [
            h for h in history
            if h.get("date") and date.fromisoformat(h["date"]) >= start_of_week
        ]

    elif period == "month":
        return [
            h for h in history
            if h.get("date") and h["date"].startswith(today.strftime("%Y-%m"))
        ]

    return history


# ── Calculate stats ────────────────────────────────────────────────────────────

def calculate_stats(trades: list[dict]) -> dict:
    """
    Calculate all performance stats from a list of trade records.
    Returns a comprehensive stats dict.
    """
    if not trades:
        return _empty_stats()

    wins   = [t for t in trades if t.get("win")]
    losses = [t for t in trades if not t.get("win")]

    total       = len(trades)
    win_count   = len(wins)
    loss_count  = len(losses)
    win_rate    = round(win_count / total * 100, 1) if total else 0

    avg_win     = round(sum(float(t["pnl_pct"]) for t in wins)   / len(wins),   3) if wins   else 0
    avg_loss    = round(sum(float(t["pnl_pct"]) for t in losses) / len(losses), 3) if losses else 0

    gross_wins   = sum(float(t["pnl_pct"]) for t in wins)
    gross_losses = abs(sum(float(t["pnl_pct"]) for t in losses))
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else 999

    total_pnl_pct     = round(sum(float(t["pnl_pct"])     for t in trades), 2)
    total_pnl_dollars = round(sum(float(t["pnl_dollars"]) for t in trades), 2)

    # ── Streak ──────────────────────────────────────────────────────────────
    current_streak = 0
    streak_type    = None
    for t in trades:  # trades[0] is most recent
        w = t.get("win")
        if streak_type is None:
            streak_type    = "win" if w else "loss"
            current_streak = 1
        elif (streak_type == "win" and w) or (streak_type == "loss" and not w):
            current_streak += 1
        else:
            break

    max_win_streak  = _max_streak(trades, win=True)
    max_loss_streak = _max_streak(trades, win=False)

    # ── Regime breakdown ─────────────────────────────────────────────────────
    regime_stats = defaultdict(lambda: {"count": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        r = t.get("regime", "Unknown")
        regime_stats[r]["count"] += 1
        regime_stats[r]["pnl"]   += float(t["pnl_pct"])
        if t.get("win"):
            regime_stats[r]["wins"] += 1
    regime_breakdown = {}
    for r, s in regime_stats.items():
        regime_breakdown[r] = {
            "count":    s["count"],
            "win_rate": round(s["wins"] / s["count"] * 100, 1) if s["count"] else 0,
            "avg_pnl":  round(s["pnl"] / s["count"], 3) if s["count"] else 0,
        }

    # ── Best / worst ticker ──────────────────────────────────────────────────
    ticker_stats = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        sym = t.get("ticker", "?")
        ticker_stats[sym]["count"] += 1
        ticker_stats[sym]["pnl"]   += float(t["pnl_pct"])
        if t.get("win"):
            ticker_stats[sym]["wins"] += 1

    by_pnl = sorted(ticker_stats.items(), key=lambda x: x[1]["pnl"], reverse=True)
    if by_pnl:
        best_ticker      = by_pnl[0][0]
        worst_ticker     = by_pnl[-1][0]
        best_ticker_pnl  = round(by_pnl[0][1]["pnl"],  2)
        worst_ticker_pnl = round(by_pnl[-1][1]["pnl"], 2)
    else:
        best_ticker = worst_ticker = "—"
        best_ticker_pnl = worst_ticker_pnl = 0

    # ── Best / worst time of day ─────────────────────────────────────────────
    hour_stats = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for t in trades:
        h = t.get("hour", 0)
        hour_stats[h]["count"] += 1
        hour_stats[h]["pnl"]   += float(t["pnl_pct"])
    by_hour = sorted(hour_stats.items(), key=lambda x: x[1]["pnl"] / max(x[1]["count"], 1), reverse=True)
    best_hour  = f"{by_hour[0][0]}:00"  if by_hour else "—"
    worst_hour = f"{by_hour[-1][0]}:00" if by_hour else "—"

    return {
        "total_trades":       total,
        "win_count":          win_count,
        "loss_count":         loss_count,
        "win_rate":           win_rate,
        "avg_win":            avg_win,
        "avg_loss":           avg_loss,
        "profit_factor":      profit_factor,
        "total_pnl_pct":      total_pnl_pct,
        "total_pnl_dollars":  total_pnl_dollars,
        "current_streak":     current_streak,
        "streak_type":        streak_type,
        "max_win_streak":     max_win_streak,
        "max_loss_streak":    max_loss_streak,
        "regime_breakdown":   regime_breakdown,
        "best_ticker":        best_ticker,
        "best_ticker_pnl":    best_ticker_pnl,
        "worst_ticker":       worst_ticker,
        "worst_ticker_pnl":   worst_ticker_pnl,
        "best_hour":          best_hour,
        "worst_hour":         worst_hour,
        "ticker_stats":       dict(ticker_stats),
    }


def _empty_stats() -> dict:
    return {
        "total_trades": 0, "win_count": 0, "loss_count": 0,
        "win_rate": 0, "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
        "total_pnl_pct": 0, "total_pnl_dollars": 0,
        "current_streak": 0, "streak_type": None,
        "max_win_streak": 0, "max_loss_streak": 0,
        "regime_breakdown": {}, "best_ticker": "—", "best_ticker_pnl": 0,
        "worst_ticker": "—", "worst_ticker_pnl": 0,
        "best_hour": "—", "worst_hour": "—", "ticker_stats": {},
    }


def _max_streak(trades: list[dict], win: bool) -> int:
    max_s = 0; cur = 0
    for t in reversed(trades):
        if t.get("win") == win:
            cur += 1; max_s = max(max_s, cur)
        else:
            cur = 0
    return max_s


def get_stats_for_period(period: str) -> dict:
    """Convenience: load history, filter, and calculate stats in one call."""
    history = load_pnl_history()
    filtered = filter_by_period(history, period)
    return calculate_stats(filtered)
