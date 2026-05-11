# ================================================
# KILL_SWITCH.PY — EOD 0DTE Liquidation Engine
#
# Entry Lock : 15:30 ET — block new 0DTE buy orders
# Hard Close : 15:50 ET — liquidate all 0DTE positions
# Exception  : swing-flagged equity tickers are skipped (hold overnight)
#
# Uses America/New_York so DST (EST ↔ EDT) is handled automatically.
# ================================================

from __future__ import annotations
from datetime import datetime, time as _time
from zoneinfo import ZoneInfo

_NY = ZoneInfo("America/New_York")

# ── Time thresholds ──────────────────────────────────────────────────────────────
_ENTRY_LOCK  = _time(15, 30)   # 3:30 PM ET — no new 0DTE entries
_HARD_CLOSE  = _time(15, 50)   # 3:50 PM ET — liquidate all 0DTE
_MARKET_OPEN = _time( 9, 30)   # 9:30 AM ET


def _now_et() -> datetime:
    """Current datetime in America/New_York (handles EST/EDT automatically)."""
    return datetime.now(_NY)


def _time_et() -> _time:
    t = _now_et().time()
    return t.replace(tzinfo=None)


# ── Public state queries ─────────────────────────────────────────────────────────

def is_entry_locked() -> bool:
    """True from 15:30 ET — block all new 0DTE buy/long orders."""
    return _time_et() >= _ENTRY_LOCK


def is_hard_close_time() -> bool:
    """True from 15:50 ET — trigger 0DTE liquidation sweep."""
    return _time_et() >= _HARD_CLOSE


def is_market_hours() -> bool:
    """True during regular session 09:30–16:00 ET."""
    t = _time_et()
    return _MARKET_OPEN <= t < _time(16, 0)


def get_eod_status() -> dict:
    """
    Snapshot dict consumed by app.py on every rerun.

    Keys
    ----
    now_et        : current time string e.g. '15:47:22 ET'
    today         : ISO date string for dedup ('2025-03-14')
    entry_locked  : True after 15:30 ET
    hard_close    : True after 15:50 ET
    market_hours  : True during 09:30–16:00 ET
    """
    t = _now_et()
    return {
        'now_et':       t.strftime('%H:%M:%S ET'),
        'today':        t.date().isoformat(),
        'entry_locked': is_entry_locked(),
        'hard_close':   is_hard_close_time(),
        'market_hours': is_market_hours(),
    }


# ── Targeted liquidation ─────────────────────────────────────────────────────────

def close_0dte_positions(executor, swing_tickers: list | None = None) -> dict:
    """
    Close open options (0DTE) positions while preserving ALL equity positions.

    Safety-first rules
    ------------------
    - asset_class contains 'option' → close (these are 0DTE contracts).
    - asset_class is equity (anything without 'option') → ALWAYS skip.
      Equity held intraday is still safer to skip than to accidentally close
      a manual swing position. Options are the only thing that MUST expire today.
    - swing_tickers is kept as an extra explicit safeguard but is no longer
      the sole protection for equity — even untagged equity is preserved.

    Parameters
    ----------
    executor      : Executor instance (from lib/executor.py)
    swing_tickers : list of tickers flagged SWING in the session order log

    Returns
    -------
    {'closed': [...], 'skipped': [...], 'errors': [...]}
    """
    swing_set = {t.upper() for t in (swing_tickers or [])}
    closed: list[str] = []
    skipped: list[str] = []
    errors:  list[str] = []

    if executor is None:
        return {'closed': [], 'skipped': [], 'errors': ['No executor available']}

    try:
        positions = executor.get_positions()
    except Exception as exc:
        return {'closed': [], 'skipped': [], 'errors': [f'get_positions failed: {exc}']}

    for pos in positions:
        sym         = str(pos.symbol).upper()
        asset_class = str(getattr(pos, 'asset_class', '')).lower()
        is_option   = 'option' in asset_class

        # Equity: ALWAYS skip — never risk closing a manual or swing equity hold
        if not is_option:
            skipped.append(sym)
            continue

        try:
            result = executor.close_position(sym)
            status = result.get('status', '')
            if status in ('submitted', 'accepted', 'DRY_RUN'):
                closed.append(sym)
            else:
                errors.append(f"{sym}: {result.get('error', status)}")
        except Exception as exc:
            errors.append(f"{sym}: {exc}")

    return {'closed': closed, 'skipped': skipped, 'errors': errors}
