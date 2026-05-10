"""
db.py — Supabase client singleton for Trading Realm.

Usage anywhere in lib/:
    from lib.db import get_db
    res = get_db().table('pnl_history').select('*').execute()
"""
from __future__ import annotations
from functools import lru_cache
from datetime import date as _date
from supabase import create_client, Client


@lru_cache(maxsize=1)
def get_db() -> Client:
    """Return the shared Supabase client, initialised on first call."""
    from config import SUPABASE_URL, SUPABASE_KEY
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_recent_trades(
    table: str = "pnl_history",
    limit: int = 20,
    date_col: str | None = "date",
    order_col: str = "timestamp",
) -> tuple[list[dict], str]:
    """
    Return (rows, label) from *table*, newest-first.

    Market-closed / weekend fallback:
      1. If date_col is set, try rows where date_col == today.
      2. If that returns nothing (holiday, weekend, empty table),
         fall back to the most recent rows regardless of date.
      3. label is 'today', 'historical (YYYY-MM-DD)', or 'no data'.

    Never raises — callers can always unpack safely.
    """
    db = get_db()

    if date_col:
        today = _date.today().isoformat()
        try:
            res = (
                db.table(table)
                .select("*")
                .eq(date_col, today)
                .order(order_col, desc=True)
                .limit(limit)
                .execute()
            )
            if res.data:
                return res.data, "today"
        except Exception:
            pass  # fall through to historical fetch

    # Fallback: most recent rows regardless of date
    try:
        res = (
            db.table(table)
            .select("*")
            .order(order_col, desc=True)
            .limit(limit)
            .execute()
        )
        rows = res.data or []
    except Exception:
        return [], "no data"

    if not rows:
        return [], "no data"

    # Label with the date of the most recent row
    sample = rows[0].get(date_col or order_col, "") if date_col else rows[0].get(order_col, "")
    date_str = str(sample)[:10] if sample else ""
    label = f"historical ({date_str})" if date_str else "historical"
    return rows, label
