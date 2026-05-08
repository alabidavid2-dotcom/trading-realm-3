"""
db.py — Supabase client singleton for Trading Realm.

Usage anywhere in lib/:
    from lib.db import get_db
    res = get_db().table('pnl_history').select('*').execute()
"""
from functools import lru_cache
from supabase import create_client, Client


@lru_cache(maxsize=1)
def get_db() -> Client:
    """Return the shared Supabase client, initialised on first call."""
    from config import SUPABASE_URL, SUPABASE_KEY
    return create_client(SUPABASE_URL, SUPABASE_KEY)
