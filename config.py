# ================================================
# CONFIG.PY - System Configuration
# The Strat + HMM Regime Trading System
# Built for 0DTE + Swing Trading
# ================================================

import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# ── .env loading ────────────────────────────────────────────────────────────────
# Resolve path relative to THIS file so it loads correctly regardless of where
# Streamlit is launched from.  override=True beats any stale os.environ values.
_env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(_env_path, override=True)


# ── Key accessor ────────────────────────────────────────────────────────────────
def get_alpaca_keys() -> tuple[str, str]:
    """
    Return (api_key, secret_key) freshly read from the environment and
    stripped of any accidental whitespace or Windows CRLF artifacts.
    Call this instead of reading the module-level variables directly
    so any whitespace in the .env file is always scrubbed.
    """
    key = (os.getenv('ALPACA_API_KEY') or '').strip()
    sec = (os.getenv('ALPACA_SECRET_KEY') or '').strip()
    return key, sec


# ── Alpaca credentials (module-level for legacy imports) ────────────────────────
ALPACA_API_KEY, ALPACA_SECRET_KEY = get_alpaca_keys()

# ── Base URL logic ───────────────────────────────────────────────────────────────
# Alpaca paper keys always start with 'PK'.  Live keys start with 'AK'.
# The BROKER (trading) URL differs between environments.
# The DATA URL is always https://data.alpaca.markets regardless of environment.
ALPACA_PAPER_TRADING = ALPACA_API_KEY.startswith('PK') if ALPACA_API_KEY else True
ALPACA_PAPER         = ALPACA_PAPER_TRADING                # alias used in app.py

ALPACA_BASE_URL = (
    'https://paper-api.alpaca.markets'
    if ALPACA_PAPER_TRADING
    else 'https://api.alpaca.markets'
)
ALPACA_DATA_URL = 'https://data.alpaca.markets'   # never changes

DATA_FEED              = 'iex'   # Free tier: 15-min delayed intraday, real-time daily
ALPACA_DATA_DELAY_MINS = 15

# ── SUPABASE ─────────────────────────────────────────────────────────────────────
SUPABASE_URL = (os.getenv('SUPABASE_URL') or '').strip()
SUPABASE_KEY = (os.getenv('SUPABASE_KEY') or '').strip()

# ── Startup validation ───────────────────────────────────────────────────────────
_required = {
    'ALPACA_API_KEY':    ALPACA_API_KEY,
    'ALPACA_SECRET_KEY': ALPACA_SECRET_KEY,
    'SUPABASE_URL':      SUPABASE_URL,
    'SUPABASE_KEY':      SUPABASE_KEY,
}
_missing = [k for k, v in _required.items() if not v]
if _missing:
    raise EnvironmentError(
        f"Missing required environment variable(s): {', '.join(_missing)}\n"
        f"Check {_env_path} — all four credentials must be set."
    )

# ── UNIVERSE ─────────────────────────────────────────────────────────────────────
CORE_INDICES = ['SPY', 'QQQ', 'IWM']
WATCHLIST    = ['NVDA', 'AAPL', 'TSLA', 'MU', 'WMT', 'UNH', 'ELF', 'GM']
ALL_TICKERS  = CORE_INDICES + WATCHLIST

# ── HMM REGIME SETTINGS ──────────────────────────────────────────────────────────
HMM_N_REGIMES    = 5
HMM_TRAIN_DAYS   = 700
HMM_COVARIANCE   = 'full'
HMM_N_ITER       = 200
HMM_RANDOM_SEED  = 42
HMM_FEATURES     = ['daily_return', 'vol_20', 'volume_ratio', 'range_pct']

REGIME_LABELS = {
    0: "Bear_Volatile",
    1: "Bear_Quiet",
    2: "Chop",
    3: "Bull_Quiet",
    4: "Bull_Volatile",
}

# ── INDICATOR SETTINGS ───────────────────────────────────────────────────────────
RSI_PERIOD          = 14
RSI_OVERBOUGHT      = 70
RSI_OVERSOLD        = 30
ADX_PERIOD          = 14
ADX_TREND_THRESHOLD = 25
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
MOMENTUM_PERIOD     = 10
BB_PERIOD           = 20
BB_STD              = 2.0
VOLUME_MA_PERIOD    = 20

# ── THE STRAT SETTINGS ───────────────────────────────────────────────────────────
STRAT_LOOKBACK   = 5
STRAT_TIMEFRAMES = {
    '0dte':  {'period': '5d',  'interval': '15m'},
    'swing': {'period': '60d', 'interval': '1d'},
}

# ── RISK MANAGEMENT ──────────────────────────────────────────────────────────────
RISK_BASE_0DTE   = 75
RISK_BASE_SWING  = 150
CONTRACTS_MIN    = 2
CONTRACTS_MAX    = 4

REGIME_RISK_MULTIPLIER = {
    "Bull_Quiet":    1.0,
    "Bull_Volatile": 0.75,
    "Bear_Quiet":    1.0,
    "Bear_Volatile": 0.5,
    "Chop":          0.4,
}

COOLDOWN_HOURS_0DTE  = 1
COOLDOWN_HOURS_SWING = 48

# ── BACKTEST SETTINGS ────────────────────────────────────────────────────────────
BACKTEST_START   = '2024-01-01'
BACKTEST_END     = None
INITIAL_CAPITAL  = 10000
COMMISSION_PER_TRADE = 1.50

# ── OUTPUT ───────────────────────────────────────────────────────────────────────
REPORT_DATE = datetime.now().strftime("%Y-%m-%d")

# ── SUPABASE CLIENT ──────────────────────────────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
