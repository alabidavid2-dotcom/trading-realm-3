# ================================================
# CONFIG.PY - System Configuration
# The Strat + HMM Regime Trading System
# Built for 0DTE + Swing Trading
# ================================================

import os
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# --- UNIVERSE ---
CORE_INDICES = ['SPY', 'QQQ', 'IWM']
WATCHLIST = ['NVDA', 'AAPL', 'TSLA', 'MU', 'WMT', 'UNH', 'ELF', 'GM']
ALL_TICKERS = CORE_INDICES + WATCHLIST

# --- HMM REGIME SETTINGS ---
HMM_N_REGIMES = 5           # Number of hidden states (bull_quiet, bull_vol, bear_quiet, bear_vol, chop)
HMM_TRAIN_DAYS = 700        # Training lookback in calendar days
HMM_COVARIANCE = 'full'     # 'full', 'diag', 'tied', 'spherical'
HMM_N_ITER = 200            # EM algorithm iterations
HMM_RANDOM_SEED = 42
HMM_FEATURES = ['daily_return', 'vol_20', 'volume_ratio', 'range_pct']  # Features fed to HMM

# --- REGIME LABELS (mapped after training by sorting on mean return) ---
REGIME_LABELS = {
    0: "Bear_Volatile",
    1: "Bear_Quiet",
    2: "Chop",
    3: "Bull_Quiet",
    4: "Bull_Volatile",
}

# --- INDICATOR SETTINGS ---
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25    # ADX > 25 = trending
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MOMENTUM_PERIOD = 10
BB_PERIOD = 20
BB_STD = 2.0
VOLUME_MA_PERIOD = 20

# --- THE STRAT SETTINGS ---
STRAT_LOOKBACK = 5          # Candles to scan for patterns
STRAT_TIMEFRAMES = {
    '0dte': {'period': '5d', 'interval': '15m'},
    'swing': {'period': '60d', 'interval': '1d'},
}

# --- RISK MANAGEMENT ---
RISK_BASE_0DTE = 75         # Base risk per 0DTE trade
RISK_BASE_SWING = 150       # Base risk per swing trade
CONTRACTS_MIN = 2
CONTRACTS_MAX = 4

# Regime-based risk multipliers
REGIME_RISK_MULTIPLIER = {
    "Bull_Quiet":    1.0,    # Full size
    "Bull_Volatile": 0.75,   # Slightly reduced
    "Bear_Quiet":    1.0,    # Full size (puts)
    "Bear_Volatile": 0.5,    # Reduced - high vol = wider stops
    "Chop":          0.4,    # Heavy reduction
}

# Cooldown: hours to wait after an exit before re-entering
COOLDOWN_HOURS_0DTE = 1
COOLDOWN_HOURS_SWING = 48

# --- BACKTEST SETTINGS ---
BACKTEST_START = '2024-01-01'
BACKTEST_END = None          # None = today
INITIAL_CAPITAL = 10000
COMMISSION_PER_TRADE = 1.50  # Per contract

# --- ALPACA API ---
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')

# --- SUPABASE ---
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

_required = {
    'ALPACA_API_KEY':   ALPACA_API_KEY,
    'ALPACA_SECRET_KEY': ALPACA_SECRET_KEY,
    'SUPABASE_URL':     SUPABASE_URL,
    'SUPABASE_KEY':     SUPABASE_KEY,
}
_missing = [k for k, v in _required.items() if not v]
if _missing:
    raise EnvironmentError(
        f"Missing required environment variable(s): {', '.join(_missing)}\n"
        "Copy .env.example to .env and fill in all credentials."
    )
ALPACA_PAPER_TRADING = True
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
ALPACA_DATA_URL = 'https://data.alpaca.markets'
DATA_FEED = 'iex'              # Free tier (15-min delayed intraday); use 'sip' for real-time
ALPACA_DATA_DELAY_MINS = 15    # IEX intraday delay in minutes

# --- OUTPUT ---
REPORT_DATE = datetime.now().strftime("%Y-%m-%d")

# --- SUPABASE CLIENT ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
