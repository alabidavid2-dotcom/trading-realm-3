# Trading Realm 3.0
### AI-Powered Automated Strategy System | Built for 0DTE + Swing Trading

---

## What It Does

Trading Realm is a quantitative trading dashboard that combines Hidden Markov Model (HMM) regime detection with The Strat candle classification system to generate, score, and execute high-probability trade signals on US equities and options via the Alpaca API.

Core loop: **Market data → Regime detection → Indicator scoring → Strat patterns → Composite signal → Risk-gated execution**

Only **Grade A+ and A** setups pass the final filter. The system is designed to keep you OUT of bad trades as much as it is to find good ones.

---

## File Structure

```
Trading Realm/
│
├── app.py                  # Streamlit dashboard — entry point, run this
├── config.py               # All tunable parameters (tickers, risk, HMM, indicators)
├── requirements.txt        # Python dependencies
├── .env                    # API keys (NEVER commit — in .gitignore)
├── .env.example            # Template for .env setup
├── .gitignore
│
└── lib/                    # Core engine — all backend logic lives here
    ├── __init__.py
    │
    │   ── Data Layer ──
    ├── data_client.py          # Alpaca market data singleton (OHLCV fetcher)
    │
    │   ── Analysis Engine ──
    ├── hmm_regime.py           # HMM regime detection (5 states: Bull/Bear/Chop)
    ├── indicators.py           # RSI, ADX, MACD, Bollinger Bands, ATR, Volume
    ├── strat_classifier.py     # The Strat candle types (1/2/3) + pattern scanner
    ├── signal_engine.py        # THE BRAIN — regime-gated signal scoring (-100→+100)
    │
    │   ── Execution Layer ──
    ├── executor.py             # Alpaca order execution (equity + 0DTE options)
    ├── risk_manager.py         # Position sizing, cooldowns, daily loss limits
    │
    │   ── Strategy & Backtest ──
    ├── backtester.py           # Walk-forward backtest engine (Sharpe, win rate, DD)
    ├── walkforward.py          # Out-of-sample walk-forward analysis
    ├── playbook.py             # Daily orchestrator — full morning analysis pipeline
    ├── trade_grader.py         # Setup grading (A+/A/B/C) with FTFC stack
    │
    │   ── Scanner ──
    ├── scanner.py              # Two-tier S&P 500 scanner (Tier 1: fast, Tier 2: full HMM)
    │
    │   ── Tracking & Alerts ──
    ├── tracker.py              # Watchlist trade tracker (persistent JSON)
    ├── alerts.py               # Price alert system (72-hour TTL)
    ├── pnl_tracker.py          # Persistent P&L stats (win rate, streaks, regime breakdown)
    ├── universe.py             # Refined trading universe manager (PTR scoring)
    ├── notifier.py             # Windows desktop + browser sound notifications
    │
    │   ── Research ──
    └── colab_full_system.py    # Self-contained single-file version for Google Colab
```

> **For AI models:** The entry point is `app.py`. All importable logic is in `lib/`. Configuration lives in root `config.py`. API credentials are loaded from `.env` via `python-dotenv`.

---

## Quick Start

### 1. Environment Setup

```bash
# Clone and install dependencies
pip install -r requirements.txt

# Set up your API credentials
copy .env.example .env
# Then open .env and fill in your Alpaca keys
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

### 3. Google Colab (Research / No API Keys Needed)

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `lib/colab_full_system.py`
3. Run: `!pip install yfinance hmmlearn pandas numpy matplotlib -q`
4. Run all cells

---

## How the Signal Engine Works

Signals are scored on a **-100 to +100 scale** from three components:

| Component | Max Points | Logic |
|---|---|---|
| Regime Score | ±40 | HMM-detected market state gates everything else |
| Indicator Score | ±30 | RSI/ADX/MACD — only regime-relevant rules fire |
| Strat Score | ±20 | Pattern grade × direction alignment with regime |

**Signal thresholds:**

| Score | Direction | Strength | Trade Type |
|---|---|---|---|
| ≥ +50 | LONG | STRONG | 0DTE Call |
| +30 to +49 | LONG | MODERATE | Swing Long |
| ≤ -50 | SHORT | STRONG | 0DTE Put |
| -30 to -49 | SHORT | MODERATE | Swing Short |
| -15 to +15 | FLAT | NO_TRADE | — |

---

## Configuration

All parameters live in `config.py`:

- **Universe:** `CORE_INDICES`, `WATCHLIST`
- **HMM:** `HMM_N_REGIMES`, `HMM_TRAIN_DAYS`, `HMM_COVARIANCE`
- **Indicators:** RSI, ADX, MACD, Bollinger Band periods
- **Risk:** `RISK_BASE_0DTE` ($75), `RISK_BASE_SWING` ($150), regime multipliers
- **Cooldowns:** `COOLDOWN_HOURS_0DTE` (1h), `COOLDOWN_HOURS_SWING` (48h)

---

## Backtest

```python
from lib.hmm_regime import analyze_regime
from lib.backtester import Backtester
from lib.data_client import get_daily

df = get_daily('SPY', days=730)
rd = analyze_regime('SPY', df)
bt = Backtester(rd['df'], ticker='SPY')
metrics = bt.run(hold_bars=5, stop_loss_atr=2.0, take_profit_atr=3.0)
bt.print_report()
```

---

## Disclaimer

Not financial advice. Backtest results do not guarantee live performance. Always paper trade before going live. HMM regimes detect historical patterns — they do not predict the future.
