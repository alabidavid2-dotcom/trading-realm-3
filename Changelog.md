# Changelog

All notable changes to Trading Realm are documented here.

---

## [3.0.0] — 2026-05-08

### Security Update — API Key Management

**Problem:** `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` were hardcoded as fallback defaults inside `config.py`, meaning live credentials were embedded directly in source code and at risk of being committed to version control.

**Changes:**
- Installed `python-dotenv>=1.0.0` and added it to `requirements.txt`
- Created `.env.example` as a safe, committable credential template
- Created `.gitignore` with `.env` explicitly excluded
- Refactored `config.py` to call `load_dotenv()` at startup and read keys via `os.environ.get()` with no hardcoded fallbacks
- Added a startup guard in `config.py` that raises a clear `EnvironmentError` listing any missing keys, so misconfigured environments fail loudly rather than silently

**No breaking changes:** Existing `from config import ALPACA_API_KEY, ALPACA_SECRET_KEY` imports in `executor.py` and `data_client.py` continue to work without modification.

---

### Architecture Refactor — `lib/` Package Structure

**Problem:** All 18+ Python modules were co-located in the root directory alongside `app.py` and `config.py`, making it impossible to distinguish the entry point from the engine, and creating ambiguity for AI-assisted development tools.

**Changes:**
- Created `lib/` as a proper Python package (`lib/__init__.py`)
- Moved all backend logic files into `lib/`:
  - Data layer: `data_client.py`
  - Analysis engine: `hmm_regime.py`, `indicators.py`, `strat_classifier.py`, `signal_engine.py`
  - Execution layer: `executor.py`, `risk_manager.py`
  - Strategy & backtest: `backtester.py`, `walkforward.py`, `playbook.py`, `trade_grader.py`
  - Scanner: `scanner.py`
  - Tracking & alerts: `tracker.py`, `alerts.py`, `pnl_tracker.py`, `universe.py`, `notifier.py`
  - Research: `colab_full_system.py`
- Kept at root: `app.py` (entry point), `config.py` (configuration), utility scripts
- Updated all 31 affected import statements across 12 files to use `lib.module` paths
- Fixed `os.path.dirname(__file__)` references in `pnl_tracker.py`, `tracker.py`, `alerts.py`, and `universe.py` to resolve JSON data files to the root directory rather than `lib/`
- Updated `README.md` with the new structure map and corrected code examples

**Root after refactor:**
```
app.py | config.py | requirements.txt | .env | .env.example | .gitignore
```

**`lib/` after refactor:**
```
18 logic modules + __init__.py
```

---

## [2.x] — Pre-May 2026

- Initial implementation of HMM regime detection with GaussianHMM (5 states)
- The Strat candle classification system (types 1/2/3 + pattern scanner)
- Regime-gated signal engine with composite scoring (-100 to +100)
- Alpaca paper trading integration (equity + 0DTE options)
- Streamlit dashboard with live scanner, tracker, alerts, P&L, and backtest tabs
- Two-tier S&P 500 scanner (Tier 1 fast screen, Tier 2 full HMM)
- Walk-forward backtester with Sharpe, win rate, max drawdown, regime breakdown
