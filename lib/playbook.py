# ================================================
# PLAYBOOK.PY - Daily Pre-Market Playbook Generator
# Orchestrates all modules into an actionable morning report
# ================================================

import pandas as pd
import numpy as np
from datetime import datetime

from config import (
    CORE_INDICES, WATCHLIST, REPORT_DATE,
    STRAT_TIMEFRAMES, HMM_TRAIN_DAYS
)
from lib.hmm_regime import analyze_regime
from lib.indicators import add_all_indicators, indicator_snapshot
from lib.strat_classifier import strat_analysis, ftfc_check
from lib.signal_engine import generate_signal, filter_grade_a_only
from lib.risk_manager import RiskManager
from lib.backtester import Backtester


def fetch_data(ticker, days=HMM_TRAIN_DAYS, interval='1d'):
    """Fetch OHLCV data from Alpaca."""
    from lib.data_client import get_daily, get_weekly, get_15min
    if interval == '1wk':
        df = get_weekly(ticker, days=days)
    elif interval == '15m':
        df = get_15min(ticker, days=min(days, 5))
    else:
        df = get_daily(ticker, days=days)
    if df.empty:
        print(f"  ⚠️  No data for {ticker}")
    return df


def analyze_ticker(ticker, run_backtest=False):
    """
    Full analysis pipeline for a single ticker.
    Returns a comprehensive analysis dict.
    """
    print(f"  Analyzing {ticker}...")

    # 1. Fetch daily data
    df_daily = fetch_data(ticker, days=HMM_TRAIN_DAYS, interval='1d')
    if df_daily.empty:
        return None

    # 2. HMM Regime Detection
    regime_data = analyze_regime(ticker, df_daily)

    # 3. Technical Indicators
    df_ind = add_all_indicators(regime_data['df'])
    ind_snap = indicator_snapshot(df_ind)

    # 4. The Strat - Daily
    daily_strat = strat_analysis(df_ind, "Daily")

    # 5. The Strat - 15m (intraday)
    df_15m = fetch_data(ticker, days=5, interval='15m')
    strat_15m = None
    if not df_15m.empty:
        strat_15m = strat_analysis(df_15m, "15m")

    # 6. Generate Signal
    signal = generate_signal(
        regime=regime_data['regime'],
        ind_snapshot=ind_snap,
        strat_patterns=daily_strat['patterns'],
        regime_probs=regime_data['regime_probabilities'],
    )

    # 7. FTFC Check — include weekly direction when available
    daily_dir = daily_strat['last_dir']
    weekly_dir = None
    try:
        df_weekly = fetch_data(ticker, days=365, interval='1wk')
        if not df_weekly.empty and len(df_weekly) > 1:
            last_wk = df_weekly.iloc[-1]
            weekly_dir = 'up' if last_wk['Close'] > last_wk['Open'] else 'down'
    except Exception:
        pass
    ftfc = ftfc_check(daily_dir, weekly_dir=weekly_dir)

    # 8. Backtest (optional - slow)
    bt_metrics = None
    if run_backtest:
        bt = Backtester(regime_data['df'], ticker=ticker)
        bt_metrics = bt.run()

    return {
        'ticker': ticker,
        'regime': regime_data,
        'indicators': ind_snap,
        'strat_daily': daily_strat,
        'strat_15m': strat_15m,
        'signal': signal,
        'ftfc': ftfc,
        'backtest': bt_metrics,
    }


def generate_playbook(run_backtest=False):
    """
    Generate the full morning playbook across all tickers.
    """
    print(f"🚀 Generating Playbook for {REPORT_DATE}\n")
    risk_mgr = RiskManager()
    results = {}

    # Analyze core indices first (regime context)
    print("--- Core Indices ---")
    for t in CORE_INDICES:
        result = analyze_ticker(t, run_backtest=run_backtest)
        if result:
            results[t] = result

    # Watchlist
    print("\n--- Watchlist ---")
    for t in WATCHLIST:
        result = analyze_ticker(t, run_backtest=False)  # Skip backtest on watchlist for speed
        if result:
            results[t] = result

    # Build the report
    print_playbook(results, risk_mgr)
    return results


def print_playbook(results, risk_mgr):
    """
    Print the formatted daily playbook.
    """
    spy = results.get('SPY')
    if not spy:
        print("❌ No SPY data. Cannot generate playbook.")
        return

    spy_regime = spy['regime']['regime']
    spy_vol = spy['regime']['vol_20']
    spy_probs = spy['regime'].get('regime_probabilities', {})

    print(f"\n{'='*65}")
    print(f"  📅 THE STRAT + HMM REGIME PLAYBOOK — {REPORT_DATE}")
    print(f"{'='*65}")
    print(f"\n  🏛️  SPY REGIME: {spy_regime} | Volatility: {spy_vol}%")

    if spy_probs:
        print(f"  📊 Regime Probabilities:")
        for regime, prob in sorted(spy_probs.items(), key=lambda x: -x[1]):
            bar = '█' * int(prob / 5)
            print(f"      {regime:<18} {prob:>5.1f}%  {bar}")

    # Risk sizing
    sizing = risk_mgr.calculate_position_size(spy['signal'])
    print(f"\n  💰 Risk Today: ${sizing['risk_amount']} per trade | "
          f"Regime mult: {sizing['regime_multiplier']}x")
    print(f"  📦 Contracts: {sizing['contracts']} | Type: {sizing['trade_type']}")

    # Core indices
    print(f"\n{'─'*65}")
    print(f"  CORE INDICES")
    print(f"{'─'*65}")

    for t in CORE_INDICES:
        r = results.get(t)
        if not r:
            continue

        sig = r['signal']
        grade = '✅' if filter_grade_a_only(sig) else '⬜'
        arrow = '🟢' if sig['direction'] == 'LONG' else ('🔴' if sig['direction'] == 'SHORT' else '⚪')

        print(f"\n  {arrow} {t} @ ${r['regime']['close']}")
        print(f"     Regime: {r['regime']['regime']} | FTFC: {r['ftfc']}")
        print(f"     Signal: {sig['direction']} {sig['strength']} (score: {sig['composite_score']:+d}) {grade}")
        print(f"     RSI: {r['indicators']['rsi']} | ADX: {r['indicators']['adx']} | "
              f"MACD hist: {r['indicators']['macd_hist']}")

        # Strat patterns
        for p in r['strat_daily']['patterns']:
            print(f"     🎯 STRAT: {p['name']} [{p['grade']}] → {p['direction']} - {p['description']}")

        if not r['strat_daily']['patterns']:
            print(f"     Strat: No active patterns (last type: {r['strat_daily']['last_type']})")

        # Signal reasoning
        for reason in sig['reasoning']:
            print(f"        → {reason}")

    # Watchlist highlights (only show tradeable)
    print(f"\n{'─'*65}")
    print(f"  WATCHLIST — Actionable Only")
    print(f"{'─'*65}")

    actionable = 0
    for t in WATCHLIST:
        r = results.get(t)
        if not r:
            continue
        sig = r['signal']
        if filter_grade_a_only(sig):
            actionable += 1
            arrow = '🟢' if sig['direction'] == 'LONG' else '🔴'
            print(f"\n  {arrow} {t} @ ${r['regime']['close']}")
            print(f"     {sig['direction']} {sig['strength']} (score: {sig['composite_score']:+d})")
            print(f"     Regime: {r['regime']['regime']} | RSI: {r['indicators']['rsi']}")
            for p in r['strat_daily']['patterns']:
                print(f"     🎯 {p['name']} [{p['grade']}] → {p['direction']}")

    if actionable == 0:
        print("\n  No Grade A+ or A setups in watchlist today. Sit on hands. 🧘")

    # Backtest results (if run)
    spy_bt = spy.get('backtest')
    if spy_bt and 'error' not in spy_bt:
        print(f"\n{'─'*65}")
        print(f"  BACKTEST SUMMARY (SPY)")
        print(f"{'─'*65}")
        print(f"  Win Rate: {spy_bt['win_rate']}% | Trades: {spy_bt['total_trades']}")
        print(f"  Sharpe: {spy_bt['sharpe_ratio']} | Profit Factor: {spy_bt['profit_factor']}")
        print(f"  Return: {spy_bt['total_return_pct']}% | Max DD: {spy_bt['max_drawdown_pct']}%")

    print(f"\n{'='*65}")
    print(f"  Execute only Grade A+ / A setups. Respect the regime.")
    print(f"  Cooldown after every exit. No revenge trades.")
    print(f"{'='*65}\n")


# --------------------------------------------------
# Run directly
# --------------------------------------------------
if __name__ == "__main__":
    results = generate_playbook(run_backtest=False)
