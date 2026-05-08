# ================================================
# APP.PY - Trading Dashboard v3
# Single-ticker + Scanner + Backtest + Live Trading
# Alpaca Paper/Live Execution Engine Integrated
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

from lib.scanner import (
    run_full_scan, get_sp500_tickers,
    load_scan_history, save_scan_history, merge_scan_results,
)
from lib.notifier import (
    send_desktop_notification, format_scan_notification,
    get_alert_sound_html, get_next_scan_time, should_auto_scan, SCAN_TIMES,
)

# ── New modules ────────────────────────────────────────────────────────────────
try:
    from lib.tracker import (
        load_tracked, save_tracked, add_tracked, remove_tracked,
        clear_all_tracked, update_tracked_prices, estimate_option_contract,
    )
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False

try:
    from lib.alerts import (
        load_alerts, save_alerts, create_alert, acknowledge_alert,
        mark_banner_shown, get_pending_banner_alerts, clear_all_alerts,
        get_time_remaining, get_alert_banner_html,
    )
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False

try:
    from lib.universe import (
        load_universe, save_universe, add_to_universe, remove_from_universe,
        get_ranked_universe, auto_build_universe, update_universe_from_scan,
        SECTOR_GROUPS, TICKER_TO_SECTOR,
    )
    UNIVERSE_AVAILABLE = True
except ImportError:
    UNIVERSE_AVAILABLE = False

try:
    from lib.pnl_tracker import (
        load_pnl_history, log_trade_result, get_stats_for_period,
        clear_pnl_history,
    )
    PNL_AVAILABLE = True
except ImportError:
    PNL_AVAILABLE = False

# ── Alpaca Executor (graceful fallback if keys not set) ────────────────────────
EXECUTOR_AVAILABLE = False
try:
    from lib.executor import AlpacaExecutor, execute_signal, start_kill_switch_scheduler
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
    EXECUTOR_AVAILABLE = True
except Exception:
    ALPACA_API_KEY = ALPACA_SECRET_KEY = None
    ALPACA_PAPER = True

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Realm",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');

    /* ── Base ── */
    .stApp {
        background-color: #07080d;
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99,102,241,0.08) 0%, transparent 60%),
            radial-gradient(ellipse 40% 30% at 80% 80%, rgba(34,197,94,0.04) 0%, transparent 50%);
    }

    /* ── Glow utility ── */
    .glow-green { box-shadow: 0 0 24px rgba(34,197,94,0.15), 0 0 2px rgba(34,197,94,0.3); }
    .glow-red   { box-shadow: 0 0 24px rgba(239,68,68,0.15),  0 0 2px rgba(239,68,68,0.3); }
    .glow-blue  { box-shadow: 0 0 24px rgba(99,102,241,0.15), 0 0 2px rgba(99,102,241,0.3); }

    /* ── Page title bar ── */
    .page-title-bar {
        display: flex; align-items: baseline; gap: 16px;
        margin-bottom: 6px;
    }
    .page-title {
        font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 700;
        color: #e2e8f0; letter-spacing: -0.3px;
    }
    .how-to-link {
        font-family: 'DM Sans', sans-serif; font-size: 11px; font-weight: 500;
        color: #4b5563; text-decoration: none; cursor: pointer;
        border-bottom: 1px dashed #374151; padding-bottom: 1px;
        transition: color 0.2s;
    }
    .how-to-link:hover { color: #6366f1; border-color: #6366f1; }

    /* ── Cards ── */
    .signal-card {
        border-radius: 16px; padding: 26px; text-align: center;
        border: 1px solid rgba(255,255,255,0.07);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .signal-card:hover { transform: translateY(-2px); }
    .signal-long  {
        background: linear-gradient(145deg, #0d3320 0%, #071a12 100%);
        border-color: rgba(34,197,94,0.4);
        box-shadow: 0 0 32px rgba(34,197,94,0.08), inset 0 1px 0 rgba(34,197,94,0.1);
    }
    .signal-short {
        background: linear-gradient(145deg, #3b0f1e 0%, #1c0710 100%);
        border-color: rgba(239,68,68,0.4);
        box-shadow: 0 0 32px rgba(239,68,68,0.08), inset 0 1px 0 rgba(239,68,68,0.1);
    }
    .signal-flat  {
        background: linear-gradient(145deg, #16182e 0%, #0f1020 100%);
        border-color: rgba(99,102,241,0.3);
        box-shadow: 0 0 32px rgba(99,102,241,0.06), inset 0 1px 0 rgba(99,102,241,0.08);
    }
    .signal-label {
        font-family: 'DM Sans', sans-serif; font-size: 10px; font-weight: 600;
        color: #4b5563; text-transform: uppercase; letter-spacing: 2.5px; margin-bottom: 10px;
    }
    .signal-value { font-family: 'DM Mono', monospace; font-size: 32px; font-weight: 500; margin: 4px 0; letter-spacing: -1px; }
    .signal-long  .signal-value { color: #4ade80; text-shadow: 0 0 20px rgba(74,222,128,0.4); }
    .signal-short .signal-value { color: #f87171; text-shadow: 0 0 20px rgba(248,113,113,0.4); }
    .signal-flat  .signal-value { color: #818cf8; }
    .signal-sub { font-family: 'DM Sans', sans-serif; font-size: 12px; color: #4b5563; margin-top: 6px; }

    /* ── Regime badges ── */
    .regime-badge {
        display: inline-block; padding: 5px 16px; border-radius: 100px;
        font-family: 'DM Mono', monospace; font-size: 12px; font-weight: 500;
        letter-spacing: 0.5px;
    }
    .regime-bull_quiet    { background: rgba(22,101,52,0.6);  color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
    .regime-bull_volatile { background: rgba(20,83,45,0.6);   color: #86efac; border: 1px solid rgba(134,239,172,0.3); }
    .regime-bear_quiet    { background: rgba(127,29,29,0.6);  color: #fca5a5; border: 1px solid rgba(252,165,165,0.3); }
    .regime-bear_volatile { background: rgba(153,27,27,0.6);  color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
    .regime-chop          { background: rgba(55,65,81,0.6);   color: #9ca3af; border: 1px solid rgba(156,163,175,0.2); }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(145deg, #0f1018 0%, #0b0c14 100%);
        border: 1px solid rgba(255,255,255,0.05); border-radius: 14px;
        padding: 20px; text-align: center;
        transition: border-color 0.2s, transform 0.2s;
    }
    .metric-card:hover { border-color: rgba(99,102,241,0.2); transform: translateY(-1px); }
    .metric-label {
        font-family: 'DM Sans', sans-serif; font-size: 10px; font-weight: 600;
        color: #4b5563; text-transform: uppercase; letter-spacing: 2px;
    }
    .metric-value {
        font-family: 'DM Mono', monospace; font-size: 24px;
        font-weight: 500; color: #e2e8f0; margin: 8px 0 2px 0; letter-spacing: -0.5px;
    }
    .metric-positive { color: #4ade80 !important; }
    .metric-negative { color: #f87171 !important; }
    .metric-sub { font-family: 'DM Sans', sans-serif; font-size: 11px; color: #374151; }

    /* ── Grade pills ── */
    .strat-pill {
        display: inline-block; padding: 4px 14px; border-radius: 100px;
        font-family: 'DM Mono', monospace; font-size: 11px;
        font-weight: 500; margin: 2px 4px; letter-spacing: 0.3px;
    }
    .grade-aplus { background: rgba(22,101,52,0.5);  color: #4ade80; border: 1px solid rgba(74,222,128,0.4); }
    .grade-a     { background: rgba(30,58,95,0.5);   color: #93c5fd; border: 1px solid rgba(147,197,253,0.4); }
    .grade-bplus { background: rgba(59,50,22,0.5);   color: #fcd34d; border: 1px solid rgba(252,211,77,0.4); }
    .grade-b     { background: rgba(45,45,45,0.5);   color: #a1a1aa; border: 1px solid rgba(161,161,170,0.3); }

    /* ── Watchlist rows ── */
    .watchlist-row {
        display: flex; align-items: center; padding: 14px 18px;
        border-radius: 12px; margin: 4px 0;
        font-family: 'DM Mono', monospace; font-size: 13px;
    }
    .watchlist-current  {
        background: linear-gradient(135deg,#111827,#0f172a);
        border: 1px solid rgba(99,102,241,0.15);
        box-shadow: 0 1px 0 rgba(99,102,241,0.05);
    }
    .watchlist-previous { background: rgba(15,18,30,0.4); border:1px solid rgba(255,255,255,0.03); opacity:0.45; }

    /* ── Section headers ── */
    .section-header {
        font-family: 'DM Sans', sans-serif; font-size: 10px; font-weight: 600;
        color: #374151; text-transform: uppercase; letter-spacing: 2.5px;
        padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.04);
        margin-bottom: 16px;
    }

    /* ── Account bar ── */
    .account-bar {
        background: linear-gradient(135deg, #0c0e18 0%, #0f1120 100%);
        border: 1px solid rgba(99,102,241,0.12); border-radius: 16px;
        padding: 16px 28px; margin-bottom: 24px;
        display: flex; align-items: center; gap: 36px;
        box-shadow: 0 0 40px rgba(99,102,241,0.04);
    }
    .account-bar-item { text-align: center; }
    .account-bar-label { font-size: 9px; color: #374151; text-transform: uppercase; letter-spacing: 2px; font-family: 'DM Sans', sans-serif; margin-bottom: 4px; }
    .account-bar-value { font-family: 'DM Mono', monospace; font-size: 17px; font-weight: 500; color: #e2e8f0; }

    /* ── Position rows ── */
    .position-row {
        background: linear-gradient(135deg, #0f1018, #0b0c14);
        border: 1px solid rgba(255,255,255,0.05); border-radius: 14px;
        padding: 18px 22px; margin: 8px 0;
        font-family: 'DM Mono', monospace;
        transition: border-color 0.2s;
    }
    .position-profit { border-left: 3px solid #4ade80; box-shadow: -4px 0 16px rgba(74,222,128,0.06); }
    .position-loss   { border-left: 3px solid #f87171; box-shadow: -4px 0 16px rgba(248,113,113,0.06); }

    /* ── Order log rows ── */
    .order-row {
        display: flex; align-items: center; gap: 16px; padding: 13px 18px;
        border-radius: 12px; margin: 4px 0; font-family: 'DM Mono', monospace; font-size: 13px;
        background: #0b0c14; border: 1px solid rgba(255,255,255,0.04);
    }
    .order-filled  { border-left: 3px solid #4ade80; }
    .order-pending { border-left: 3px solid #fbbf24; }
    .order-error   { border-left: 3px solid #f87171; }

    /* ── Execute button ── */
    .exec-btn-container { margin-top: 10px; }

    /* ── Kill switch ── */
    .kill-switch-banner {
        background: linear-gradient(135deg, #1f0707, #160505);
        border: 1px solid rgba(239,68,68,0.3); border-radius: 16px;
        padding: 20px 24px; text-align: center; margin-bottom: 24px;
        box-shadow: 0 0 40px rgba(239,68,68,0.06);
    }

    /* ── Scan schedule ── */
    .scan-schedule {
        font-family: 'DM Mono', monospace; font-size: 12px;
        color: #4b5563; padding: 8px 14px; border-radius: 8px;
        background: rgba(11,12,20,0.6); margin: 4px 0;
    }
    .scan-active { border-left: 3px solid #4ade80; color: #94a3b8; }
    .scan-next   { border-left: 3px solid #fbbf24; color: #fbbf24; }
    .scan-done   { border-left: 3px solid #1f2937; opacity: 0.4; }

    /* ── How-to sidebar panel ── */
    .help-header {
        font-family: 'Syne', sans-serif; font-size: 16px; font-weight: 700;
        color: #e2e8f0; margin-bottom: 4px;
    }
    .help-subtitle {
        font-family: 'DM Sans', sans-serif; font-size: 11px; color: #4b5563;
        text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 20px;
    }
    .help-section {
        font-family: 'DM Sans', sans-serif; font-size: 11px; font-weight: 600;
        color: #6366f1; text-transform: uppercase; letter-spacing: 1.5px;
        margin: 18px 0 8px 0;
    }
    .help-body {
        font-family: 'DM Sans', sans-serif; font-size: 13px; color: #94a3b8;
        line-height: 1.7;
    }
    .help-rule {
        background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.15);
        border-radius: 8px; padding: 10px 14px; margin: 6px 0;
        font-family: 'DM Mono', monospace; font-size: 12px; color: #a5b4fc;
    }
    .help-warning {
        background: rgba(251,191,36,0.06); border: 1px solid rgba(251,191,36,0.2);
        border-radius: 8px; padding: 10px 14px; margin: 6px 0;
        font-family: 'DM Sans', sans-serif; font-size: 12px; color: #fcd34d;
    }

    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTOR INIT — one instance for the whole session
# ══════════════════════════════════════════════════════════════════════════════

def get_executor():
    """Return a cached AlpacaExecutor, or None if unavailable."""
    if not EXECUTOR_AVAILABLE:
        return None
    if 'executor' not in st.session_state:
        try:
            exe = AlpacaExecutor(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
            start_kill_switch_scheduler(exe)
            st.session_state.executor = exe
            if 'order_log' not in st.session_state:
                st.session_state.order_log = []
        except Exception as e:
            st.session_state.executor = None
            st.session_state._executor_error = str(e)
    return st.session_state.get('executor')


# ══════════════════════════════════════════════════════════════════════════════
# HOW-TO GUIDES — per tab content
# ══════════════════════════════════════════════════════════════════════════════

HOW_TO = {
    "ticker": {
        "title": "Single Ticker",
        "subtitle": "How to read & act on signals",
        "sections": [
            ("What this tab does", """
Analyzes any single ticker using the full pipeline: HMM regime detection → technical indicators → Strat candle classification → composite signal score.
Use it to deep-dive a specific stock before deciding to trade it.
"""),
            ("Reading the Signal card", """
The top-left card shows your action. Possible values:
"""),
            ("rules", [
                "0DTE CALL — Strong bullish signal, consider same-day call option",
                "0DTE PUT  — Strong bearish signal, consider same-day put option",
                "SWING LONG — Moderate bullish, multi-day long position",
                "SWING SHORT — Moderate bearish, multi-day short",
                "CASH — No edge detected. Stay flat.",
            ]),
            ("Reading the Regime card", """
The regime tells you the market environment. This gates everything else:
"""),
            ("rules", [
                "Bull Quiet  — Trending up, low volatility. Best for longs.",
                "Bull Volatile — Trending up but choppy. Reduce size.",
                "Bear Quiet  — Trending down. Best for shorts.",
                "Bear Volatile — Trending down and choppy. Reduce size.",
                "Chop — No trend. System says CASH. Respect it.",
            ]),
            ("Regime Probabilities bar", """
Shows the HMM confidence in each regime. Only trade when the top regime is ≥ 85% confident. Below that, the signal is unreliable.
"""),
            ("Execute button", """
Only appears on A+ or A grade signals. Always confirms before firing. This sends a paper order to Alpaca with a bracket (stop-loss + take-profit) already attached.
"""),
            ("Best practices", None),
            ("warning", "Never override a CASH signal because you 'feel' bullish. The regime is the filter."),
            ("warning", "Check the ADX value — below 20 means no trend. Even a bullish regime with low ADX is weak."),
        ]
    },
    "scanner": {
        "title": "S&P 500 Scanner",
        "subtitle": "Finding setups across the full market",
        "sections": [
            ("What this tab does", """
Runs a 3-tier scan across the full S&P 500 universe (~500 tickers) to surface the highest-quality setups meeting both regime AND signal thresholds.
"""),
            ("3-Tier architecture", None),
            ("rules", [
                "Tier 1 — Fast parallel screen (~500 tickers). Filters out low-momentum tickers in seconds.",
                "Tier 2 — Full HMM analysis on top candidates only. Regime confidence must be ≥ 85%.",
                "Tier 3 — Trade grader. Composite signal score must be ≥ 30. Grades A+/A/B/C separately for 0DTE and Swing.",
            ]),
            ("Scheduled scan times (ET)", None),
            ("rules", [
                "9:00 AM  — Pre-market setup scan",
                "9:30 AM  — Market open scan",
                "11:45 AM — Mid-morning momentum check",
                "2:45 PM  — Late session swing setups",
            ]),
            ("Reading a signal card", """
Each card shows two tracks — Intraday 0DTE and Swing Trade — graded independently. FTC (Full Timeframe Continuity), sector correlation, and ATR room are all factored in.
"""),
            ("Execute buttons", """
⚡ 0DTE button — fires a same-day equity order. Only visible on A+ or A intraday grade.
📊 Swing button — fires a multi-day position. Only visible on A+ or A swing grade.
Both require confirmation before executing.
"""),
            ("Session context banner", """
The banner at the top of the scanner shows the current session quality. Green = prime time. Yellow = caution. Red = avoid or exit only.
"""),
            ("Best practices", None),
            ("warning", "Run the 9:00 AM scan first — this is your morning briefing before the open."),
            ("warning", "Don't chase tickers that appeared in 'Earlier Today' — those setups are stale."),
            ("warning", "No entries in the first 15 minutes (before 9:45 AM ET). The system blocks this automatically."),
        ]
    },
    "live": {
        "title": "Live Trading",
        "subtitle": "Monitoring positions & account",
        "sections": [
            ("What this tab does", """
Real-time view of your Alpaca paper (or live) account. Shows buying power, open positions with live P&L, and gives you manual close controls.
"""),
            ("Account bar", """
Updates every 30 seconds. Shows Portfolio Value, Buying Power, Cash, and Today's P&L. The mode badge shows PAPER (blue) or LIVE (red).
"""),
            ("Open positions table", """
Each row shows entry price, current price, unrealized P&L in dollars and percent. Green left border = profitable. Red left border = loss.
Use the Close button on any position to exit it manually at market.
"""),
            ("Kill switch", """
The red 🚨 button at the top closes ALL open positions immediately. Requires double-confirmation. Use only if something goes badly wrong or at end of day.
"""),
            ("Automatic kill switch", None),
            ("rules", [
                "The system fires the kill switch automatically at 3:45 PM ET for all 0DTE positions.",
                "This runs in a background thread — it fires even if you close this tab.",
                "Swing positions are NOT auto-closed. You manage those manually.",
            ]),
            ("Best practices", None),
            ("warning", "Check this tab at 3:30 PM ET to review 0DTE positions before the auto-close fires."),
            ("warning", "Paper trading: treat every P&L number as if it were real money. Build the discipline now."),
            ("warning", "Refresh button forces a live account pull — use it if numbers look stale."),
        ]
    },
    "orders": {
        "title": "Order Log",
        "subtitle": "Reviewing your trade history",
        "sections": [
            ("What this tab does", """
Full record of every order placed today via Alpaca, plus a session log of orders fired through this dashboard.
"""),
            ("Today's Orders section", """
Pulls directly from Alpaca's API. Shows every order regardless of how it was submitted. Status color coding:
"""),
            ("rules", [
                "Green border — Filled. Order executed successfully.",
                "Yellow border — Pending / New. Order is live but not yet filled.",
                "Red border — Canceled or Rejected. Order did not execute.",
            ]),
            ("Session log section", """
Shows only orders fired through this dashboard during the current session. Includes grade, regime, trade type, and exact execution price logged at time of submission.
"""),
            ("What to review here", None),
            ("rules", [
                "Did A+ grades fill? Check filled_avg_price vs your entry assumption.",
                "Any rejected orders? May indicate market is closed or symbol is halted.",
                "Count of trades vs your max 6/day limit — don't overtrrade.",
            ]),
            ("Best practices", None),
            ("warning", "Clear Session Log at the start of each day — it resets with a fresh session."),
            ("warning", "Screenshot your order log at end of day for your trading journal."),
        ]
    },
    "backtest": {
        "title": "Walk-Forward Backtest",
        "subtitle": "Honest out-of-sample performance",
        "sections": [
            ("What this tab does", """
Runs a walk-forward analysis — the only honest way to test a trading system. Unlike standard backtests that optimize and test on the same data, this method keeps blind test periods completely separate.
"""),
            ("How it works", None),
            ("rules", [
                "Train on 1 year of data — finds best RSI/ADX parameters for that window.",
                "Test BLINDLY on the next window — the optimizer has never seen this data.",
                "Slide forward and repeat across all available history.",
                "Only the blind test results are stitched together for final metrics.",
            ]),
            ("Intraday vs Swing modes", """
Intraday (3-month blind test) — shorter test windows, focuses on daily signal quality. Run this to evaluate 0DTE setups.
Swing (18-month blind test) — longer windows, evaluates multi-day position holding. Run this for swing trade validation.
"""),
            ("Key metrics explained", None),
            ("rules", [
                "WF Efficiency — blind test Sharpe ÷ train Sharpe. Above 50% = system generalizes well. Below 0% = likely overfit.",
                "Alpha vs B&H — how much you beat simply holding the index. Must be positive to justify the system.",
                "Profit Factor — gross wins ÷ gross losses. Above 1.5 is solid. Above 2.0 is excellent.",
                "Max Drawdown — worst peak-to-trough loss in the blind test. Know this before trading live.",
            ]),
            ("Transaction costs", """
0.30% round-trip cost is baked into every backtest. This covers exchange fees (0.10%) and slippage (0.05% each way). Your real results will be close to these numbers.
"""),
            ("Best practices", None),
            ("warning", "Run backtest on SPY first before any individual stock — it's your baseline."),
            ("warning", "If WF Efficiency is below 0%, the system is curve-fit for that ticker. Don't trade it live."),
            ("warning", "Backtest results are historical. They don't guarantee future performance."),
        ]
    },
}


def render_help_sidebar(tab_key: str):
    """Render the how-to guide for the given tab in the sidebar."""
    guide = HOW_TO.get(tab_key)
    if not guide:
        return

    st.markdown(f"""
    <div class="help-header">📖 {guide['title']}</div>
    <div class="help-subtitle">{guide['subtitle']}</div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.05);margin:0 0 16px 0;'>", unsafe_allow_html=True)

    i = 0
    sections = guide['sections']
    while i < len(sections):
        item = sections[i]
        label, content = item

        if label == "rules":
            for rule in content:
                st.markdown(f'<div class="help-rule">→ {rule}</div>', unsafe_allow_html=True)
        elif label == "warning":
            st.markdown(f'<div class="help-warning">⚠️ {content}</div>', unsafe_allow_html=True)
        elif content is None:
            st.markdown(f'<div class="help-section">{label}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="help-section">{label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="help-body">{content.strip()}</div>', unsafe_allow_html=True)
        i += 1

    st.markdown("<hr style='border-color:rgba(255,255,255,0.05);margin:20px 0 12px 0;'>", unsafe_allow_html=True)
    if st.button("✕ Close Guide", use_container_width=True):
        st.session_state.show_help = False
        st.rerun()


def show_tab_header(page_key: str, title: str, date_str: str = ""):
    """Render tab title + fine-print how-to link."""
    date_part = f" — {date_str}" if date_str else ""
    st.markdown(f"""
    <div class="page-title-bar">
        <span class="page-title">{title}{date_part}</span>
        <span class="how-to-link" id="howto_{page_key}" style="cursor:pointer;">
            📖 how to use this tab
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Tiny button that triggers the help panel
    col_spacer, col_btn = st.columns([6, 1])
    with col_btn:
        if st.button("📖 Guide", key=f"help_btn_{page_key}",
                     help="Open the how-to guide in the left panel"):
            st.session_state.show_help = True
            st.session_state.help_tab  = page_key
            st.rerun()

    st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)


def get_account_data():
    """Fetch live account summary, cached for 30s."""
    exe = get_executor()
    if exe is None:
        return None
    cache_key = '_acct_cache'
    cache_time_key = '_acct_cache_time'
    if (cache_key in st.session_state and
            time.time() - st.session_state.get(cache_time_key, 0) < 30):
        return st.session_state[cache_key]
    try:
        data = exe.get_account_summary()
        st.session_state[cache_key] = data
        st.session_state[cache_time_key] = time.time()
        return data
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ACCOUNT BAR — shown on trading-related pages
# ══════════════════════════════════════════════════════════════════════════════

def render_account_bar():
    acct = get_account_data()
    if acct is None:
        return
    pnl = acct.get('pnl_today', 0)
    pnl_color = '#22c55e' if pnl >= 0 else '#ef4444'
    pnl_sign  = '+' if pnl >= 0 else ''
    mode_badge = '📄 PAPER' if ALPACA_PAPER else '🔴 LIVE'

    st.markdown(f"""
    <div class="account-bar">
        <div class="account-bar-item">
            <div class="account-bar-label">Mode</div>
            <div class="account-bar-value" style="font-size:14px;color:#6366f1;">{mode_badge}</div>
        </div>
        <div style="width:1px;height:36px;background:rgba(255,255,255,0.06);"></div>
        <div class="account-bar-item">
            <div class="account-bar-label">Portfolio Value</div>
            <div class="account-bar-value">${acct.get('portfolio_value',0):,.2f}</div>
        </div>
        <div class="account-bar-item">
            <div class="account-bar-label">Buying Power</div>
            <div class="account-bar-value">${acct.get('buying_power',0):,.2f}</div>
        </div>
        <div class="account-bar-item">
            <div class="account-bar-label">Cash</div>
            <div class="account-bar-value">${acct.get('cash',0):,.2f}</div>
        </div>
        <div style="width:1px;height:36px;background:rgba(255,255,255,0.06);"></div>
        <div class="account-bar-item">
            <div class="account-bar-label">Today's P&L</div>
            <div class="account-bar-value" style="color:{pnl_color};">{pnl_sign}${pnl:,.2f}</div>
        </div>
        <div style="margin-left:auto;font-family:JetBrains Mono,monospace;font-size:11px;color:#374151;">
            Last updated {datetime.now().strftime('%H:%M:%S')}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE TICKER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def run_single_analysis(ticker, days=730):
    from lib.data_client import get_daily
    df = get_daily(ticker, days=days)
    if df.empty:
        return None

    df['daily_return'] = df['Close'].pct_change()
    df['vol_20']       = df['daily_return'].rolling(20).std() * np.sqrt(252)
    df['range_pct']    = (df['High'] - df['Low']) / df['Close']
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    else:
        df['volume_ratio'] = 1.0
    df = df.dropna()

    features    = ['daily_return', 'vol_20', 'volume_ratio', 'range_pct']
    X           = df[features].values
    regime_probs = None

    try:
        from hmmlearn.hmm import GaussianHMM
        model  = GaussianHMM(n_components=5, covariance_type='full', n_iter=200, random_state=42)
        model.fit(X)
        states = model.predict(X)
        df['regime_id'] = states
        labels  = {0:"Bear_Volatile",1:"Bear_Quiet",2:"Chop",3:"Bull_Quiet",4:"Bull_Volatile"}
        means   = df.groupby('regime_id')['daily_return'].mean().sort_values()
        sorted_ids = means.index.tolist()
        lk      = sorted(labels.keys())[:len(sorted_ids)]
        id_map  = {rid: labels[lk[i]] for i, rid in enumerate(sorted_ids) if i < len(lk)}
        df['regime'] = df['regime_id'].map(id_map).fillna('Chop')
        posteriors   = model.predict_proba(X)
        latest_probs = posteriors[-1]
        regime_probs = {}
        for i, rid in enumerate(sorted_ids):
            if i < len(lk):
                regime_probs[labels[lk[i]]] = round(latest_probs[rid] * 100, 1)
    except ImportError:
        ret_q = df['daily_return'].quantile([0.15, 0.85])
        vol_q = df['vol_20'].quantile(0.5)
        def assign(row):
            r, v = row['daily_return'], row['vol_20']
            if r > ret_q[0.85] and v < vol_q: return "Bull_Quiet"
            elif r > ret_q[0.85]:              return "Bull_Volatile"
            elif r < ret_q[0.15] and v < vol_q: return "Bear_Quiet"
            elif r < ret_q[0.15]:              return "Bear_Volatile"
            else:                              return "Chop"
        df['regime'] = df.apply(assign, axis=1)

    # Indicators
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0.0).ewm(span=14, adjust=False).mean()
    loss  = (-delta).where(delta < 0, 0.0).ewm(span=14, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)

    h, lo, c  = df['High'], df['Low'], df['Close']
    pdm       = h.diff(); mdm = -lo.diff()
    pdm       = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm       = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    tr        = pd.concat([h-lo, (h-c.shift(1)).abs(), (lo-c.shift(1)).abs()], axis=1).max(axis=1)
    atr_s     = tr.ewm(span=14, adjust=False).mean()
    pdi       = 100*(pdm.ewm(span=14, adjust=False).mean()/atr_s.replace(0, np.nan))
    mdi       = 100*(mdm.ewm(span=14, adjust=False).mean()/atr_s.replace(0, np.nan))
    dx        = (abs(pdi-mdi)/(pdi+mdi).replace(0, np.nan))*100
    df['adx'] = dx.ewm(span=14, adjust=False).mean()
    df['adx_rising'] = df['adx'] > df['adx'].shift(1)

    ef = df['Close'].ewm(span=12, adjust=False).mean()
    es = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd']        = ef - es
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['macd_cross_up']   = (df['macd']>df['macd_signal'])&(df['macd'].shift(1)<=df['macd_signal'].shift(1))
    df['macd_cross_down'] = (df['macd']<df['macd_signal'])&(df['macd'].shift(1)>=df['macd_signal'].shift(1))
    df['momentum']    = df['Close'].pct_change(10) * 100

    df['bb_mid']   = df['Close'].rolling(20).mean()
    std            = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + std * 2
    df['bb_lower'] = df['bb_mid'] - std * 2
    df['bb_pct']   = (df['Close']-df['bb_lower'])/(df['bb_upper']-df['bb_lower']).replace(0, np.nan)
    df['bb_squeeze'] = std < std.rolling(50).mean()
    df['atr']      = tr.rolling(14).mean()

    # Strat
    df['strat_type'] = 0; df['strat_dir'] = 'neutral'
    for i in range(1, len(df)):
        prev, curr = df.iloc[i-1], df.iloc[i]
        bh     = curr['High'] > prev['High']
        bl     = curr['Low']  < prev['Low']
        inside = curr['Low'] >= prev['Low'] and curr['High'] <= prev['High']
        if inside:
            df.iloc[i, df.columns.get_loc('strat_type')] = 1
        elif bh and bl:
            df.iloc[i, df.columns.get_loc('strat_type')] = 3
            df.iloc[i, df.columns.get_loc('strat_dir')]  = 'up' if curr['Close'] > curr['Open'] else 'down'
        elif bh:
            df.iloc[i, df.columns.get_loc('strat_type')] = 2
            df.iloc[i, df.columns.get_loc('strat_dir')]  = 'up'
        elif bl:
            df.iloc[i, df.columns.get_loc('strat_type')] = 2
            df.iloc[i, df.columns.get_loc('strat_dir')]  = 'down'

    types = df['strat_type'].tail(5).tolist()
    dirs  = df['strat_dir'].tail(5).tolist()
    patterns = []
    if len(types) >= 3:
        t3,t2,t1 = types[-3],types[-2],types[-1]
        d3,d2,d1 = dirs[-3],dirs[-2],dirs[-1]
        if t3==2 and t2==1 and t1==2 and d3!=d1 and d1!='neutral':
            patterns.append({'name':'2-1-2 Reversal','direction':d1,'grade':'A+'})
        if t3==2 and t2==1 and t1==2 and d3==d1 and d1!='neutral':
            patterns.append({'name':'2-1-2 Continuation','direction':d1,'grade':'A'})
        if t2==3 and t1==2 and d1!='neutral':
            patterns.append({'name':'3-2 Continuation','direction':d1,'grade':'A'})
        if t2==3 and t1==1:
            patterns.append({'name':'3-1 Compression','direction':'pending','grade':'B+'})
    if len(types) >= 2:
        if types[-2]==2 and types[-1]==2 and dirs[-2]!=dirs[-1] and dirs[-1]!='neutral':
            patterns.append({'name':'2-2 Reversal','direction':dirs[-1],'grade':'A'})
    if sum(1 for t in types[-3:] if t == 1) >= 2:
        patterns.append({'name':'Inside Compression','direction':'pending','grade':'B'})

    regime  = df.iloc[-1]['regime']
    latest  = df.iloc[-1]
    close_price = float(latest['Close'])
    snap = {
        'rsi':         round(float(latest.get('rsi', 50)), 1),
        'adx':         round(float(latest.get('adx', 0)), 1),
        'adx_rising':  bool(latest.get('adx_rising', False)),
        'macd_hist':   round(float(latest.get('macd_hist', 0)), 4),
        'macd_cross_up':   bool(latest.get('macd_cross_up', False)),
        'macd_cross_down': bool(latest.get('macd_cross_down', False)),
        'momentum':    round(float(latest.get('momentum', 0)), 2),
        'bb_pct':      round(float(latest.get('bb_pct', 0.5)), 2),
        'bb_squeeze':  bool(latest.get('bb_squeeze', False)),
        'vol_ratio':   round(float(latest.get('volume_ratio', 1.0)), 2),
        'high_volume': float(latest.get('volume_ratio', 1.0)) > 1.5,
        'close':       close_price,
    }

    regime_scores = {"Bull_Quiet":40,"Bull_Volatile":25,"Chop":0,"Bear_Quiet":-40,"Bear_Volatile":-25}
    r_score = regime_scores.get(regime, 0)
    i_score = 0
    rsi = snap['rsi']; adx = snap['adx']
    if regime in ["Bull_Quiet","Bull_Volatile"]:
        if 50 < rsi < 70: i_score += 15
        elif rsi >= 70:   i_score -= 10
        if adx > 25 and snap['adx_rising']: i_score += 15
        elif adx > 25:    i_score += 8
        if snap['macd_cross_up']:   i_score += 12
        elif snap['macd_hist'] > 0: i_score += 5
        elif snap['macd_cross_down']: i_score -= 15
        if snap['momentum'] > 0:  i_score += 5
        if snap['vol_ratio'] > 1.5: i_score += 5
    elif regime in ["Bear_Quiet","Bear_Volatile"]:
        if 30 < rsi < 50: i_score -= 15
        elif rsi <= 30:   i_score += 10
        if adx > 25 and snap['adx_rising']: i_score -= 15
        elif adx > 25:    i_score -= 8
        if snap['macd_cross_down']:  i_score -= 12
        elif snap['macd_hist'] < 0:  i_score -= 5
        elif snap['macd_cross_up']:  i_score += 15
        if snap['momentum'] < 0:  i_score -= 5
    else:
        if rsi > 70 and snap['bb_pct'] > 0.95: i_score -= 10
        elif rsi < 30 and snap['bb_pct'] < 0.05: i_score += 10
        if adx < 20: i_score = int(i_score * 0.5)

    s_score = 0
    for p in patterns:
        w = {'A+':20,'A':15,'B+':8,'B':5}.get(p['grade'], 3)
        if p['direction'] == 'pending': s_score += w * 0.3; continue
        bp = p['direction'] == 'up'
        if bp and regime in ["Bull_Quiet","Bull_Volatile"]:     s_score += w
        elif not bp and regime in ["Bear_Quiet","Bear_Volatile"]: s_score -= w
        else: s_score += (1 if bp else -1) * w * 0.5

    composite  = max(-100, min(100, r_score + i_score + int(s_score)))
    confidence = 50
    if regime_probs:
        confidence = min(95, max(regime_probs.values()))
        if confidence < 50: composite = int(composite * 0.7)

    a          = abs(composite)
    strength   = 'STRONG' if a >= 50 else ('MODERATE' if a >= 30 else ('WEAK' if a >= 15 else 'NO_TRADE'))
    direction  = 'LONG' if composite > 15 else ('SHORT' if composite < -15 else 'FLAT')
    if strength == 'NO_TRADE' or direction == 'FLAT': tt = 'CASH'
    elif direction == 'LONG' and strength == 'STRONG':  tt = '0DTE CALL'
    elif direction == 'LONG':                            tt = 'SWING LONG'
    elif direction == 'SHORT' and strength == 'STRONG': tt = '0DTE PUT'
    else:                                                tt = 'SWING SHORT'

    signal = {
        'composite': composite, 'direction': direction, 'strength': strength,
        'trade_type': tt, 'confidence': confidence,
        'r_score': r_score, 'i_score': i_score, 's_score': int(s_score),
    }

    return {
        'df': df, 'regime': regime, 'regime_probs': regime_probs,
        'signal': signal, 'snap': snap, 'patterns': patterns,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_candlestick_chart(df, ticker, show_days=120):
    df_plot = df.tail(show_days).copy()
    df_plot.index = pd.to_datetime(df_plot.index)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.6, 0.2, 0.2])

    regime_colors = {
        "Bull_Quiet":    "rgba(34,197,94,0.12)",
        "Bull_Volatile": "rgba(34,197,94,0.08)",
        "Bear_Quiet":    "rgba(239,68,68,0.12)",
        "Bear_Volatile": "rgba(239,68,68,0.08)",
        "Chop":          "rgba(148,163,184,0.06)",
    }

    prev_regime = None; start_idx = None
    for idx, row in df_plot.iterrows():
        regime = row.get('regime', 'Chop')
        if regime != prev_regime:
            if prev_regime is not None and start_idx is not None:
                fig.add_vrect(x0=start_idx, x1=idx,
                    fillcolor=regime_colors.get(prev_regime, "rgba(0,0,0,0)"),
                    layer="below", line_width=0, row=1, col=1)
            start_idx = idx; prev_regime = regime
    if prev_regime and start_idx:
        fig.add_vrect(x0=start_idx, x1=df_plot.index[-1],
            fillcolor=regime_colors.get(prev_regime, "rgba(0,0,0,0)"),
            layer="below", line_width=0, row=1, col=1)

    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'],
        increasing_line_color='#22c55e', decreasing_line_color='#ef4444',
        increasing_fillcolor='#22c55e', decreasing_fillcolor='#ef4444',
        showlegend=False), row=1, col=1)

    if 'bb_upper' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['bb_upper'],
            line=dict(color='rgba(99,102,241,0.3)', width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['bb_lower'],
            line=dict(color='rgba(99,102,241,0.3)', width=1),
            fill='tonexty', fillcolor='rgba(99,102,241,0.05)', showlegend=False), row=1, col=1)

    if 'rsi' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['rsi'],
            line=dict(color='#a78bfa', width=1.5), showlegend=False), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.4)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(34,197,94,0.4)", row=2, col=1)

    if 'macd_hist' in df_plot.columns:
        colors = ['#22c55e' if v >= 0 else '#ef4444' for v in df_plot['macd_hist']]
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['macd_hist'],
            marker_color=colors, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['macd'],
            line=dict(color='#60a5fa', width=1), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['macd_signal'],
            line=dict(color='#f97316', width=1), showlegend=False), row=3, col=1)

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#0a0b0f', plot_bgcolor='#0a0b0f',
        height=600, margin=dict(l=50, r=20, t=30, b=30), xaxis_rangeslider_visible=False,
        font=dict(family='Inter', size=12, color='#94a3b8'), hovermode='x unified')
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.03)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.03)')
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI",   row=2, col=1)
    fig.update_yaxes(title_text="MACD",  row=3, col=1)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

if 'page' not in st.session_state:
    st.session_state.page = "ticker"

nav_cols = st.columns([1, 1, 1, 1, 1, 1, 1])
pages = [
    ("📊 Ticker",     "ticker"),
    ("🎯 Scanner",    "scanner"),
    ("📌 Tracker",    "tracker"),
    ("💼 Trading",    "live"),
    ("🚨 Alerts",     "orders"),
    ("📜 Order Log",  "orderlog"),
    ("📈 Backtest",   "backtest"),
]
for col, (label, key) in zip(nav_cols, pages):
    with col:
        btn_type = "primary" if st.session_state.page == key else "secondary"
        if st.button(label, use_container_width=True, type=btn_type):
            st.session_state.page = key
            st.rerun()

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── Show help guide if active, otherwise show settings ───────────────────
    if st.session_state.get('show_help') and st.session_state.get('help_tab'):
        render_help_sidebar(st.session_state.help_tab)
    else:
        st.markdown("## ⚙️ Settings")

        ticker     = st.selectbox("Ticker",
            ['SPY','QQQ','IWM','NVDA','AAPL','TSLA','MU','WMT','UNH','ELF','GM','AMD','AMZN','META','GOOGL'], index=0)
        chart_days = st.slider("Chart Window", 30, 365, 120)
        train_days = st.selectbox("HMM Training", [365, 500, 730, 1000], index=2)

        st.markdown("---")
        st.markdown("### Risk")
        risk_0dte  = st.number_input("0DTE ($)",  value=75,  step=5)
        risk_swing = st.number_input("Swing ($)", value=150, step=10)

        st.markdown("---")
        st.markdown("### Scan Schedule (ET)")
        next_scan = get_next_scan_time()
        for scan in SCAN_TIMES:
            t      = f"{scan['hour']}:{scan['minute']:02d}"
            dt     = datetime.now().replace(hour=scan['hour'], minute=scan['minute'])
            is_past = datetime.now() > dt
            is_next = scan['label'] == next_scan.get('label', '')
            css    = 'scan-done' if is_past and not is_next else ('scan-next' if is_next else '')
            icon   = '✅' if is_past and not is_next else ('⏳' if is_next else '⬜')
            st.markdown(f'<div class="scan-schedule {css}">{icon} {scan["label"]} — {t}</div>', unsafe_allow_html=True)
        if next_scan.get('minutes_until'):
            st.markdown(f"**Next scan in {next_scan['minutes_until']} min**")

        st.markdown("---")

        # Alpaca status in sidebar
        if EXECUTOR_AVAILABLE:
            acct = get_account_data()
            if acct:
                status_color = '#4ade80' if acct['status'] == 'ACTIVE' else '#f87171'
                st.markdown(f"""
                <div style="background:#0b0c14;border-radius:12px;padding:14px;border:1px solid rgba(99,102,241,0.12);">
                    <div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;font-family:'DM Sans',sans-serif;">Alpaca Account</div>
                    <div style="font-family:'DM Mono',monospace;font-size:12px;color:{status_color};margin-bottom:4px;">● {acct['status']}</div>
                    <div style="font-family:'DM Mono',monospace;font-size:12px;color:#94a3b8;">💰 ${acct['cash']:,.0f} cash</div>
                    <div style="font-family:'DM Mono',monospace;font-size:12px;color:#94a3b8;">📊 ${acct['portfolio_value']:,.0f} portfolio</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#120e04;border-radius:10px;padding:12px;border:1px solid rgba(251,191,36,0.2);">
                <div style="font-size:11px;color:#fbbf24;font-family:'DM Sans',sans-serif;">⚠️ Executor not connected.<br>Check config.py for Alpaca keys.</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Free-tier data delay banner ────────────────────────────────────────
        st.markdown("""
        <div style="background:#1a1205;border-radius:10px;padding:10px 12px;
            border:1px solid rgba(251,191,36,0.35);margin-bottom:8px;">
            <div style="font-size:11px;color:#f59e0b;font-family:'DM Sans',sans-serif;line-height:1.5;">
                ⏱️ <b>Free Tier (IEX)</b><br>
                Intraday data: 15-min delayed<br>
                Daily / Weekly: accurate
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Psychology pre-trade check ─────────────────────────────────────────
        st.markdown("### 🧠 Mindset Check")
        mood = st.select_slider(
            "Pre-trade state",
            options=["Revenge", "Fearful", "Neutral", "Focused", "Sharp"],
            value="Neutral",
            label_visibility="collapsed",
        )
        mood_colors = {
            "Revenge": "#ef4444", "Fearful": "#f59e0b",
            "Neutral": "#6366f1", "Focused": "#22c55e", "Sharp": "#22c55e",
        }
        mood_notes = {
            "Revenge": "Step away. Revenge trading destroys accounts.",
            "Fearful": "Size down or sit out. Fear makes you exit early.",
            "Neutral": "Proceed carefully. Wait for A-grade setups only.",
            "Focused": "Good state. Execute your plan.",
            "Sharp":   "Prime state. Full conviction on A+ setups.",
        }
        mood_color = mood_colors.get(mood, "#6366f1")
        st.markdown(f"""
        <div style="background:{mood_color}11;border-radius:8px;padding:8px 10px;
            border:1px solid {mood_color}44;margin-bottom:4px;">
            <div style="font-size:11px;color:{mood_color};font-family:'DM Sans',sans-serif;">
                {mood_notes.get(mood, '')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        if mood in ("Revenge", "Fearful"):
            st.session_state['trading_locked_mood'] = True
        else:
            st.session_state['trading_locked_mood'] = False

        st.markdown("---")
        st.markdown("<div style='text-align:center;color:#1f2937;font-size:11px;font-family:DM Sans,sans-serif;'>Trading Realm v3 · Not financial advice</div>", unsafe_allow_html=True)

# ── Session state defaults for settings (used outside sidebar when help is shown) ──
if 'ticker'     not in st.session_state: st.session_state.ticker     = 'SPY'
if 'chart_days' not in st.session_state: st.session_state.chart_days = 120
if 'train_days' not in st.session_state: st.session_state.train_days = 730
if 'risk_0dte'  not in st.session_state: st.session_state.risk_0dte  = 75
if 'risk_swing' not in st.session_state: st.session_state.risk_swing = 150

# Use session state values when help panel hides settings
if st.session_state.get('show_help'):
    ticker     = st.session_state.get('ticker', 'SPY')
    chart_days = st.session_state.get('chart_days', 120)
    train_days = st.session_state.get('train_days', 730)
    risk_0dte  = st.session_state.get('risk_0dte', 75)
    risk_swing = st.session_state.get('risk_swing', 150)
else:
    # Save to session state for persistence
    try:
        st.session_state.ticker     = ticker
        st.session_state.chart_days = chart_days
        st.session_state.train_days = train_days
        st.session_state.risk_0dte  = risk_0dte
        st.session_state.risk_swing = risk_swing
    except Exception:
        ticker = chart_days = train_days = risk_0dte = risk_swing = None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def grade_css_style(grade):
    return {
        'A+':      'background:#166534;color:#4ade80;border:1px solid #22c55e;',
        'A':       'background:#1e3a5f;color:#60a5fa;border:1px solid #3b82f6;',
        'B':       'background:#3b3216;color:#fbbf24;border:1px solid #f59e0b;',
        'C':       'background:#2d2d2d;color:#a1a1aa;border:1px solid #71717a;',
        'NO_TRADE':'background:#1a1a1a;color:#64748b;border:1px solid #374151;',
    }.get(grade, 'background:#1a1a1a;color:#64748b;')


def is_executable_grade(grade):
    return grade in ('A+', 'A')


def log_order(result: dict):
    if 'order_log' not in st.session_state:
        st.session_state.order_log = []
    st.session_state.order_log.insert(0, result)


def render_daily_loss_gate():
    """Show a red banner if today's P&L losses exceed 2× average trade risk."""
    if not PNL_AVAILABLE:
        return
    try:
        stats = get_stats_for_period('today')
        if not stats or stats.get('total_trades', 0) == 0:
            return
        total_pnl = stats.get('total_pnl', 0)
        avg_loss  = abs(stats.get('avg_loss', 0))
        threshold = max(avg_loss * 2, 150)  # 2× avg loss or $150 min
        if total_pnl < -threshold:
            st.markdown(f"""
            <div style="background:#1a0505;border:1px solid #ef4444;border-radius:10px;
                padding:14px 20px;margin-bottom:16px;display:flex;align-items:center;gap:12px;">
                <div style="font-size:24px;">🚨</div>
                <div>
                    <div style="color:#ef4444;font-weight:700;font-size:14px;font-family:'DM Mono',monospace;">
                        DAILY LOSS GATE — Down ${abs(total_pnl):.0f} today
                    </div>
                    <div style="color:#94a3b8;font-size:12px;font-family:'DM Sans',sans-serif;">
                        Loss exceeds 2× average. Review your trades before continuing.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SINGLE TICKER
# ══════════════════════════════════════════════════════════════════════════════

if page == "ticker":
    with st.spinner(f"Analyzing {ticker}..."):
        data = run_single_analysis(ticker, days=train_days)

    if data is None:
        st.error(f"No data for {ticker}"); st.stop()

    regime   = data['regime']
    signal   = data['signal']
    snap     = data['snap']
    patterns = data['patterns']

    show_tab_header("ticker", f"📊 {ticker}", datetime.now().strftime('%B %d, %Y'))
    render_daily_loss_gate()

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        sc = 'signal-long' if signal['direction']=='LONG' else ('signal-short' if signal['direction']=='SHORT' else 'signal-flat')
        st.markdown(f"""
        <div class="signal-card {sc}">
            <div class="signal-label">Signal</div>
            <div class="signal-value">{signal['trade_type']}</div>
            <div class="signal-sub">{signal['strength']} | Score: {signal['composite']:+d}</div>
        </div>""", unsafe_allow_html=True)

        # Execute button for A+/A grades on single ticker
        if EXECUTOR_AVAILABLE and is_executable_grade(signal.get('strength','')) and signal['direction'] != 'FLAT':
            exe_key = f"exec_ticker_{ticker}"
            if st.button(f"⚡ EXECUTE ON ALPACA", key=exe_key, type="primary", use_container_width=True):
                st.session_state[f"confirm_{exe_key}"] = True

            if st.session_state.get(f"confirm_{exe_key}"):
                st.warning(f"⚠️ Confirm paper trade: {signal['direction']} {ticker} @ ~${snap['close']:.2f}")
                col_y, col_n = st.columns(2)
                with col_y:
                    if st.button("✅ YES, EXECUTE", key=f"yes_{exe_key}", type="primary"):
                        exe = get_executor()
                        result = exe.submit_equity_order(
                            symbol=ticker,
                            side='buy' if signal['direction']=='LONG' else 'sell',
                            price=snap['close'],
                            regime=regime,
                            trade_type='0DTE' if '0DTE' in signal['trade_type'] else 'swing',
                            grade='A',
                        )
                        if result:
                            log_order(result)
                            st.success(f"✅ Order submitted! ID: {result['order_id'][:8]}...")
                        else:
                            st.error("❌ Order blocked by session/regime rules.")
                        st.session_state[f"confirm_{exe_key}"] = False
                with col_n:
                    if st.button("❌ Cancel", key=f"no_{exe_key}"):
                        st.session_state[f"confirm_{exe_key}"] = False

    with c2:
        rc = regime.lower().replace(' ','_')
        rm = {"Bull_Quiet":1.0,"Bull_Volatile":0.75,"Chop":0.4,"Bear_Quiet":1.0,"Bear_Volatile":0.5}.get(regime, 0.5)
        st.markdown(f"""
        <div class="signal-card signal-flat">
            <div class="signal-label">Regime</div>
            <div class="signal-value" style="font-size:26px;color:#e2e8f0;">
                <span class="regime-badge regime-{rc}">{regime}</span>
            </div>
            <div class="signal-sub">Risk: ${round(risk_0dte*rm)} | Conf: {signal['confidence']}%</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        if data['regime_probs']:
            html = '<div class="signal-card signal-flat"><div class="signal-label">Regime Probabilities</div>'
            for r, p in sorted(data['regime_probs'].items(), key=lambda x: -x[1]):
                clr = '#22c55e' if 'Bull' in r else ('#ef4444' if 'Bear' in r else '#94a3b8')
                html += f'<div style="display:flex;align-items:center;margin:4px 0;font-family:JetBrains Mono,monospace;font-size:13px;"><span style="width:120px;color:#94a3b8;">{r}</span><div style="flex:1;background:rgba(255,255,255,0.05);border-radius:4px;height:16px;margin:0 8px;"><div style="width:{p}%;background:{clr};height:100%;border-radius:4px;opacity:0.7;"></div></div><span style="width:45px;text-align:right;color:{clr};">{p}%</span></div>'
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)

    if patterns:
        st.markdown("<div class='section-header' style='margin-top:24px;'>Active Strat Patterns</div>", unsafe_allow_html=True)
        pills = ""
        for p in patterns:
            gc    = {'A+':'grade-aplus','A':'grade-a','B+':'grade-bplus','B':'grade-b'}.get(p['grade'],'grade-b')
            arrow = '↑' if p['direction']=='up' else ('↓' if p['direction']=='down' else '⏳')
            pills += f'<span class="strat-pill {gc}">{p["name"]} [{p["grade"]}] {arrow}</span>'
        st.markdown(pills, unsafe_allow_html=True)

    st.markdown(f"<div class='section-header' style='margin-top:24px;'>{ticker} — Regime Chart</div>", unsafe_allow_html=True)
    fig = build_candlestick_chart(data['df'], ticker, show_days=chart_days)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div class='section-header'>Indicators</div>", unsafe_allow_html=True)
    ic       = st.columns(6)
    ind_data = [
        ("RSI",      f"{snap['rsi']:.0f}",       "metric-positive" if 40<snap['rsi']<70 else "metric-negative", ""),
        ("ADX",      f"{snap['adx']:.0f}",        "metric-positive" if snap['adx']>25 else "", f"{'↑ Rising' if snap['adx_rising'] else '↓ Falling'}"),
        ("MACD",     f"{snap['macd_hist']:.3f}",  "metric-positive" if snap['macd_hist']>0 else "metric-negative", ""),
        ("Momentum", f"{snap['momentum']:.1f}%",  "metric-positive" if snap['momentum']>0 else "metric-negative", ""),
        ("BB %",     f"{snap['bb_pct']:.2f}",     "", f"{'🔥 Squeeze' if snap['bb_squeeze'] else ''}"),
        ("Volume",   f"{snap['vol_ratio']:.1f}x", "metric-positive" if snap['vol_ratio']>1.5 else "", f"{'High' if snap['high_volume'] else 'Normal'}"),
    ]
    for col, (label, val, css, sub) in zip(ic, ind_data):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {css}">{val}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

    # ── FTFC Stack ────────────────────────────────────────────────────────────
    try:
        from lib.trade_grader import build_ftfc_stack
        with st.expander("🏗️ Full Timeframe Continuity (FTFC) Stack", expanded=False):
            ftfc_mode = st.radio("Mode", ["intraday", "swing"], horizontal=True, key="ftfc_mode_ticker")
            with st.spinner("Fetching timeframe data..."):
                ftfc = build_ftfc_stack(ticker, mode=ftfc_mode)
            consensus_color = '#22c55e' if 'bull' in ftfc['consensus'] else ('#ef4444' if 'bear' in ftfc['consensus'] else '#94a3b8')
            st.markdown(f"""
            <div style="display:flex;gap:8px;margin-bottom:12px;align-items:center;">
                <span style="font-family:'DM Mono',monospace;font-size:14px;font-weight:700;color:{consensus_color};">
                    {ftfc['consensus'].upper()}
                </span>
                <span style="font-size:12px;color:#6b7280;">
                    {ftfc['up_count']}↑ {ftfc['dn_count']}↓ / {ftfc['total']} TFs
                    {'✅ CONFIRMED' if ftfc['confirmed'] else ''}
                </span>
            </div>
            """, unsafe_allow_html=True)
            cols = st.columns(len(ftfc['stack']))
            for col, s in zip(cols, ftfc['stack']):
                color = '#22c55e' if s['direction']=='up' else ('#ef4444' if s['direction']=='down' else '#6b7280')
                arrow = '↑' if s['direction']=='up' else ('↓' if s['direction']=='down' else '—')
                with col:
                    st.markdown(f"""
                    <div style="text-align:center;background:#0f1117;border:1px solid {color}44;
                        border-radius:8px;padding:8px 4px;">
                        <div style="font-size:10px;color:#6b7280;font-family:'DM Mono',monospace;">{s['tf']}</div>
                        <div style="font-size:20px;color:{color};">{arrow}</div>
                        <div style="font-size:9px;color:#4b5563;">${s['close'] or '—'}</div>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception:
        pass


    st.markdown("<div class='section-header' style='margin-top:24px;'>Signal Breakdown</div>", unsafe_allow_html=True)
    sb     = st.columns(4)
    scores = [
        ("Regime",    signal['r_score']),
        ("Indicators",signal['i_score']),
        ("Strat",     signal['s_score']),
        ("Composite", signal['composite']),
    ]
    for col, (label, val) in zip(sb, scores):
        css = "metric-positive" if val > 0 else ("metric-negative" if val < 0 else "")
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {css}">{val:+d}</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: S&P 500 SCANNER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "scanner":
    show_tab_header("scanner", "🎯 S&P 500 Scanner", datetime.now().strftime('%B %d, %Y'))
    render_daily_loss_gate()

    if 'scan_history'      not in st.session_state: st.session_state.scan_history      = load_scan_history()
    if 'last_scan_results' not in st.session_state: st.session_state.last_scan_results = None
    if 'play_sound'        not in st.session_state: st.session_state.play_sound        = False

    col_btn1, col_btn2, col_info = st.columns([1, 1, 3])
    with col_btn1:
        run_scan   = st.button("🚀 Run Full Scan", type="primary", use_container_width=True)
    with col_btn2:
        clear_hist = st.button("🗑️ Clear History", use_container_width=True)
    with col_info:
        next_scan = get_next_scan_time()
        if next_scan.get('minutes_until'):
            st.markdown(f"<div style='padding:10px;color:#94a3b8;font-family:JetBrains Mono,monospace;font-size:14px;'>⏳ Next auto-scan: <b>{next_scan['label']}</b> at {next_scan['time']} ({next_scan['minutes_until']} min)</div>", unsafe_allow_html=True)

    if clear_hist:
        st.session_state.scan_history = []; save_scan_history([]); st.rerun()

    auto_trigger, auto_label = should_auto_scan()
    if auto_trigger and not run_scan:
        last_auto = st.session_state.get('last_auto_scan', '')
        if last_auto != auto_label:
            run_scan = True
            st.session_state.last_auto_scan = auto_label
            st.toast(f"⏰ Auto-scan triggered: {auto_label}")

    if run_scan:
        st.markdown(get_alert_sound_html("scan_start"), unsafe_allow_html=True)
        progress_bar = st.progress(0); status_text = st.empty()

        def update_progress(stage, current, total, message):
            pct = int((current/total)*40) if stage=="tier1" else (40+int((current/max(1,total))*40) if stage=="tier2" else 80+int((current/max(1,total))*20))
            progress_bar.progress(min(pct,100))
            status_text.markdown(f"<div style='color:#94a3b8;font-family:JetBrains Mono,monospace;font-size:13px;'>{message}</div>", unsafe_allow_html=True)

        results = run_full_scan(progress_callback=update_progress)
        st.session_state.last_scan_results = results
        progress_bar.progress(100)
        status_text.markdown(f"<div style='color:#22c55e;font-family:JetBrains Mono,monospace;font-size:13px;'>✅ Complete: {results['total_scanned']} scanned → {len(results['all_qualified'])} qualified</div>", unsafe_allow_html=True)

        history = merge_scan_results(results['all_qualified'], st.session_state.scan_history)
        st.session_state.scan_history = history; save_scan_history(history)

        # Update universe tracking after each scan
        if UNIVERSE_AVAILABLE and results.get('all_qualified'):
            try:
                update_universe_from_scan(results['all_qualified'])
            except Exception:
                pass

        if results['all_qualified']:
            title, message = format_scan_notification(results)
            if title: send_desktop_notification(title, message)
            st.session_state.play_sound = True

    if st.session_state.get('play_sound', False):
        st.markdown(get_alert_sound_html("success"), unsafe_allow_html=True)
        st.session_state.play_sound = False

    history = st.session_state.scan_history

    # ── Sector break banner ───────────────────────────────────────────────────
    last_results = st.session_state.get('last_scan_results') or {}
    sector_breaks = last_results.get('sector_breaks', [])
    for sb in sector_breaks:
        sb_color = '#22c55e' if sb['direction'] == 'LONG' else '#ef4444'
        sb_arrow = '↑' if sb['direction'] == 'LONG' else '↓'
        st.markdown(f"""
        <div style="background:{sb_color}0d;border:1px solid {sb_color}55;border-radius:10px;
            padding:10px 16px;margin:6px 0;display:flex;align-items:center;gap:10px;">
            <div style="font-size:18px;">🚨</div>
            <div>
                <span style="color:{sb_color};font-weight:700;font-family:'DM Mono',monospace;">
                    SECTOR BREAK — {sb['sector']} {sb_arrow}
                </span>
                <span style="color:#94a3b8;font-size:12px;margin-left:8px;">
                    {sb['count']} stocks: {', '.join(sb['tickers'][:6])}{'…' if sb['count']>6 else ''}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Universe panel ────────────────────────────────────────────────────────
    if UNIVERSE_AVAILABLE:
        ranked = get_ranked_universe(min_appearances=1)
        if ranked:
            with st.expander(f"📚 Universe Tracker — Top {len(ranked)} performers", expanded=False):
                u_cols = st.columns([2, 1, 1, 1, 1, 1])
                headers = ["Ticker", "Score", "A-Rate", "Conf%", "Appearances", "Sector"]
                for col, h in zip(u_cols, headers):
                    col.markdown(f"<div style='font-size:10px;color:#6b7280;font-family:\"DM Mono\",monospace;'>{h}</div>", unsafe_allow_html=True)
                for entry in ranked[:10]:
                    ec = st.columns([2, 1, 1, 1, 1, 1])
                    ec[0].markdown(f"**{entry['ticker']}**")
                    ec[1].markdown(f"{entry.get('score', 0):.0f}")
                    ec[2].markdown(f"{entry.get('a_grade_rate', 0)*100:.0f}%")
                    ec[3].markdown(f"{entry.get('avg_confidence', 0):.0f}")
                    ec[4].markdown(f"{entry.get('appearances', 0)}")
                    ec[5].markdown(f"{entry.get('sector', '—')}")

    try:
        from lib.trade_grader import get_session_context
        sess = get_session_context()
        sess_color = '#22c55e' if sess['quality']=='prime' else ('#f59e0b' if sess['quality'] in ['caution','good'] else ('#ef4444' if sess['quality'] in ['avoid','exit'] else '#6366f1'))
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid {sess_color}33;
            border-radius:12px;padding:16px 24px;margin:16px 0;display:flex;align-items:center;gap:20px;">
            <div style="font-size:28px;">{'🟢' if sess['quality']=='prime' else ('🟡' if sess['quality'] in ['caution','good'] else '🔴')}</div>
            <div>
                <div style="font-family:JetBrains Mono,monospace;font-size:16px;font-weight:700;color:{sess_color};">{sess['session']}</div>
                <div style="font-family:Inter,sans-serif;font-size:13px;color:#94a3b8;">{sess['note']}</div>
            </div>
            <div style="margin-left:auto;font-family:JetBrains Mono,monospace;font-size:13px;color:#64748b;">
                0DTE: {'✅' if sess['allow_0dte'] else '❌'} | Swing: {'✅' if sess['allow_swing'] else '❌'}
            </div>
        </div>""", unsafe_allow_html=True)
    except ImportError:
        pass

    if history:
        current  = [h for h in history if h.get('status')=='current']
        previous = [h for h in history if h.get('status')=='previous']

        st.markdown(f"<div class='section-header' style='margin-top:24px;'>🟢 Active Setups — {len(current)} Found</div>", unsafe_allow_html=True)

        if current:
            # ── Track Selected controls ───────────────────────────────────────
            if TRACKER_AVAILABLE:
                if 'track_selections' not in st.session_state:
                    st.session_state.track_selections = {}
                tc1, tc2 = st.columns([3, 1])
                with tc2:
                    selected_syms = [s for s, v in st.session_state.track_selections.items() if v]
                    if st.button(f"📌 Track Selected ({len(selected_syms)})",
                                  type="primary", use_container_width=True,
                                  disabled=len(selected_syms)==0):
                        for sym_t in selected_syms:
                            # Find the result for this symbol
                            r_match = next((r for r in current if r.get('ticker')==sym_t), None)
                            if r_match:
                                g_match = r_match.get('grade',{}) or {}
                                add_tracked(
                                    ticker=sym_t,
                                    grade=g_match.get('grade_intraday','B'),
                                    direction=r_match.get('direction','LONG'),
                                    regime=r_match.get('regime','Chop'),
                                    signal_score=r_match.get('composite',0),
                                    entry_price=float(r_match.get('close',0)),
                                    trade_type=g_match.get('trade_intraday','—'),
                                    patterns=r_match.get('patterns',[]),
                                )
                        st.session_state.track_selections = {}
                        st.success(f"📌 {len(selected_syms)} ticker(s) added to Tracker tab!")
                        st.rerun()

            for r in current[:10]:
                g        = r.get('grade', {}) or {}
                gi       = g.get('grade_intraday', '—')
                gs       = g.get('grade_swing', '—')
                ti       = g.get('trade_intraday', '—')
                ts_val   = g.get('trade_swing', '—')
                ri_val   = g.get('risk_0dte', '—')
                rs_val   = g.get('risk_swing', '—')
                ci       = g.get('contracts_0dte', 2)
                cs       = g.get('contracts_swing', 2)
                sym      = r.get('ticker', '')
                price    = r.get('close', 0)
                regime_r = r.get('regime', 'Chop')
                direction_r = r.get('direction', 'LONG')

                # ── Checkbox for multi-select tracking ──────────────────────
                if TRACKER_AVAILABLE:
                    chk_col, card_col = st.columns([0.3, 9.7])
                    with chk_col:
                        checked = st.checkbox("", key=f"chk_{sym}",
                                               value=st.session_state.track_selections.get(sym, False),
                                               label_visibility="collapsed")
                        st.session_state.track_selections[sym] = checked
                else:
                    card_col = st.container()

                with card_col:
                    arrow_color = '#4ade80' if direction_r=='LONG' else '#f87171'
                    arrow       = '▲' if direction_r=='LONG' else '▼'

                    pats_html = ""
                    for p in r.get('patterns', []):
                        gc2 = {'A+':'grade-aplus','A':'grade-a','B+':'grade-bplus'}.get(p.get('grade',''),'grade-b')
                        pats_html += f'<span class="strat-pill {gc2}" style="font-size:10px;padding:2px 8px;">{p["name"]}</span>'

                    ftc   = g.get('ftc', {}); ftc_txt = f"{ftc.get('aligned',0)}/{ftc.get('total',0)} {'✅' if ftc.get('ftc_confirmed') else '⚠️'}" if ftc else '—'
                    sec   = g.get('sector_corr', {}); sec_txt = f"{sec.get('sector','—')} {'✅' if sec.get('correlated') else '❌'}" if sec else '—'
                    atr_d = g.get('atr', {}); atr_txt = atr_d.get('move_potential', '—') if atr_d else '—'
                    atr_val = float(atr_d.get('atr_value', 0)) if atr_d else None

                    # Gap badge
                    gap_info = r.get('gap', {})
                    gap_type = gap_info.get('gap_type', 'none')
                    gap_pct  = gap_info.get('gap_pct', 0.0)
                    gap_badge = ''
                    if gap_type == 'strong_up':
                        gap_badge = f'<span style="background:#166534;color:#4ade80;border-radius:6px;padding:2px 7px;font-size:10px;font-family:\'DM Mono\',monospace;">⬆ GAP +{gap_pct:.1f}%</span>'
                    elif gap_type == 'moderate_up':
                        gap_badge = f'<span style="background:#1e3a1e;color:#86efac;border-radius:6px;padding:2px 7px;font-size:10px;font-family:\'DM Mono\',monospace;">↑ gap +{gap_pct:.1f}%</span>'
                    elif gap_type == 'strong_down':
                        gap_badge = f'<span style="background:#7f1d1d;color:#fca5a5;border-radius:6px;padding:2px 7px;font-size:10px;font-family:\'DM Mono\',monospace;">⬇ GAP {gap_pct:.1f}%</span>'
                    elif gap_type == 'moderate_down':
                        gap_badge = f'<span style="background:#3a1e1e;color:#fca5a5;border-radius:6px;padding:2px 7px;font-size:10px;font-family:\'DM Mono\',monospace;">↓ gap {gap_pct:.1f}%</span>'

                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#111827,#0f172a);border:1px solid rgba(99,102,241,0.15);
                        border-radius:14px;padding:16px 20px;margin:4px 0;">
                        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
                            <span style="font-size:22px;color:{arrow_color};">{arrow}</span>
                            <span style="font-family:'DM Mono',monospace;font-size:20px;font-weight:700;color:#e2e8f0;">{sym}</span>
                            <span style="color:#4b5563;font-family:'DM Mono',monospace;font-size:14px;">${price}</span>
                            <span class="regime-badge regime-{regime_r.lower().replace(' ','_')}" style="font-size:11px;padding:3px 10px;">{regime_r}</span>
                            <span style="color:#4b5563;font-size:12px;">{r.get('confidence',0)}% conf</span>
                            {gap_badge}
                            <span style="margin-left:auto;color:#374151;font-size:11px;">{r.get('scan_time','')}</span>
                        </div>
                        <div style="display:flex;gap:16px;margin-bottom:8px;">
                            <div style="flex:1;background:#07080d;border-radius:10px;padding:12px;text-align:center;">
                                <div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:1px;">Intraday 0DTE</div>
                                <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin:6px 0;">
                                    <span style="display:inline-block;padding:3px 12px;border-radius:12px;font-family:'DM Mono',monospace;font-size:14px;font-weight:700;{grade_css_style(gi)}">{gi}</span>
                                    <span style="font-family:'DM Mono',monospace;font-size:13px;color:{arrow_color};font-weight:600;">{ti}</span>
                                </div>
                                <div style="font-size:11px;color:#4b5563;">Risk: ${ri_val} | {ci} contracts</div>
                            </div>
                            <div style="flex:1;background:#07080d;border-radius:10px;padding:12px;text-align:center;">
                                <div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:1px;">Swing Trade</div>
                                <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin:6px 0;">
                                    <span style="display:inline-block;padding:3px 12px;border-radius:12px;font-family:'DM Mono',monospace;font-size:14px;font-weight:700;{grade_css_style(gs)}">{gs}</span>
                                    <span style="font-family:'DM Mono',monospace;font-size:13px;color:{arrow_color};font-weight:600;">{ts_val}</span>
                                </div>
                                <div style="font-size:11px;color:#4b5563;">Risk: ${rs_val} | {cs} contracts</div>
                            </div>
                            <div style="flex:1;background:#07080d;border-radius:10px;padding:12px;">
                                <div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:1px;">Confirmation</div>
                                <div style="font-size:12px;color:#94a3b8;margin-top:6px;font-family:'DM Mono',monospace;">
                                    FTC: {ftc_txt}<br>Sector: {sec_txt}<br>ATR: {atr_txt}
                                </div>
                            </div>
                        </div>
                        <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:4px;">{pats_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Options contract estimate ──────────────────────────────
                    if TRACKER_AVAILABLE and is_executable_grade(gi):
                        opt   = estimate_option_contract(sym, direction_r, float(price),
                                    '0DTE' if '0DTE' in str(ti) else 'swing', atr_val)
                        c_clr = '#4ade80' if opt['c_or_p']=='CALL' else '#f87171'
                        st.markdown(f"""
                        <div style="background:#070810;border:1px solid rgba(99,102,241,0.07);border-radius:10px;
                            padding:10px 16px;margin:2px 0 6px 0;font-family:'DM Mono',monospace;font-size:12px;
                            display:flex;gap:20px;flex-wrap:wrap;align-items:center;">
                            <span style="font-size:10px;color:#374151;text-transform:uppercase;letter-spacing:1.5px;">📋 Option Est.</span>
                            <span style="color:#e2e8f0;">${opt['strike']} Strike</span>
                            <span style="color:{c_clr};font-weight:600;">{opt['c_or_p']}</span>
                            <span style="color:#94a3b8;">Exp: {opt['expiration']}</span>
                            <span style="color:#fcd34d;">~${opt['est_premium']}/sh · ~${opt['est_cost_1x']}/contract</span>
                            <span style="color:#1f2937;font-size:10px;">formula est. · live API coming</span>
                        </div>
                        """, unsafe_allow_html=True)

                # ── Execute buttons for A+/A grades ──────────────────────────
                if EXECUTOR_AVAILABLE:
                    ecols = st.columns([1, 1, 2])
                    with ecols[0]:
                        if is_executable_grade(gi):
                            btn_label = f"⚡ 0DTE {direction_r} {sym}"
                            if st.button(btn_label, key=f"exec_0dte_{sym}", type="primary", use_container_width=True):
                                st.session_state[f"confirm_0dte_{sym}"] = True
                    with ecols[1]:
                        if is_executable_grade(gs):
                            btn_label = f"📊 SWING {direction_r} {sym}"
                            if st.button(btn_label, key=f"exec_swing_{sym}", use_container_width=True):
                                st.session_state[f"confirm_swing_{sym}"] = True

                    # 0DTE confirm
                    if st.session_state.get(f"confirm_0dte_{sym}"):
                        st.warning(f"⚠️ Confirm 0DTE paper trade: {direction_r} {sym} @ ~${price} | Grade: {gi}")
                        cy, cn = st.columns(2)
                        with cy:
                            if st.button(f"✅ CONFIRM 0DTE {sym}", key=f"yes_0dte_{sym}", type="primary"):
                                exe = get_executor()
                                result = exe.submit_equity_order(
                                    symbol=sym, side='buy' if direction_r=='LONG' else 'sell',
                                    price=float(price), regime=regime_r,
                                    trade_type='0DTE', grade=gi,
                                )
                                if result:
                                    log_order(result)
                                    st.success(f"✅ 0DTE order fired! ID: {result['order_id'][:8]}...")
                                else:
                                    st.error("❌ Order blocked by session/regime rules.")
                                st.session_state[f"confirm_0dte_{sym}"] = False
                        with cn:
                            if st.button(f"❌ Cancel", key=f"cancel_0dte_{sym}"):
                                st.session_state[f"confirm_0dte_{sym}"] = False

                    # Swing confirm
                    if st.session_state.get(f"confirm_swing_{sym}"):
                        st.warning(f"⚠️ Confirm SWING paper trade: {direction_r} {sym} @ ~${price} | Grade: {gs}")
                        cy2, cn2 = st.columns(2)
                        with cy2:
                            if st.button(f"✅ CONFIRM SWING {sym}", key=f"yes_swing_{sym}", type="primary"):
                                exe = get_executor()
                                result = exe.submit_equity_order(
                                    symbol=sym, side='buy' if direction_r=='LONG' else 'sell',
                                    price=float(price), regime=regime_r,
                                    trade_type='swing', grade=gs,
                                )
                                if result:
                                    log_order(result)
                                    st.success(f"✅ Swing order fired! ID: {result['order_id'][:8]}...")
                                else:
                                    st.error("❌ Order blocked by session/regime rules.")
                                st.session_state[f"confirm_swing_{sym}"] = False
                        with cn2:
                            if st.button(f"❌ Cancel", key=f"cancel_swing_{sym}"):
                                st.session_state[f"confirm_swing_{sym}"] = False

                with st.expander(f"📋 {sym} — Signal Reasoning", expanded=False):
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.markdown("**Intraday Reasons:**")
                        for reason in g.get('reasons_intraday', []): st.markdown(f"- {reason}")
                    with col_r2:
                        st.markdown("**Swing Reasons:**")
                        for reason in g.get('reasons_swing', []): st.markdown(f"- {reason}")
                    if g.get('flags'):
                        st.markdown("**⚠️ Flags:**")
                        for flag in g['flags']: st.markdown(f"- {flag}")
        else:
            st.markdown("<div style='color:#64748b;padding:20px;text-align:center;font-family:Inter,sans-serif;'>No active setups. Run a scan or wait for the next scheduled window.</div>", unsafe_allow_html=True)

        if previous:
            st.markdown(f"<div class='section-header' style='margin-top:24px;'>⬜ Earlier Today — {len(previous)} Previous Hits</div>", unsafe_allow_html=True)
            for r in previous[:15]:
                ac = '#22c55e88' if r.get('direction')=='LONG' else '#ef444488'
                ar = '▲' if r.get('direction')=='LONG' else '▼'
                g  = r.get('grade', {}) or {}
                st.markdown(f"""
                <div class="watchlist-row watchlist-previous">
                    <div style="width:50px;font-size:18px;color:{ac};">{ar}</div>
                    <div style="width:70px;font-weight:600;color:#94a3b8;font-size:14px;">{r.get('ticker','')}</div>
                    <div style="width:70px;color:#64748b;">${r.get('close','')}</div>
                    <div style="width:100px;color:#64748b;">{r.get('regime','')}</div>
                    <div style="width:50px;color:#64748b;">{g.get('grade_intraday','—')}</div>
                    <div style="width:50px;color:#64748b;">{g.get('grade_swing','—')}</div>
                    <div style="width:60px;color:#64748b;">{r.get('composite',0):+d}</div>
                    <div style="flex:1;"></div>
                    <div style="width:110px;color:#475569;font-size:11px;">First: {r.get('first_seen','')}<br>Last: {r.get('last_seen','')}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;color:#64748b;font-family:Inter,sans-serif;'>
            <div style='font-size:48px;margin-bottom:16px;'>🎯</div>
            <div style='font-size:18px;font-weight:600;color:#94a3b8;margin-bottom:8px;'>No scan data yet</div>
            <div style='font-size:14px;'>Click "Run Full Scan" to screen the S&P 500</div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.get('last_scan_results'):
        r = st.session_state.last_scan_results
        st.markdown("<div class='section-header' style='margin-top:24px;'>Last Scan Summary</div>", unsafe_allow_html=True)
        mc    = st.columns(5)
        stats = [
            ("Universe",      str(r['total_scanned']),           "",                                           "tickers"),
            ("Tier 1 Pass",   str(r['tier1_count']),             "",                                           "candidates"),
            ("Tier 2 Analyzed",str(r['tier2_count']),            "",                                           "full HMM"),
            ("Qualified",     str(len(r['all_qualified'])),      "metric-positive" if r['all_qualified'] else "","85%+ & 30+"),
            ("Scan Time",     r['scan_time'],                    "",                                           ""),
        ]
        for col, (label, val, css, sub) in zip(mc, stats):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {css}">{val}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

    auto_scan_on, _ = should_auto_scan()
    if auto_scan_on:
        time.sleep(3); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE TRADING — Positions + Account
# ══════════════════════════════════════════════════════════════════════════════

elif page == "live":
    show_tab_header("live", "💼 Live Trading", datetime.now().strftime('%B %d, %Y %H:%M'))

    if not EXECUTOR_AVAILABLE:
        st.error("⚠️ Alpaca executor not connected. Check your config.py for ALPACA_API_KEY and ALPACA_SECRET_KEY.")
        st.stop()

    render_account_bar()

    exe = get_executor()
    if exe is None:
        st.error(f"Could not connect to Alpaca: {st.session_state.get('_executor_error', 'Unknown error')}")
        st.stop()

    # ── Kill Switch ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="kill-switch-banner">
        <div style="font-family:JetBrains Mono,monospace;font-size:13px;color:#ef4444;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">⚠️ Emergency Controls</div>
        <div style="font-family:Inter,sans-serif;font-size:13px;color:#fca5a5;">Kill switch closes ALL open positions immediately. Use only in emergencies.</div>
    </div>
    """, unsafe_allow_html=True)

    ks_col1, ks_col2 = st.columns([1, 3])
    with ks_col1:
        if st.button("🚨 KILL SWITCH — CLOSE ALL", type="primary", use_container_width=True):
            st.session_state['confirm_kill'] = True
    if st.session_state.get('confirm_kill'):
        st.error("**ARE YOU SURE?** This will close ALL open positions immediately.")
        kc1, kc2 = st.columns(2)
        with kc1:
            if st.button("⚠️ YES — CLOSE EVERYTHING", type="primary"):
                count = exe.kill_switch_0dte()
                st.success(f"✅ Kill switch executed — {count} position(s) closed.")
                st.session_state['confirm_kill'] = False
                st.session_state['_acct_cache_time'] = 0  # Force refresh
        with kc2:
            if st.button("❌ Cancel"):
                st.session_state['confirm_kill'] = False

    st.markdown("<div class='section-header' style='margin-top:24px;'>Open Positions</div>", unsafe_allow_html=True)

    positions = exe.get_open_positions()

    if positions:
        # Summary row
        total_pnl   = sum(p['unrealized_pnl'] for p in positions)
        pnl_color   = '#22c55e' if total_pnl >= 0 else '#ef4444'
        pnl_sign    = '+' if total_pnl >= 0 else ''
        st.markdown(f"""
        <div style="background:#0f1119;border-radius:12px;padding:14px 20px;margin-bottom:16px;
            border:1px solid rgba(255,255,255,0.05);display:flex;gap:32px;align-items:center;">
            <div>
                <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;">Open Positions</div>
                <div style="font-family:JetBrains Mono,monospace;font-size:22px;font-weight:700;color:#e2e8f0;">{len(positions)}</div>
            </div>
            <div>
                <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;">Unrealized P&L</div>
                <div style="font-family:JetBrains Mono,monospace;font-size:22px;font-weight:700;color:{pnl_color};">{pnl_sign}${total_pnl:,.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for p in positions:
            pnl      = p['unrealized_pnl']
            pnl_pct  = p['pnl_pct']
            is_profit = pnl >= 0
            clr      = '#22c55e' if is_profit else '#ef4444'
            sign     = '+' if is_profit else ''
            side_str = str(p.get('side', '')).upper()
            border   = 'position-profit' if is_profit else 'position-loss'

            st.markdown(f"""
            <div class="position-row {border}">
                <div style="display:flex;align-items:center;gap:20px;">
                    <div>
                        <div style="font-size:20px;font-weight:700;color:#e2e8f0;">{p['symbol']}</div>
                        <div style="font-size:12px;color:#64748b;">{side_str} | {p['qty']:.0f} shares</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Entry</div>
                        <div style="font-size:16px;color:#94a3b8;">${p['entry']:.2f}</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Current</div>
                        <div style="font-size:16px;color:#e2e8f0;">${p['current']:.2f}</div>
                    </div>
                    <div style="text-align:center;margin-left:auto;">
                        <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">P&L</div>
                        <div style="font-size:20px;font-weight:700;color:{clr};">{sign}${pnl:,.2f}</div>
                        <div style="font-size:12px;color:{clr};">{sign}{pnl_pct:.2f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Per-position close button
            if st.button(f"Close {p['symbol']}", key=f"close_{p['symbol']}", use_container_width=False):
                if exe.close_position(p['symbol']):
                    st.success(f"✅ {p['symbol']} position closed.")
                    st.session_state['_acct_cache_time'] = 0
                    st.rerun()
                else:
                    st.error(f"❌ Failed to close {p['symbol']}")
    else:
        st.markdown("""
        <div style='text-align:center;padding:50px 20px;'>
            <div style='font-size:40px;'>💤</div>
            <div style='color:#64748b;font-family:Inter,sans-serif;font-size:15px;margin-top:12px;'>No open positions</div>
        </div>""", unsafe_allow_html=True)

    # ── P&L Equity Curve ──────────────────────────────────────────────────────
    if st.session_state.get('order_log'):
        st.markdown("<div class='section-header' style='margin-top:24px;'>Session Equity Curve</div>", unsafe_allow_html=True)
        orders = st.session_state.order_log
        timestamps = [o.get('submitted_at','') for o in orders]
        prices_list = [o.get('price', 0) for o in orders]

        if len(prices_list) > 1:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=list(range(len(prices_list))), y=prices_list,
                mode='lines+markers',
                line=dict(color='#6366f1', width=2),
                marker=dict(size=8, color='#6366f1'),
            ))
            fig2.update_layout(
                template='plotly_dark', paper_bgcolor='#0a0b0f', plot_bgcolor='#0a0b0f',
                height=250, margin=dict(l=40, r=20, t=20, b=30),
                font=dict(family='Inter', size=12, color='#94a3b8'),
                xaxis_title="Order #", yaxis_title="Execution Price", showlegend=False,
            )
            fig2.update_xaxes(gridcolor='rgba(255,255,255,0.03)')
            fig2.update_yaxes(gridcolor='rgba(255,255,255,0.03)')
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    # Auto-refresh
    if st.button("🔄 Refresh", use_container_width=False):
        st.session_state['_acct_cache_time'] = 0
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WATCHLIST TRACKER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "tracker":
    show_tab_header("tracker", "📌 Watchlist Tracker", datetime.now().strftime('%B %d, %Y'))

    if not TRACKER_AVAILABLE:
        st.error("tracker.py not found in Trading Realm folder.")
        st.stop()

    # ── On-demand ticker analysis ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔍 Analyze Any Ticker</div>", unsafe_allow_html=True)
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        custom_ticker = st.text_input("Enter any ticker", placeholder="e.g. AMZN, MSFT, AMD...",
                                       label_visibility="collapsed").upper().strip()
    with col_btn:
        run_custom = st.button("⚡ Analyze", type="primary", use_container_width=True)

    if run_custom and custom_ticker:
        with st.spinner(f"Running full analysis on {custom_ticker}..."):
            cdata = run_single_analysis(custom_ticker, days=730)
        if cdata:
            csig = cdata['signal']; csnap = cdata['snap']; creg = cdata['regime']
            sc   = 'signal-long' if csig['direction']=='LONG' else ('signal-short' if csig['direction']=='SHORT' else 'signal-flat')
            rc   = creg.lower().replace(' ','_')
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#111827,#0f172a);border:1px solid rgba(99,102,241,0.2);
                border-radius:14px;padding:20px;margin:12px 0;">
                <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
                    <div style="font-family:'DM Mono',monospace;font-size:24px;font-weight:700;color:#e2e8f0;">{custom_ticker}</div>
                    <div class="signal-card {sc}" style="padding:12px 20px;flex:1;min-width:160px;">
                        <div class="signal-label">Signal</div>
                        <div class="signal-value" style="font-size:20px;">{csig['trade_type']}</div>
                        <div class="signal-sub">{csig['strength']} | Score: {csig['composite']:+d}</div>
                    </div>
                    <div style="flex:1;min-width:160px;">
                        <span class="regime-badge regime-{rc}">{creg}</span>
                        <div style="font-size:12px;color:#64748b;margin-top:6px;">Conf: {csig['confidence']}%</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:11px;color:#4b5563;text-transform:uppercase;letter-spacing:1.5px;">Price</div>
                        <div style="font-family:'DM Mono',monospace;font-size:20px;color:#e2e8f0;">${csnap['close']:.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Option estimate
            opt = estimate_option_contract(
                custom_ticker, csig['direction'], csnap['close'],
                '0DTE' if '0DTE' in csig['trade_type'] else 'swing'
            )
            st.markdown(f"""
            <div style="background:#0b0c14;border:1px solid rgba(99,102,241,0.1);border-radius:12px;
                padding:14px 20px;margin:8px 0;font-family:'DM Mono',monospace;font-size:13px;">
                <div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">
                    📋 Option Contract Estimate
                </div>
                <div style="display:flex;gap:24px;flex-wrap:wrap;color:#94a3b8;">
                    <span>Strike: <b style="color:#e2e8f0;">${opt['strike']}</b></span>
                    <span>Type: <b style="color:{'#4ade80' if opt['c_or_p']=='CALL' else '#f87171'};">{opt['c_or_p']}</b></span>
                    <span>Expiry: <b style="color:#e2e8f0;">{opt['expiration']}</b></span>
                    <span>Est. Premium: <b style="color:#fcd34d;">${opt['est_premium']}</b>/share</span>
                    <span>Est. Cost (1 contract): <b style="color:#fcd34d;">${opt['est_cost_1x']}</b></span>
                </div>
                <div style="font-size:10px;color:#374151;margin-top:8px;">{opt['note']}</div>
            </div>
            """, unsafe_allow_html=True)

            fig_custom = build_candlestick_chart(cdata['df'], custom_ticker, show_days=60)
            st.plotly_chart(fig_custom, use_container_width=True, config={'displayModeBar': False})
        else:
            st.error(f"No data found for {custom_ticker}")

    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

    # ── Active tracked tickers ────────────────────────────────────────────────
    st.markdown("<div class='section-header' style='margin-top:8px;'>📌 Tracked Positions</div>", unsafe_allow_html=True)

    # Refresh prices button
    col_r1, col_r2 = st.columns([2, 1])
    with col_r2:
        if st.button("🔄 Refresh Prices", use_container_width=True):
            tracked = load_tracked()
            tracked = update_tracked_prices(tracked)
            st.session_state['tracked_cache'] = tracked
            st.rerun()

    tracked = st.session_state.get('tracked_cache') or load_tracked()

    if tracked:
        for t in tracked:
            if not t.get('active', True):
                continue
            pnl = t.get('pnl_pct', 0)
            pnl_d = t.get('pnl_dollars', 0)
            pnl_color = '#4ade80' if pnl >= 0 else '#f87171'
            sign = '+' if pnl >= 0 else ''
            dir_color = '#4ade80' if t.get('direction','LONG')=='LONG' else '#f87171'
            arrow = '▲' if t.get('direction','LONG')=='LONG' else '▼'
            grade_s = grade_css_style(t.get('grade','B'))
            border_class = 'position-profit' if pnl >= 0 else 'position-loss'

            st.markdown(f"""
            <div class="position-row {border_class}" style="margin:8px 0;">
                <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
                    <div>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="font-size:20px;font-weight:700;color:#e2e8f0;">{t['ticker']}</span>
                            <span style="font-size:16px;color:{dir_color};">{arrow}</span>
                            <span style="display:inline-block;padding:2px 10px;border-radius:100px;font-size:12px;{grade_s}">{t.get('grade','')}</span>
                        </div>
                        <div style="font-size:11px;color:#4b5563;margin-top:2px;">{t.get('trade_type','')} | {t.get('regime','')} | Added {t.get('added_at','')[:10]}</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:1px;">Entry</div>
                        <div style="font-family:'DM Mono',monospace;font-size:16px;color:#94a3b8;">${t.get('entry_price',0):.2f}</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:1px;">Current</div>
                        <div style="font-family:'DM Mono',monospace;font-size:16px;color:#e2e8f0;">${t.get('current_price',0):.2f}</div>
                    </div>
                    <div style="text-align:center;margin-left:auto;">
                        <div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:1px;">P&L</div>
                        <div style="font-family:'DM Mono',monospace;font-size:20px;font-weight:600;color:{pnl_color};">{sign}{pnl:.2f}%</div>
                        <div style="font-size:12px;color:{pnl_color};">{sign}${abs(pnl_d):.2f}/share</div>
                    </div>
                </div>
                {'<div style="margin-top:10px;font-size:12px;color:#64748b;font-style:italic;">📝 ' + t['note'] + '</div>' if t.get('note') else ''}
            </div>
            """, unsafe_allow_html=True)

            # Note edit + remove
            note_col, rm_col = st.columns([4, 1])
            with note_col:
                new_note = st.text_input(f"Note for {t['ticker']}", value=t.get('note',''),
                                          key=f"note_{t['ticker']}", label_visibility="collapsed",
                                          placeholder=f"Add note for {t['ticker']}...")
                if new_note != t.get('note',''):
                    t['note'] = new_note
                    save_tracked(tracked)
            with rm_col:
                if st.button(f"✕ Remove", key=f"rm_{t['ticker']}", use_container_width=True):
                    remove_tracked(t['ticker'])
                    st.session_state['tracked_cache'] = None
                    st.rerun()

        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All Tracked", use_container_width=False):
            clear_all_tracked()
            st.session_state['tracked_cache'] = None
            st.rerun()
    else:
        st.markdown("""
        <div style='text-align:center;padding:40px 20px;'>
            <div style='font-size:36px;'>📌</div>
            <div style='color:#4b5563;font-family:DM Sans,sans-serif;font-size:14px;margin-top:10px;'>
                No tickers tracked yet.<br>Go to the Scanner tab and check the boxes on A+/A setups.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── P&L Stats ─────────────────────────────────────────────────────────────
    if PNL_AVAILABLE:
        st.markdown("<div class='section-header' style='margin-top:24px;'>📊 Performance Stats</div>", unsafe_allow_html=True)

        # Period switcher
        period_map = {"Today": "today", "This Week": "week", "This Month": "month", "All Time": "all"}
        selected_period = st.radio("Period", list(period_map.keys()), horizontal=True,
                                    label_visibility="collapsed")
        period_key = period_map[selected_period]
        stats = get_stats_for_period(period_key)

        if stats['total_trades'] == 0:
            st.markdown(f"<div style='color:#4b5563;font-family:DM Sans,sans-serif;font-size:13px;padding:16px 0;'>No trades logged for {selected_period.lower()}.</div>", unsafe_allow_html=True)
        else:
            # Top metrics row
            mc = st.columns(6)
            streak_color = '#4ade80' if stats['streak_type']=='win' else '#f87171'
            streak_icon  = '🔥' if stats['streak_type']=='win' else '❄️'
            for col, (label, val, css, sub) in zip(mc, [
                ("Win Rate",     f"{stats['win_rate']}%",        "metric-positive" if stats['win_rate']>50 else "metric-negative", f"{stats['win_count']}W / {stats['loss_count']}L"),
                ("Profit Factor",f"{stats['profit_factor']}",    "metric-positive" if stats['profit_factor']>1.5 else "",          ""),
                ("Avg Win",      f"{stats['avg_win']:+.2f}%",    "metric-positive",                                                 "per trade"),
                ("Avg Loss",     f"{stats['avg_loss']:.2f}%",    "metric-negative",                                                 "per trade"),
                ("Total P&L",    f"{stats['total_pnl_pct']:+.2f}%","metric-positive" if stats['total_pnl_pct']>0 else "metric-negative",""),
                ("Streak",       f"{streak_icon} {stats['current_streak']}",streak_color if stats['current_streak']>2 else "",     str(stats['streak_type']).capitalize() if stats['streak_type'] else ""),
            ]):
                with col:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {css}">{val}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

            # Second row
            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
            mc2 = st.columns(4)
            for col, (label, val, css) in zip(mc2, [
                ("Best Ticker",  f"{stats['best_ticker']} ({stats['best_ticker_pnl']:+.2f}%)",  "metric-positive"),
                ("Worst Ticker", f"{stats['worst_ticker']} ({stats['worst_ticker_pnl']:+.2f}%)", "metric-negative"),
                ("Best Time",    stats['best_hour'],   "metric-positive"),
                ("Worst Time",   stats['worst_hour'],  "metric-negative"),
            ]):
                with col:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {css}" style="font-size:18px;">{val}</div></div>', unsafe_allow_html=True)

            # Regime breakdown
            if stats['regime_breakdown']:
                st.markdown("<div class='section-header' style='margin-top:20px;'>Performance by Regime</div>", unsafe_allow_html=True)
                for regime_name, rs in sorted(stats['regime_breakdown'].items(), key=lambda x: -x[1]['win_rate']):
                    wr_c = '#4ade80' if rs['win_rate']>50 else '#f87171'
                    avg_c = '#4ade80' if rs['avg_pnl']>0 else '#f87171'
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:16px;padding:10px 16px;margin:3px 0;
                        background:#0b0c14;border-radius:10px;font-family:'DM Mono',monospace;font-size:13px;
                        border:1px solid rgba(255,255,255,0.04);">
                        <div style="width:130px;color:#94a3b8;font-weight:600;">{regime_name}</div>
                        <div style="width:70px;color:#4b5563;">{rs['count']} trades</div>
                        <div style="width:100px;">WR: <span style="color:{wr_c};font-weight:600;">{rs['win_rate']}%</span></div>
                        <div style="width:120px;">Avg: <span style="color:{avg_c};">{rs['avg_pnl']:+.3f}%</span></div>
                        <div style="flex:1;background:rgba(255,255,255,0.04);border-radius:4px;height:12px;">
                            <div style="width:{min(rs['win_rate'],100)}%;background:{wr_c};height:100%;border-radius:4px;opacity:0.5;"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Clear P&L History"):
            clear_pnl_history()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ALERTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "orders":
    show_tab_header("orders", "🚨 Alerts", datetime.now().strftime('%B %d, %Y'))

    if not ALERTS_AVAILABLE:
        st.error("alerts.py not found in Trading Realm folder.")
        st.stop()

    alerts = load_alerts()

    # ── Show pending banner if on this tab ────────────────────────────────────
    pending = get_pending_banner_alerts()
    for a in pending[:1]:  # Show one banner at a time
        st.markdown(get_alert_banner_html(a, banner_duration_sec=77), unsafe_allow_html=True)
        mark_banner_shown(a['id'])

    # ── Summary stats ─────────────────────────────────────────────────────────
    total_alerts  = len(alerts)
    unread        = len([a for a in alerts if not a.get('acknowledged')])
    acknowledged  = len([a for a in alerts if a.get('acknowledged')])

    mc = st.columns(4)
    for col, (label, val, css) in zip(mc, [
        ("Total Alerts",    str(total_alerts), ""),
        ("Unacknowledged",  str(unread),       "metric-negative" if unread else ""),
        ("Acknowledged",    str(acknowledged), "metric-positive" if acknowledged else ""),
        ("72hr Auto-Expire","Active",          ""),
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {css}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    col_a1, col_a2 = st.columns([1, 4])
    with col_a1:
        if st.button("✅ Acknowledge All", use_container_width=True):
            for a in alerts:
                acknowledge_alert(a['id'])
            st.rerun()

    # ── Alert log ─────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header' style='margin-top:16px;'>Previous Alerts (Last 72 Hours)</div>", unsafe_allow_html=True)

    if alerts:
        for a in alerts:
            is_ack     = a.get('acknowledged', False)
            grade      = a.get('grade', 'A')
            direction  = a.get('direction', 'LONG')
            dir_color  = '#4ade80' if direction=='LONG' else '#f87171'
            dir_arrow  = '▲' if direction=='LONG' else '▼'
            grade_s    = grade_css_style(grade)
            time_left  = get_time_remaining(a)
            ack_style  = "opacity:0.45;" if is_ack else ""
            ts         = a.get('timestamp','')[:19].replace('T',' ')

            st.markdown(f"""
            <div style="background:#0b0c14;border:1px solid {'rgba(255,255,255,0.03)' if is_ack else 'rgba(99,102,241,0.15)'};
                border-radius:12px;padding:16px 20px;margin:6px 0;{ack_style}
                {'border-left:3px solid #4ade80;' if is_ack else 'border-left:3px solid #6366f1;'}">
                <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                    <span style="display:inline-block;padding:2px 10px;border-radius:100px;font-size:12px;{grade_s}">{grade}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:18px;font-weight:700;color:#e2e8f0;">{a.get('ticker','')}</span>
                    <span style="font-size:16px;color:{dir_color};">{dir_arrow} {direction}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:13px;color:#64748b;">{a.get('trade_type','')}</span>
                    <span style="color:#64748b;font-size:12px;">${a.get('price',0):.2f}</span>
                    <span style="color:#374151;font-size:11px;font-family:'DM Mono',monospace;">{a.get('regime','')}</span>
                    <div style="margin-left:auto;text-align:right;">
                        <div style="font-size:11px;color:#374151;">{ts}</div>
                        <div style="font-size:10px;color:#1f2937;">Expires: {time_left}</div>
                    </div>
                    {'<span style="color:#4ade80;font-size:11px;">✓ Acknowledged</span>' if is_ack else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if not is_ack:
                if st.button(f"✅ Acknowledge {a.get('ticker','')}", key=f"ack_{a['id']}"):
                    acknowledge_alert(a['id'])
                    st.rerun()
    else:
        st.markdown("""
        <div style='text-align:center;padding:50px 20px;'>
            <div style='font-size:40px;'>🔕</div>
            <div style='color:#4b5563;font-family:DM Sans,sans-serif;font-size:14px;margin-top:10px;'>
                No alerts in the last 72 hours.
            </div>
        </div>""", unsafe_allow_html=True)

    if st.button("🗑️ Clear All Alerts"):
        clear_all_alerts()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ORDER LOG (renamed from orders)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "orderlog":
    show_tab_header("orders", "📜 Order Log", datetime.now().strftime('%B %d, %Y'))

    if not EXECUTOR_AVAILABLE:
        st.error("⚠️ Alpaca executor not connected.")
        st.stop()

    render_account_bar()

    exe = get_executor()

    # ── Today's orders from Alpaca ────────────────────────────────────────────
    st.markdown("<div class='section-header'>Today's Orders (from Alpaca)</div>", unsafe_allow_html=True)
    live_orders = exe.get_todays_orders() if exe else []

    if live_orders:
        # Summary
        filled  = [o for o in live_orders if 'filled' in str(o.get('status',''))]
        pending = [o for o in live_orders if 'new' in str(o.get('status','')) or 'open' in str(o.get('status',''))]
        errors  = [o for o in live_orders if 'canceled' in str(o.get('status','')) or 'rejected' in str(o.get('status',''))]

        sm = st.columns(4)
        for col, (label, val, css) in zip(sm, [
            ("Total Orders", len(live_orders), ""),
            ("Filled",   len(filled),  "metric-positive" if filled else ""),
            ("Pending",  len(pending), ""),
            ("Canceled", len(errors),  "metric-negative" if errors else ""),
        ]):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {css}">{val}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

        for o in live_orders:
            status_str = str(o.get('status', ''))
            is_filled  = 'filled' in status_str
            is_pending = 'new' in status_str or 'open' in status_str
            css_class  = 'order-filled' if is_filled else ('order-pending' if is_pending else 'order-error')
            status_color = '#22c55e' if is_filled else ('#f59e0b' if is_pending else '#ef4444')
            side_str = str(o.get('side', '')).replace('OrderSide.', '').upper()
            side_color = '#22c55e' if side_str == 'BUY' else '#ef4444'
            filled_price = f"${o['filled_avg_price']:.2f}" if o.get('filled_avg_price') else '—'

            st.markdown(f"""
            <div class="order-row {css_class}">
                <div style="width:100px;color:#e2e8f0;font-weight:700;font-size:15px;">{o.get('symbol','')}</div>
                <div style="width:60px;color:{side_color};font-weight:600;">{side_str}</div>
                <div style="width:60px;color:#94a3b8;">{o.get('qty',0):.0f} sh</div>
                <div style="width:100px;color:#94a3b8;">Fill: {filled_price}</div>
                <div style="flex:1;"></div>
                <div style="width:100px;color:{status_color};font-size:12px;text-align:right;">{status_str.replace('OrderStatus.','').upper()}</div>
                <div style="width:80px;color:#475569;font-size:11px;text-align:right;">{o.get('id','')[:8]}...</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#64748b;padding:20px;text-align:center;font-family:Inter,sans-serif;'>No orders placed today.</div>", unsafe_allow_html=True)

    # ── Session order log (orders fired through this dashboard) ───────────────
    session_log = st.session_state.get('order_log', [])
    if session_log:
        st.markdown("<div class='section-header' style='margin-top:32px;'>This Session — Fired via Dashboard</div>", unsafe_allow_html=True)

        # Daily P&L from session log
        session_pnl = sum(
            (o.get('target', 0) - o.get('price', 0)) * o.get('qty', 0)
            for o in session_log if o.get('side','') == 'buy'
        )
        st.markdown(f"""
        <div style="background:#0f1119;border-radius:12px;padding:14px 20px;margin-bottom:16px;
            border:1px solid rgba(255,255,255,0.05);">
            <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;">Session Orders</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:20px;font-weight:700;color:#e2e8f0;margin-top:4px;">
                {len(session_log)} orders fired
            </div>
        </div>""", unsafe_allow_html=True)

        for o in session_log:
            side_c  = '#22c55e' if o.get('side','')=='buy' else '#ef4444'
            grade_s = grade_css_style(o.get('grade',''))
            tt_str  = o.get('trade_type', '—')
            submitted = o.get('submitted_at', '')[:19] if o.get('submitted_at') else '—'

            st.markdown(f"""
            <div class="order-row order-filled">
                <div style="width:100px;color:#e2e8f0;font-weight:700;font-size:15px;">{o.get('symbol','')}</div>
                <div style="width:60px;color:{side_c};font-weight:600;">{o.get('side','').upper()}</div>
                <div style="width:60px;color:#94a3b8;">{o.get('qty',0)} sh</div>
                <div style="width:90px;color:#94a3b8;">${o.get('price',0):.2f}</div>
                <div style="width:70px;color:#64748b;font-size:11px;">{tt_str}</div>
                <span style="display:inline-block;padding:2px 10px;border-radius:10px;font-size:12px;font-family:JetBrains Mono,monospace;{grade_s}">{o.get('grade','')}</span>
                <div style="width:80px;color:#64748b;font-size:11px;">{o.get('regime','')}</div>
                <div style="flex:1;"></div>
                <div style="color:#475569;font-size:11px;">{submitted}</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Session Log"):
            st.session_state.order_log = []
            st.rerun()

    if st.button("🔄 Refresh Orders"):
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

elif page == "backtest":
    show_tab_header("backtest", "📈 Walk-Forward Backtest", ticker)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid rgba(99,102,241,0.15);
        border-radius:12px;padding:20px;margin-bottom:20px;">
        <div style="font-family:Inter,sans-serif;font-size:14px;color:#94a3b8;line-height:1.7;">
            <span style="color:#6366f1;font-weight:600;">Walk-Forward Analysis</span> exposes curve-fitting by splitting data into rolling Train/Test windows.
            <br><br>
            <span style="color:#e2e8f0;">1.</span> Trains on 1-year window — optimizes RSI/ADX for best Sharpe<br>
            <span style="color:#e2e8f0;">2.</span> Tests <span style="color:#f59e0b;font-weight:600;">blindly</span> on next window — optimizer has never seen this data<br>
            <span style="color:#e2e8f0;">3.</span> Slides forward and repeats — stitches only blind test results
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;gap:16px;margin-bottom:20px;">
        <div class="metric-card" style="flex:1;"><div class="metric-label">Exchange Fee</div><div class="metric-value" style="font-size:22px;">0.10%</div><div class="metric-sub">per trade</div></div>
        <div class="metric-card" style="flex:1;"><div class="metric-label">Slippage</div><div class="metric-value" style="font-size:22px;">0.05%</div><div class="metric-sub">per trade</div></div>
        <div class="metric-card" style="flex:1;"><div class="metric-label">Total Round Trip</div><div class="metric-value" style="font-size:22px;">0.30%</div><div class="metric-sub">baked in</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_intra, col_swing = st.columns(2)
    with col_intra:
        run_intra = st.button("⚡ Run Intraday WF (3mo blind test)", type="primary", use_container_width=True)
    with col_swing:
        run_swing = st.button("📊 Run Swing WF (18mo blind test)", type="primary", use_container_width=True)

    try:
        from lib.walkforward import run_walkforward
        WF_AVAILABLE = True
    except ImportError:
        WF_AVAILABLE = False
        st.error("walkforward.py not found.")

    def display_wf_results(results, mode_label):
        if 'error' in results:
            st.error(results['error']); return
        r = results
        ret_css = "metric-positive" if r['total_return_net'] > 0 else "metric-negative"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d3320,#0a2618);border:1px solid #22c55e33;
            border-radius:12px;padding:20px;margin:16px 0;text-align:center;">
            <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;">{mode_label} — {r['ticker']} — Blind Out-of-Sample</div>
            <div style="font-family:JetBrains Mono,monospace;font-size:42px;font-weight:700;" class="{ret_css}">{r['total_return_net']:+.2f}%</div>
            <div style="font-size:13px;color:#64748b;">Net of {r['total_cost_drag']:.2f}% transaction costs | Raw: {r['total_return_raw']:+.2f}%</div>
        </div>""", unsafe_allow_html=True)

        mc = st.columns(6)
        metrics = [
            ("Win Rate",     f"{r['win_rate']}%",            "metric-positive" if r['win_rate']>50 else "metric-negative", f"{r['winners']}W / {r['losers']}L"),
            ("Sharpe",       f"{r['sharpe']}",               "metric-positive" if r['sharpe']>1 else "",                   "out-of-sample"),
            ("Profit Factor",f"{r['profit_factor']}",        "metric-positive" if r['profit_factor']>1.5 else "",          ""),
            ("Max Drawdown", f"{r['max_drawdown']}%",        "metric-negative",                                             ""),
            ("Alpha vs B&H", f"{r['alpha_vs_bh']:+.1f}%",   "metric-positive" if r['alpha_vs_bh']>0 else "metric-negative",f"B&H: {r['buy_hold_return']:.1f}%"),
            ("WF Efficiency",f"{r['wf_efficiency']:.0f}%",  "metric-positive" if r['wf_efficiency']>50 else "metric-negative","test/train Sharpe"),
        ]
        for col, (label, val, css, sub) in zip(mc, metrics):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {css}">{val}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

        st.markdown("<div class='section-header' style='margin-top:24px;'>Blind Test Cumulative Returns</div>", unsafe_allow_html=True)
        cum = r['cumulative_curve']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(cum))), y=[c*100-100 for c in cum],
            fill='tozeroy',
            fillcolor='rgba(34,197,94,0.1)' if cum[-1]>1 else 'rgba(239,68,68,0.1)',
            line=dict(color='#22c55e' if cum[-1]>1 else '#ef4444', width=2),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(148,163,184,0.3)")
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#0a0b0f', plot_bgcolor='#0a0b0f',
            height=350, margin=dict(l=50,r=20,t=20,b=30),
            font=dict(family='Inter',size=12,color='#94a3b8'),
            xaxis_title="Trade #", yaxis_title="Cumulative Return %", showlegend=False,
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.03)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.03)')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<div class='section-header' style='margin-top:24px;'>Walk-Forward Windows</div>", unsafe_allow_html=True)
        for w in r['windows']:
            train_s = w['train_sharpe']; test_s = w['test_sharpe']
            eff = (test_s/train_s*100) if train_s>0 else 0
            eff_color  = '#22c55e' if eff>50 else ('#f59e0b' if eff>0 else '#ef4444')
            test_color = '#22c55e' if test_s>0 else '#ef4444'
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:10px 16px;margin:4px 0;
                background:linear-gradient(135deg,#1a2332,#172030);border-radius:8px;
                border:1px solid rgba(255,255,255,0.04);font-family:JetBrains Mono,monospace;font-size:12px;">
                <div style="width:80px;color:#94a3b8;font-weight:600;">Window {w['window']}</div>
                <div style="width:120px;color:#64748b;">Train: {w['train_range']}</div>
                <div style="width:120px;color:#64748b;">Test: {w['test_range']}</div>
                <div style="width:100px;color:#94a3b8;">RSI {w['params']['rsi_ob']}/{w['params']['rsi_os']} ADX {w['params']['adx_thresh']}</div>
                <div style="width:100px;">Train: <span style="color:#94a3b8;">{train_s:.2f}</span></div>
                <div style="width:100px;">Test: <span style="color:{test_color};font-weight:600;">{test_s:.2f}</span></div>
                <div style="width:70px;">Eff: <span style="color:{eff_color};">{eff:.0f}%</span></div>
                <div style="width:60px;color:#64748b;">{w['test_trades']} trades</div>
                <div style="width:60px;color:#64748b;">WR {w['test_win_rate']}%</div>
            </div>""", unsafe_allow_html=True)

        if r.get('regime_stats') is not None and len(r['regime_stats']) > 0:
            st.markdown("<div class='section-header' style='margin-top:24px;'>Performance by Regime</div>", unsafe_allow_html=True)
            for regime_name, row in r['regime_stats'].iterrows():
                wr = row.get('wr',0); avg = row.get('avg_net',0); cnt = int(row.get('count',0))
                wr_c = '#22c55e' if wr>50 else '#ef4444'; avg_c = '#22c55e' if avg>0 else '#ef4444'
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:16px;padding:8px 16px;margin:3px 0;
                    background:#0a0b0f;border-radius:8px;font-family:JetBrains Mono,monospace;font-size:13px;">
                    <div style="width:130px;color:#94a3b8;font-weight:600;">{regime_name}</div>
                    <div style="width:80px;color:#64748b;">{cnt} trades</div>
                    <div style="width:100px;">WR: <span style="color:{wr_c};font-weight:600;">{wr:.1f}%</span></div>
                    <div style="width:120px;">Avg: <span style="color:{avg_c};">{avg:.3f}%</span></div>
                    <div style="flex:1;background:rgba(255,255,255,0.04);border-radius:4px;height:14px;">
                        <div style="width:{min(wr,100)}%;background:{wr_c};height:100%;border-radius:4px;opacity:0.5;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-header' style='margin-top:24px;'>Trade Statistics</div>", unsafe_allow_html=True)
        tc = st.columns(4)
        for col, (label, val, css, sub) in zip(tc, [
            ("Avg Win",     f"{r['avg_win_pct']:.3f}%",  "metric-positive", "per trade"),
            ("Avg Loss",    f"{r['avg_loss_pct']:.3f}%", "metric-negative",  "per trade"),
            ("Total Trades",f"{r['total_trades']}",       "",                 f"across {len(r['windows'])} windows"),
            ("Avg Hold",    f"{r['avg_bars_held']:.0f} bars","",              ""),
        ]):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {css}">{val}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

    if WF_AVAILABLE and (run_intra or run_swing):
        mode       = 'intraday' if run_intra else 'swing'
        mode_label = "Intraday 0DTE (3-month blind test)" if run_intra else "Swing Trade (18-month blind test)"
        progress_bar = st.progress(0); status_text = st.empty()

        def wf_progress(current, total, message):
            progress_bar.progress(int(current/total*100))
            status_text.markdown(f"<div style='color:#94a3b8;font-family:JetBrains Mono,monospace;font-size:13px;'>{message}</div>", unsafe_allow_html=True)

        with st.spinner(f"Running Walk-Forward on {ticker} ({mode_label})..."):
            results = run_walkforward(ticker, mode=mode, progress_callback=wf_progress)

        progress_bar.progress(100)
        status_text.markdown("<div style='color:#22c55e;font-family:JetBrains Mono,monospace;font-size:13px;'>✅ Walk-forward complete</div>", unsafe_allow_html=True)
        display_wf_results(results, mode_label)
