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
    run_full_scan, run_watchlist_scan, get_sp500_tickers,
    load_scan_history, save_scan_history, merge_scan_results,
    generate_sparkline_base64,
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
    from lib.executor import (
        AlpacaExecutor, execute_signal, start_kill_switch_scheduler,
        log_trade_to_supabase, log_trade_exit, fetch_trade_log,
    )
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
    EXECUTOR_AVAILABLE = True
    TRADE_LOG_AVAILABLE = True
except Exception:
    ALPACA_API_KEY = ALPACA_SECRET_KEY = None
    ALPACA_PAPER = True
    TRADE_LOG_AVAILABLE = False

# ── Options sizing helpers (always available — pure math, no API) ──────────────
try:
    from lib.executor import build_trade_setup
    TRADE_SETUP_AVAILABLE = True
except Exception:
    TRADE_SETUP_AVAILABLE = False

# ── EOD Kill Switch ────────────────────────────────────────────────────────────
try:
    from lib.kill_switch import get_eod_status, close_0dte_positions
    KILL_SWITCH_AVAILABLE = True
except Exception:
    KILL_SWITCH_AVAILABLE = False
    def get_eod_status():
        return {'entry_locked': False, 'hard_close': False,
                'market_hours': True, 'now_et': '--:--:-- ET', 'today': ''}
    def close_0dte_positions(*a, **kw):
        return {'closed': [], 'skipped': [], 'errors': ['Kill switch unavailable']}

# ── HMM Regime Engine ──────────────────────────────────────────────────────────
try:
    from lib.regime_engine import (
        get_current_regime as _hmm_get_regime,
        REGIME_META        as _REGIME_META,
        model_exists       as _hmm_model_exists,
    )
    REGIME_ENGINE_AVAILABLE = True
except Exception:
    REGIME_ENGINE_AVAILABLE = False

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Realm",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

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

    /* ── Trading Cards (Pokémon-style) ── */
    @keyframes alpha-border {
        0%   { box-shadow: 0 0 10px 3px rgba(251,146,60,0.6), inset 0 1px 0 rgba(251,146,60,0.12); border-color: rgba(251,146,60,0.9); }
        33%  { box-shadow: 0 0 20px 6px rgba(251,191,36,0.7), inset 0 1px 0 rgba(251,191,36,0.15); border-color: rgba(251,191,36,1.0); }
        66%  { box-shadow: 0 0 14px 4px rgba(249,115,22,0.65), inset 0 1px 0 rgba(249,115,22,0.1); border-color: rgba(249,115,22,0.9); }
        100% { box-shadow: 0 0 10px 3px rgba(251,146,60,0.6), inset 0 1px 0 rgba(251,146,60,0.12); border-color: rgba(251,146,60,0.9); }
    }
    .trade-card {
        background: linear-gradient(165deg, #0c1016 0%, #111827 50%, #0c1016 100%);
        border: 1.5px solid rgba(99,102,241,0.2);
        border-radius: 15px; overflow: hidden; margin: 6px 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.5), inset 0 0 0 5px rgba(255,255,255,0.05);
        position: relative;
        will-change: transform;
        cursor: pointer;
        aspect-ratio: 2.5 / 3.5;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .alpha-glow {
        border-color: rgba(251,146,60,0.85);
        background: linear-gradient(165deg, #0f0900 0%, #1a0f00 50%, #0c0800 100%);
        animation: alpha-border 2.5s ease-in-out infinite;
    }
    /* Soft spotlight glare that tracks the cursor */
    .tc-glare {
        position: absolute; inset: 0; border-radius: 15px;
        pointer-events: none; z-index: 20;
        background: radial-gradient(
            farthest-corner ellipse at var(--mx, 50%) var(--my, 50%),
            rgba(255,255,255,0.30) 0%,
            rgba(255,255,255,0.12) 28%,
            transparent 58%
        );
        mix-blend-mode: screen;
        opacity: 0;
        transition: opacity 0.22s ease;
    }
    .trade-card:hover .tc-glare { opacity: 1; }
    /* Rainbow foil — colour-dodge spectrum, Alpha holos only */
    .tc-foil {
        position: absolute; inset: 0; border-radius: 15px;
        pointer-events: none; z-index: 19;
        background: linear-gradient(
            115deg,
            transparent          0%,
            rgba(255,0,120,0.20)  20%,
            rgba(255,210,0,0.24)  36%,
            rgba(0,255,90,0.20)   50%,
            rgba(0,180,255,0.24)  64%,
            rgba(200,0,255,0.20)  80%,
            transparent         100%
        );
        background-size: 200% 200%;
        background-position: var(--mx, 50%) var(--my, 50%);
        mix-blend-mode: color-dodge;
        opacity: 0;
        transition: opacity 0.22s ease;
    }
    .alpha-glow:hover .tc-foil { opacity: 0.50; }
    .tc-inner { padding: 14px 16px 12px 16px; }
    .tc-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }
    .tc-name { font-family: 'Syne', sans-serif; font-size: 21px; font-weight: 800; color: #e2e8f0; letter-spacing: -0.5px; line-height: 1; }
    .tc-hp-col { text-align: right; flex-shrink: 0; padding-left: 8px; }
    .tc-hp-num { font-family: 'DM Mono', monospace; font-size: 16px; font-weight: 700; line-height: 1; }
    .tc-art-box { background: #07080d; border-radius: 8px; overflow: hidden; margin: 8px 0; border: 1px solid rgba(255,255,255,0.04); min-height: 72px; }
    .tc-energy-row { display: flex; gap: 5px; align-items: flex-end; margin: 6px 0; flex-wrap: nowrap; }
    .tc-energy-label { font-family: 'DM Mono', monospace; font-size: 9px; color: #374151; letter-spacing: 1px; min-width: 28px; padding-bottom: 3px; }
    .tc-energy-dot { display: flex; flex-direction: column; align-items: center; gap: 2px; }
    .tc-moves { border-top: 1px solid rgba(255,255,255,0.04); border-bottom: 1px solid rgba(255,255,255,0.04); padding: 7px 0; margin: 7px 0; }
    .tc-move-line { font-family: 'DM Mono', monospace; font-size: 11px; color: #94a3b8; padding: 2px 0; display: flex; align-items: center; gap: 6px; }
    .tc-footer { display: flex; justify-content: space-between; align-items: center; padding-top: 4px; }
    .tc-footer-tag { font-family: 'DM Mono', monospace; font-size: 9px; color: #374151; text-transform: uppercase; letter-spacing: 1.5px; }

    /* ── Sparkline skeleton shimmer ── */
    @keyframes tr-shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position:  200% 0; }
    }

    /* ── Decayed Alpha card ── */
    .decayed-card {
        background: linear-gradient(165deg, #0a0808 0%, #100c0c 50%, #0a0808 100%) !important;
        border: 1.5px solid rgba(239,68,68,0.35) !important;
        animation: none !important;
        filter: saturate(0.15) brightness(0.65);
    }
    .decayed-card .tc-glare,
    .decayed-card .tc-foil { display: none !important; }
    .tc-decay-banner {
        background: linear-gradient(90deg, #450a0a, #7f1d1d, #450a0a);
        padding: 5px 10px 4px 10px;
        border-bottom: 1px solid rgba(239,68,68,0.4);
        text-align: center;
    }
    .tc-static-overlay {
        position: absolute; inset: 0; z-index: 25; pointer-events: none;
        background: repeating-linear-gradient(
            0deg, transparent, transparent 3px,
            rgba(239,68,68,0.03) 3px, rgba(239,68,68,0.03) 4px
        );
        animation: tc-glitch-scan 5s ease-in-out infinite;
    }
    @keyframes tc-glitch-scan {
        0%, 88%, 100% { opacity: 0; }
        90% { opacity: 1; transform: translateY(-1px); }
        92% { opacity: 0.6; transform: translateY(1px); }
        94% { opacity: 0; }
    }

    /* ── Battle Stations button ── */
    .pulse-btn-wrap { margin: 16px 0 8px 0; }

    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Holographic tilt engine — injected once, MutationObserver keeps it live ──
st.markdown("""
<script>
(function(){
  function onMove(e){
    var c=e.currentTarget,r=c.getBoundingClientRect();
    var x=e.clientX-r.left, y=e.clientY-r.top;
    var cx=r.width/2, cy=r.height/2;
    var rx=((y-cy)/cy)*12, ry=((cx-x)/cx)*12;
    var mx=(x/r.width)*100, my=(y/r.height)*100;
    c.style.transform='perspective(700px) rotateX('+rx+'deg) rotateY('+ry+'deg) scale3d(1.03,1.03,1.03)';
    var sh=(-ry*0.55)+'px '+(rx*0.55)+'px 28px rgba(0,0,0,0.55)';
    if(c.classList.contains('alpha-glow')) sh+=', 0 0 32px rgba(251,146,60,0.5)';
    c.style.boxShadow=sh;
    c.style.setProperty('--mx',mx+'%');
    c.style.setProperty('--my',my+'%');
  }
  function onLeave(e){
    var c=e.currentTarget;
    c.style.transform='';
    c.style.boxShadow='';
    c.style.setProperty('--mx','50%');
    c.style.setProperty('--my','50%');
  }
  function bind(){
    document.querySelectorAll('.trade-card').forEach(function(c){
      if(!c._tr){c._tr=1;c.addEventListener('mousemove',onMove);c.addEventListener('mouseleave',onLeave);}
    });
  }
  bind();
  new MutationObserver(bind).observe(document.body,{childList:true,subtree:true});
})();
</script>
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
    df = get_daily(ticker, days=max(days, 14))  # 14-day floor for weekend coverage
    if df.empty or len(df) < 2:
        # Raise — st.cache_data does NOT cache exceptions, so the next
        # rerun will retry the Alpaca call instead of serving a stale None.
        raise RuntimeError(f"No data returned for {ticker}")

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

    # Gap and candlestick patterns (reuse the same df — no extra Alpaca call)
    try:
        from lib.data_client import detect_daily_gap, detect_candle_patterns
        if not df.empty and len(df) >= 2:
            gap_info    = detect_daily_gap(df)
            candle_pats = detect_candle_patterns(df)
        else:
            gap_info    = {'gap_type': 'none', 'gap_pct': 0.0}
            candle_pats = []
    except Exception:
        gap_info    = {'gap_type': 'none', 'gap_pct': 0.0}
        candle_pats = []

    return {
        'df': df, 'regime': regime, 'regime_probs': regime_probs,
        'signal': signal, 'snap': snap, 'patterns': patterns,
        'gap': gap_info, 'candle_patterns': candle_pats,
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

nav_cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1])
pages = [
    ("📊 Ticker",      "ticker"),
    ("🎯 Scanner",     "scanner"),
    ("📌 Tracker",     "tracker"),
    ("💼 Trading",     "live"),
    ("🚨 Alerts",      "orders"),
    ("📜 Order Log",   "orderlog"),
    ("📈 Backtest",    "backtest"),
    ("🏆 Performance", "performance"),
    ("🗄️ DB",          "db_test"),
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
    # ── Admin authentication ──────────────────────────────────────────────────
    if st.session_state.authenticated:
        st.markdown(
            '<div style="background:#052e16;border:1px solid rgba(34,197,94,0.35);'
            'border-radius:8px;padding:8px 12px;font-family:DM Mono,monospace;'
            'font-size:11px;color:#4ade80;text-align:center;margin-bottom:4px;">'
            '🔓 Admin Session Active</div>',
            unsafe_allow_html=True,
        )
        if st.button("Lock Session", key="lock_admin", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
    else:
        admin_key = st.text_input("🔑 Admin Key", type="password", key="admin_key_input",
                                   placeholder="Enter key to unlock execution")
        try:
            _admin_pw = st.secrets.get("ADMIN_PASSWORD", "")
        except Exception:
            _admin_pw = ""
        if admin_key:
            if bool(_admin_pw) and (admin_key == _admin_pw):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect key", icon="❌")
        else:
            st.markdown(
                '<div style="background:#0f172a;border:1px solid rgba(99,102,241,0.15);'
                'border-radius:8px;padding:6px 12px;font-family:DM Mono,monospace;'
                'font-size:10px;color:#475569;text-align:center;margin-bottom:4px;">'
                '👁️ View-Only Mode — enter key to unlock</div>',
                unsafe_allow_html=True,
            )
    is_admin = st.session_state.authenticated

    # ── EOD Kill Switch status (computed once per sidebar render) ─────────────
    _eod = get_eod_status()
    _eod_entry_locked = _eod['entry_locked']
    _eod_hard_close   = _eod['hard_close']

    if _eod_hard_close:
        st.markdown(
            '<div style="background:linear-gradient(90deg,#450a0a,#7f1d1d,#450a0a);'
            'border:1.5px solid rgba(239,68,68,0.6);border-radius:8px;'
            'padding:10px 12px;margin:6px 0 10px 0;text-align:center;">'
            '<div style="font-family:DM Mono,monospace;font-size:10px;font-weight:700;'
            'color:#ef4444;letter-spacing:2px;text-transform:uppercase;">'
            '&#9888; EOD KILL SWITCH ACTIVE</div>'
            '<div style="font-family:DM Mono,monospace;font-size:9px;color:#fca5a5;'
            'margin-top:3px;letter-spacing:1px;">0DTE LIQUIDATION IN PROGRESS</div>'
            '<div style="font-family:DM Mono,monospace;font-size:8px;color:#7f1d1d;'
            'margin-top:4px;">' + _eod['now_et'] + '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif _eod_entry_locked:
        st.markdown(
            '<div style="background:#1c0a0a;border:1px solid rgba(239,68,68,0.35);'
            'border-radius:8px;padding:8px 12px;margin:4px 0 8px 0;text-align:center;">'
            '<div style="font-family:DM Mono,monospace;font-size:10px;font-weight:700;'
            'color:#f87171;letter-spacing:1px;">&#128274; 0DTE ENTRY LOCK</div>'
            '<div style="font-family:DM Mono,monospace;font-size:9px;color:#7f1d1d;'
            'margin-top:2px;">No new 0DTE orders after 3:30 PM ET</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Show help guide if active, otherwise show settings ───────────────────
    if st.session_state.get('show_help') and st.session_state.get('help_tab'):
        render_help_sidebar(st.session_state.help_tab)
    else:
        # ── Market Mood (HMM Regime) ──────────────────────────────────────────────
        if REGIME_ENGINE_AVAILABLE and _hmm_model_exists():
            if 'sidebar_regime' not in st.session_state:
                st.session_state.sidebar_regime      = None
                st.session_state.sidebar_regime_ts   = 0
            _regime_age = time.time() - st.session_state.get('sidebar_regime_ts', 0)
            # Refresh if: never fetched, errored last time, or cache is > 5 min old
            _regime_is_stale = (
                st.session_state.sidebar_regime is None
                or st.session_state.sidebar_regime.get('error')
                or _regime_age > 300
            )
            if _regime_is_stale:
                try:
                    _fresh = _hmm_get_regime()
                    if _fresh and not _fresh.get('error'):
                        st.session_state.sidebar_regime    = _fresh
                        st.session_state.sidebar_regime_ts = time.time()
                    elif st.session_state.sidebar_regime is None:
                        st.session_state.sidebar_regime = _fresh or {}
                    # On error, keep last known good regime — don't overwrite with error dict
                except Exception:
                    if st.session_state.sidebar_regime is None:
                        st.session_state.sidebar_regime = {}

            _sr = st.session_state.sidebar_regime or {}
            _sr_regime  = _sr.get('regime', 'Unknown')
            _sr_conf    = _sr.get('confidence', 0.0)
            _sr_meta    = _sr.get('meta') or _REGIME_META.get(_sr_regime, {})
            _sr_bias    = _sr_meta.get('bias', 'FLAT')
            _sr_color   = _sr_meta.get('color', '#6b7280')
            _sr_bg      = _sr_meta.get('bg', '#1f2937')
            _sr_sizing  = _sr_meta.get('sizing', '—')
            _sr_spy     = _sr.get('spy_close')
            _sr_bias_arrow = '▲' if _sr_bias == 'LONG' else ('▼' if _sr_bias == 'SHORT' else '—')
            _sr_bias_clr   = '#4ade80' if _sr_bias == 'LONG' else ('#f87171' if _sr_bias == 'SHORT' else '#f59e0b')

            # Confidence bar fill
            _sr_conf_fill = max(0, min(100, _sr_conf))
            _sr_conf_color = '#22c55e' if _sr_conf >= 70 else ('#f59e0b' if _sr_conf >= 45 else '#ef4444')

            _sr_spy_html = (
                f'<div style="font-size:10px;color:#4b5563;font-family:DM Mono,monospace;">SPY ${_sr_spy}</div>'
                if _sr_spy else ''
            )
            _sr_sizing_safe = _sr_sizing.replace("'", "&#39;")
            st.markdown(
                f'<div style="background:{_sr_bg};border:1px solid {_sr_color}44;border-radius:14px;'
                f'padding:14px 16px;margin-bottom:12px;">'
                f'<div style="font-size:9px;color:#6b7280;text-transform:uppercase;letter-spacing:2px;'
                f'font-family:DM Sans,sans-serif;margin-bottom:8px;">Market Mood</div>'
                f'<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:6px;">'
                f'<span style="font-family:Syne,sans-serif;font-size:16px;font-weight:700;'
                f'color:{_sr_color};">{_sr_regime}</span>'
                f'<span style="font-family:DM Mono,monospace;font-size:20px;font-weight:700;'
                f'color:{_sr_bias_clr};">{_sr_bias_arrow}</span>'
                f'</div>'
                f'<div style="margin-bottom:8px;">'
                f'<div style="font-size:9px;color:#4b5563;margin-bottom:3px;font-family:DM Sans,sans-serif;">Confidence</div>'
                f'<div style="background:#111827;border-radius:3px;height:5px;overflow:hidden;">'
                f'<div style="width:{_sr_conf_fill}%;height:100%;background:{_sr_conf_color};border-radius:3px;"></div>'
                f'</div>'
                f'<div style="font-size:10px;color:{_sr_conf_color};margin-top:2px;font-family:DM Mono,monospace;">{_sr_conf:.1f}%</div>'
                f'</div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div style="font-size:10px;color:#4b5563;font-family:DM Sans,sans-serif;">'
                f'Bias: <span style="color:{_sr_bias_clr};font-weight:600;">{_sr_bias}</span></div>'
                f'{_sr_spy_html}'
                f'</div>'
                f'<div style="margin-top:6px;font-size:9px;color:#374151;font-family:DM Sans,sans-serif;'
                f'line-height:1.4;">{_sr_sizing_safe}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Refresh Regime", key="refresh_regime_sidebar", use_container_width=True):
                try:
                    st.session_state.sidebar_regime = _hmm_get_regime()
                    st.rerun()
                except Exception as _re:
                    st.error(f"Regime refresh failed: {_re}")

        elif REGIME_ENGINE_AVAILABLE and not _hmm_model_exists():
            st.markdown("""
<div style="background:#1c1508;border:1px solid rgba(251,191,36,0.3);border-radius:12px;
    padding:12px 14px;margin-bottom:12px;">
  <div style="font-size:10px;color:#f59e0b;font-family:'DM Sans',sans-serif;">
    HMM Model not trained.<br>Run <code>python train_hmm.py</code> to activate the Market Mood.
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## Settings")

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

        # ── Auth Debug (temporary — remove once 401 resolved) ─────────────────
        try:
            from lib.data_client import get_auth_debug_info
            _dbg = get_auth_debug_info()
            if _dbg:
                _env_icon = '📄' if _dbg.get('environment') == 'PAPER' else '💰'
                st.sidebar.info(
                    f"**🔑 Alpaca Auth Debug**\n\n"
                    f"API key ends: `…{_dbg['key_last4']}` ({_dbg['key_len']} chars)\n\n"
                    f"Secret ends:  `…{_dbg['sec_last4']}` ({_dbg['sec_len']} chars)\n\n"
                    f"{_env_icon} Environment: **{_dbg['environment']}**\n\n"
                    f"Broker URL: `{_dbg['trading_url']}`\n\n"
                    f"Data URL:   `{_dbg['data_url']}`"
                )
        except Exception:
            pass

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
    if len(st.session_state.order_log) > 500:
        st.session_state.order_log = st.session_state.order_log[:500]


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
    show_tab_header("ticker", f"📊 {ticker}", datetime.now().strftime('%B %d, %Y'))
    render_daily_loss_gate()

    # ── Live Quote Search ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Live Quote</div>", unsafe_allow_html=True)

    from lib.data_client import get_live_quote
    from datetime import timezone as _tz

    with st.form("live_quote_form", clear_on_submit=False):
        _lq_c1, _lq_c2 = st.columns([6, 1])
        with _lq_c1:
            _lq_input = st.text_input(
                "lq_search",
                value=st.session_state.get("lq_ticker", ticker),
                placeholder="Enter any ticker — SPY, NVDA, AAPL, QQQ, TSLA...",
                label_visibility="collapsed",
            )
        with _lq_c2:
            _lq_go = st.form_submit_button("Search", use_container_width=True, type="primary")

    if _lq_go and _lq_input.strip():
        st.session_state.lq_ticker = _lq_input.strip().upper()

    _lq_sym = st.session_state.get("lq_ticker", ticker)

    get_live_quote._last_error = None   # reset before each call
    with st.spinner(f"Fetching {_lq_sym}..."):
        _quote = get_live_quote(_lq_sym)

    if _quote is None:
        # Snapshot failed (common on weekends with IEX) — fall back to last daily bar
        _lq_err = getattr(get_live_quote, '_last_error', None)
        try:
            from lib.data_client import get_daily as _gd
            _fb_df = _gd(_lq_sym, days=5)
            if not _fb_df.empty and len(_fb_df) >= 2:
                _fb_close  = float(_fb_df['Close'].iloc[-1])
                _fb_prev   = float(_fb_df['Close'].iloc[-2])
                _fb_chg    = _fb_close - _fb_prev
                _fb_chgpct = (_fb_chg / _fb_prev * 100) if _fb_prev else 0.0
                _quote = {
                    "ticker":          _lq_sym,
                    "price":           round(_fb_close, 2),
                    "change_pct":      round(_fb_chgpct, 2),
                    "change_dollar":   round(_fb_chg, 2),
                    "prev_close":      round(_fb_prev, 2),
                    "last_trade_time": None,   # signals market-closed path below
                    "_fallback":       True,
                }
            else:
                _quote = None
        except Exception:
            _quote = None

        if _quote is None:
            if _lq_err:
                st.error(f"Alpaca snapshot error — {_lq_err}")
            st.warning(
                f"⚠️ **{_lq_sym}** — no price data available. "
                "Ticker may be invalid or API keys incorrect."
            )

    if _quote is not None:
        _price   = _quote["price"]
        _chg_pct = _quote["change_pct"]
        _chg_dol = _quote["change_dollar"]
        _prev    = _quote["prev_close"]
        _is_up   = (_chg_pct or 0) >= 0
        _clr     = "#22c55e" if _is_up else "#ef4444"
        _arrow   = "▲" if _is_up else "▼"
        _sign    = "+" if _is_up else ""

        _chg_pct_str = f"{_sign}{_chg_pct:.2f}%" if _chg_pct is not None else "—"
        _chg_dol_str = f"{_sign}${abs(_chg_dol):.2f}" if _chg_dol is not None else "—"
        _prev_str    = f"${_prev:,.2f}" if _prev else "—"

        # Detect market-closed — fallback path or stale last_trade_time
        _ltt = _quote.get("last_trade_time")
        if _quote.get("_fallback"):
            _feed_label = "Market Closed · Last Daily Close (Alpaca)"
        elif _ltt:
            try:
                _age_h = (datetime.now(_tz.utc) - _ltt).total_seconds() / 3600
                _feed_label = "Market Closed · Last Session Price" if _age_h > 18 else "IEX · 15-min delayed"
            except Exception:
                _feed_label = "IEX · 15-min delayed"
        else:
            _feed_label = "IEX · 15-min delayed"

        _border_clr = "34,197,94" if _is_up else "239,68,68"
        _lqm1, _lqm2, _lqm3 = st.columns(3)
        with _lqm1:
            st.markdown(f"""
            <div class="metric-card" style="border-color:rgba({_border_clr},0.35);">
                <div class="metric-label">{_quote['ticker']} — Last Price</div>
                <div class="metric-value" style="font-size:34px;color:#e2e8f0;">${_price:,.2f}</div>
                <div class="metric-sub" style="color:#4b5563;">{_feed_label}</div>
            </div>""", unsafe_allow_html=True)
        with _lqm2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Daily Change</div>
                <div class="metric-value" style="font-size:28px;color:{_clr};">{_arrow} {_chg_pct_str}</div>
                <div class="metric-sub" style="color:{_clr};">{_chg_dol_str} vs prev close</div>
            </div>""", unsafe_allow_html=True)
        with _lqm3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Prev Close</div>
                <div class="metric-value" style="font-size:28px;color:#94a3b8;">{_prev_str}</div>
                <div class="metric-sub">Previous session close</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Full Analysis (HMM + signals) ─────────────────────────────────────────
    with st.spinner(f"Analyzing {ticker}..."):
        try:
            data = run_single_analysis(ticker, days=train_days)
        except Exception as _analysis_err:
            data = None
            st.warning(
                f"⚠️ **{ticker}** — Alpaca returned no daily bars. "
                "Market may be closed; data reflects the most recent trading session. "
                f"({_analysis_err})"
            )

    if data is None:
        st.stop()

    regime        = data['regime']
    signal        = data['signal']
    snap          = data['snap']
    patterns      = data['patterns']
    gap_info      = data.get('gap', {'gap_type': 'none', 'gap_pct': 0.0})
    candle_pats   = data.get('candle_patterns', [])

    _gap_type = gap_info.get('gap_type', 'none')
    _gap_pct  = gap_info.get('gap_pct', 0.0)

    # Build "Setup:" line — gap label + candle pattern names, shown inside signal card
    _setup_parts = []
    if _gap_type != 'none':
        _gap_dir_word = 'Gap Up' if 'up' in _gap_type else 'Gap Down'
        _gap_strength_word = 'Strong' if 'strong' in _gap_type else 'Moderate'
        _setup_parts.append(f"{_gap_strength_word} {_gap_dir_word} {_gap_pct:+.1f}%")
    for _cp in candle_pats:
        _cp_arrow = '↑' if _cp.get('direction') == 'up' else ('↓' if _cp.get('direction') == 'down' else '')
        _setup_parts.append(f"{_cp['name']} {_cp_arrow}".strip())
    _setup_line = ' · '.join(_setup_parts) if _setup_parts else '—'

    # Determine if a favorable gap/pattern boosts an A grade to A+
    _bullish_boost = (
        (signal['direction'] == 'LONG' and _gap_type in ('strong_up', 'moderate_up')) or
        (signal['direction'] == 'SHORT' and _gap_type in ('strong_down', 'moderate_down')) or
        any(
            (cp.get('name') in ('Bullish Engulfing',) and cp.get('direction') == 'up' and signal['direction'] == 'LONG') or
            (cp.get('name') in ('Bearish Engulfing',) and cp.get('direction') == 'down' and signal['direction'] == 'SHORT')
            for cp in candle_pats
        )
    )

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        sc = 'signal-long' if signal['direction']=='LONG' else ('signal-short' if signal['direction']=='SHORT' else 'signal-flat')
        _setup_color = '#4ade80' if ('up' in _gap_type or any(c.get('direction')=='up' for c in candle_pats)) else ('#f87171' if ('down' in _gap_type or any(c.get('direction')=='down' for c in candle_pats)) else '#94a3b8')
        st.markdown(f"""
        <div class="signal-card {sc}">
            <div class="signal-label">Signal</div>
            <div class="signal-value">{signal['trade_type']}</div>
            <div class="signal-sub">{signal['strength']} | Score: {signal['composite']:+d}</div>
            <div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.08);
                font-size:11px;font-family:'DM Mono',monospace;">
                <span style="color:#6b7280;">Setup: </span>
                <span style="color:{_setup_color};">{_setup_line}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # Execute button for A+/A grades on single ticker
        if EXECUTOR_AVAILABLE and is_executable_grade(signal.get('strength','')) and signal['direction'] != 'FLAT':
            if is_admin:
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
            else:
                st.markdown(
                    '<div style="background:#0f172a;border:1px solid rgba(99,102,241,0.12);'
                    'border-radius:8px;padding:8px 14px;font-family:DM Mono,monospace;'
                    'font-size:11px;color:#374151;text-align:center;margin-top:6px;">'
                    '🔒 Execution Disabled — Read-Only Mode</div>',
                    unsafe_allow_html=True,
                )

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

    st.markdown(
        "<div class='section-header'>Indicators "
        "<span style='font-size:10px;color:#334155;font-family:\"DM Mono\",monospace;"
        "background:#1e293b;border-radius:4px;padding:2px 7px;vertical-align:middle;"
        "margin-left:6px;'>Alpaca IEX</span></div>",
        unsafe_allow_html=True,
    )
    ic       = st.columns(6)
    ind_data = [
        ("RSI",      f"{snap['rsi']:.0f}",       "metric-positive" if 40<snap['rsi']<70 else "metric-negative",
         "Overbought" if snap['rsi']>70 else ("Oversold" if snap['rsi']<30 else "")),
        ("ADX",      f"{snap['adx']:.0f}",        "metric-positive" if snap['adx']>25 else "",
         f"{'↑ Rising' if snap['adx_rising'] else '↓ Falling'}"),
        ("MACD",     f"{snap['macd_hist']:.3f}",  "metric-positive" if snap['macd_hist']>0 else "metric-negative",
         "Cross ↑" if snap['macd_cross_up'] else ("Cross ↓" if snap['macd_cross_down'] else "")),
        ("Momentum", f"{snap['momentum']:.1f}%",  "metric-positive" if snap['momentum']>0 else "metric-negative", ""),
        ("BB %",     f"{snap['bb_pct']:.2f}",     "", f"{'Squeeze' if snap['bb_squeeze'] else ''}"),
        ("Volume",   f"{snap['vol_ratio']:.1f}x", "metric-positive" if snap['vol_ratio']>1.5 else "",
         f"{'High' if snap['high_volume'] else 'Normal'}"),
    ]
    for col, (label, val, css, sub) in zip(ic, ind_data):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {css}">{val}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

    # ── Full Trade Grade (uses pre-fetched df — no extra Alpaca call) ─────────
    with st.expander("🎯 Full Trade Grade (A+ / A / B / C / NO TRADE)", expanded=True):
        with st.spinner("Grading setup..."):
            try:
                from lib.trade_grader import grade_ticker_full
                full_grade = grade_ticker_full(
                    ticker=ticker,
                    regime=regime,
                    regime_confidence=signal['confidence'],
                    composite_score=signal['composite'],
                    direction=signal['direction'] if signal['direction'] != 'FLAT' else 'LONG',
                    indicators=snap,
                    strat_patterns=patterns,
                    df=data['df'],   # reuse the already-fetched daily DataFrame
                )
            except Exception as _ge:
                full_grade = None
                st.error(f"Grade engine error: {_ge}")

        if full_grade:
            _gi = full_grade.get('grade_intraday', '—')
            _gs = full_grade.get('grade_swing', '—')
            _ti = full_grade.get('trade_intraday', '—')
            _ts = full_grade.get('trade_swing', '—')
            _ri = full_grade.get('risk_0dte', 0)
            _rs = full_grade.get('risk_swing', 0)
            _ci = full_grade.get('contracts_0dte', 2)
            _cs = full_grade.get('contracts_swing', 2)
            _pi = full_grade.get('points_intraday', 0)
            _ps = full_grade.get('points_swing', 0)
            _strike = full_grade.get('strike_note', '')
            _swing_strike = full_grade.get('swing_strike_note', '')
            _flags = full_grade.get('flags', [])
            _ftc   = full_grade.get('ftc', {})
            _sec   = full_grade.get('sector_corr', {})
            _atr_g = full_grade.get('atr', {})

            # A → A+ UI boost when a favorable gap/engulfing pattern confirms the trade
            _boost_label = ''
            if _bullish_boost:
                if _gi == 'A':
                    _gi = 'A+'
                    _boost_label = 'Gap & Go' if _gap_type != 'none' else 'Engulfing Conf.'
                if _gs == 'A':
                    _gs = 'A+'
                    _boost_label = _boost_label or ('Gap & Go' if _gap_type != 'none' else 'Engulfing Conf.')

            _grade_colors = {'A+': '#22c55e', 'A': '#86efac', 'B': '#fbbf24', 'C': '#f97316', 'NO_TRADE': '#6b7280'}
            _gi_color = _grade_colors.get(_gi, '#94a3b8')
            _gs_color = _grade_colors.get(_gs, '#94a3b8')

            gc1, gc2 = st.columns(2)
            with gc1:
                _boost_badge = f'<div style="display:inline-block;background:#14532d;color:#4ade80;border-radius:5px;padding:2px 8px;font-size:10px;font-family:\'DM Mono\',monospace;margin-top:6px;">⬆ {_boost_label}</div>' if _boost_label else ''
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid {_gi_color}55;
                    border-radius:12px;padding:16px 20px;">
                    <div style="font-size:10px;color:#6b7280;font-family:'DM Mono',monospace;letter-spacing:1px;margin-bottom:6px;">INTRADAY (0DTE)</div>
                    <div style="font-size:36px;font-weight:800;color:{_gi_color};font-family:'JetBrains Mono',monospace;">{_gi}</div>
                    <div style="font-size:13px;color:#94a3b8;margin:4px 0;">{_ti} &nbsp;|&nbsp; {_pi} pts</div>
                    <div style="font-size:12px;color:#64748b;">Risk: ${_ri} &nbsp;·&nbsp; {_ci} contracts</div>
                    {_boost_badge}
                    {f'<div style="font-size:11px;color:#94a3b8;margin-top:6px;font-style:italic;">{_strike}</div>' if _strike else ''}
                </div>""", unsafe_allow_html=True)
            with gc2:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid {_gs_color}55;
                    border-radius:12px;padding:16px 20px;">
                    <div style="font-size:10px;color:#6b7280;font-family:'DM Mono',monospace;letter-spacing:1px;margin-bottom:6px;">SWING</div>
                    <div style="font-size:36px;font-weight:800;color:{_gs_color};font-family:'JetBrains Mono',monospace;">{_gs}</div>
                    <div style="font-size:13px;color:#94a3b8;margin:4px 0;">{_ts} &nbsp;|&nbsp; {_ps} pts</div>
                    <div style="font-size:12px;color:#64748b;">Risk: ${_rs} &nbsp;·&nbsp; {_cs} contracts</div>
                    {f'<div style="font-size:11px;color:#94a3b8;margin-top:6px;font-style:italic;">{_swing_strike}</div>' if _swing_strike else ''}
                </div>""", unsafe_allow_html=True)

            # ── Grade components ──────────────────────────────────────────────
            _ftc_txt  = f"{_ftc.get('aligned',0)}/{_ftc.get('total',0)} {'✅' if _ftc.get('ftc_confirmed') else '⚠️'} ({_ftc.get('direction','—')})" if _ftc else '—'
            _sec_txt  = f"{_sec.get('sector','—')} {'✅' if _sec.get('correlated') else '❌'} ({_sec.get('score',0)}/3)" if _sec else '—'
            _atr_txt  = _atr_g.get('move_potential', '—') if _atr_g else '—'

            st.markdown("<div style='margin-top:14px;display:flex;gap:12px;flex-wrap:wrap;'>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin-top:14px;display:flex;gap:12px;flex-wrap:wrap;">
                <div style="background:rgba(255,255,255,0.03);border:1px solid #334155;border-radius:8px;padding:10px 14px;min-width:140px;">
                    <div style="font-size:10px;color:#6b7280;font-family:'DM Mono',monospace;">FTC</div>
                    <div style="font-size:12px;color:#e2e8f0;margin-top:4px;">{_ftc_txt}</div>
                </div>
                <div style="background:rgba(255,255,255,0.03);border:1px solid #334155;border-radius:8px;padding:10px 14px;min-width:140px;">
                    <div style="font-size:10px;color:#6b7280;font-family:'DM Mono',monospace;">Sector</div>
                    <div style="font-size:12px;color:#e2e8f0;margin-top:4px;">{_sec_txt}</div>
                </div>
                <div style="background:rgba(255,255,255,0.03);border:1px solid #334155;border-radius:8px;padding:10px 14px;min-width:140px;">
                    <div style="font-size:10px;color:#6b7280;font-family:'DM Mono',monospace;">ATR Room</div>
                    <div style="font-size:12px;color:#e2e8f0;margin-top:4px;">{_atr_txt}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            if _flags:
                st.markdown("<div style='margin-top:10px;'>", unsafe_allow_html=True)
                for fl in _flags:
                    st.markdown(f"<div style='font-size:12px;color:#f59e0b;font-family:\"DM Mono\",monospace;'>⚠ {fl}</div>", unsafe_allow_html=True)

            # Grade reasons
            _reasons_i = full_grade.get('reasons_intraday', [])
            _reasons_s = full_grade.get('reasons_swing', [])
            if _reasons_i or _reasons_s:
                with st.expander("Grade reasoning", expanded=False):
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown("**Intraday**")
                        for rr in _reasons_i:
                            st.markdown(f"<div style='font-size:12px;color:#94a3b8;'>+ {rr}</div>", unsafe_allow_html=True)
                    with rc2:
                        st.markdown("**Swing**")
                        for rr in _reasons_s:
                            st.markdown(f"<div style='font-size:12px;color:#94a3b8;'>+ {rr}</div>", unsafe_allow_html=True)

    # ── FTFC Stack ────────────────────────────────────────────────────────────
    # Import must come BEFORE any attribute access on the function object.
    from lib.data_client import get_ftfc_snapshot

    with st.expander("🏗️ Full Timeframe Continuity (FTFC) Stack", expanded=False):
        ftfc_mode = st.radio("Mode", ["intraday", "swing"], horizontal=True, key="ftfc_mode_ticker")
        get_ftfc_snapshot._last_errors = []   # reset error list before each fetch
        with st.spinner("Fetching timeframe data from Alpaca..."):
            ftfc_stack = get_ftfc_snapshot(ticker, mode=ftfc_mode)

        _ftfc_errors = getattr(get_ftfc_snapshot, '_last_errors', [])
        if _ftfc_errors:
            with st.expander("⚠️ Timeframe fetch errors (debug)", expanded=False):
                for _err in _ftfc_errors:
                    st.error(_err)

        # ── Consensus header ──────────────────────────────────────────────
        dirs    = [s['direction'] for s in ftfc_stack if s['direction'] != 'neutral']
        up_cnt  = sum(1 for d in dirs if d == 'up')
        dn_cnt  = sum(1 for d in dirs if d == 'down')
        total   = len(dirs)
        if total == 0:          consensus = 'NEUTRAL'
        elif up_cnt == total:   consensus = 'BULLISH ✅'
        elif dn_cnt == total:   consensus = 'BEARISH ✅'
        elif up_cnt / total >= 0.75: consensus = 'LEANING BULL'
        elif dn_cnt / total >= 0.75: consensus = 'LEANING BEAR'
        else:                   consensus = 'MIXED'

        con_color = '#22c55e' if 'BULL' in consensus else ('#ef4444' if 'BEAR' in consensus else '#94a3b8')
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:14px;'>"
            f"<span style='font-family:\"DM Mono\",monospace;font-size:15px;font-weight:700;color:{con_color};'>{consensus}</span>"
            f"<span style='font-size:12px;color:#6b7280;'>{up_cnt}↑ &nbsp; {dn_cnt}↓ &nbsp;/&nbsp; {total} timeframes</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Per-timeframe cards ───────────────────────────────────────────
        cols = st.columns(len(ftfc_stack))
        for col, s in zip(cols, ftfc_stack):
            is_up    = s['direction'] == 'up'
            is_down  = s['direction'] == 'down'
            color    = '#22c55e' if is_up else ('#ef4444' if is_down else '#6b7280')
            bg       = 'rgba(34,197,94,0.06)' if is_up else ('rgba(239,68,68,0.06)' if is_down else 'rgba(255,255,255,0.02)')
            arrow    = '↑' if is_up else ('↓' if is_down else '—')
            sign     = '+' if is_up else ''
            chg_str  = f"{sign}{s['change_pct']:.2f}%" if s['change_pct'] is not None else '—'
            open_str  = f"${s['open']:,.2f}"  if s['open']  is not None else '—'
            close_str = f"${s['close']:,.2f}" if s['close'] is not None else '—'
            with col:
                st.markdown(f"""
                <div style="text-align:center;background:{bg};
                    border:1px solid {color}55;border-radius:10px;padding:10px 4px;">
                    <div style="font-size:10px;color:#6b7280;font-family:'DM Mono',monospace;
                        letter-spacing:0.5px;margin-bottom:4px;">{s['tf']}</div>
                    <div style="font-size:22px;color:{color};font-weight:700;line-height:1;">{arrow}</div>
                    <div style="font-size:10px;color:{color};font-weight:600;margin:3px 0;">{chg_str}</div>
                    <div style="font-size:9px;color:#94a3b8;font-family:'DM Mono',monospace;">O {open_str}</div>
                    <div style="font-size:9px;color:#e2e8f0;font-family:'DM Mono',monospace;">C {close_str}</div>
                </div>
                """, unsafe_allow_html=True)


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

    # ── EOD auto-liquidation (fires once per day after 15:50 ET) ─────────────
    if is_admin and _eod_hard_close:
        _fired_date = st.session_state.get('eod_fired_date', '')
        if _fired_date != _eod['today']:
            st.session_state['eod_fired_date'] = _eod['today']
            _eod_exe = get_executor()
            # Derive swing tickers from this session's order log (don't close overnight holds)
            _swing_held = list({
                o['ticker'] for o in st.session_state.get('order_log', [])
                if o.get('trade_type', '').lower() == 'swing'
                and o.get('status') not in ('ERROR', 'BLOCKED')
            })
            if _eod_exe:
                _ks_n = _eod_exe.kill_switch_0dte(swing_tickers=_swing_held)
                st.error(
                    f"⚠️ EOD KILL SWITCH ACTIVE — 0DTE liquidation fired at {_eod['now_et']}. "
                    f"{_ks_n} position(s) closed. "
                    + (f"Swing held: {', '.join(_swing_held)}." if _swing_held else "No swing positions."),
                    icon="🚨",
                )

    render_daily_loss_gate()

    if 'scan_history'      not in st.session_state: st.session_state.scan_history      = load_scan_history()
    if 'last_scan_results' not in st.session_state: st.session_state.last_scan_results = None
    if 'play_sound'        not in st.session_state: st.session_state.play_sound        = False
    if 'fast_scan_results' not in st.session_state: st.session_state.fast_scan_results = None
    # Pulse Scan persistence
    if 'active_tickers'   not in st.session_state: st.session_state.active_tickers    = []
    if 'alpha_snapshot'   not in st.session_state: st.session_state.alpha_snapshot     = {}
    if 'pulse_decay'      not in st.session_state: st.session_state.pulse_decay        = {}

    _RO = ('<div style="background:#0f172a;border:1px solid rgba(99,102,241,0.12);'
           'border-radius:8px;padding:7px 10px;font-family:DM Mono,monospace;'
           'font-size:10px;color:#374151;text-align:center;">🔒 Read-Only</div>')
    col_btn1, col_btn2, col_btn3, col_info = st.columns([1, 1, 1, 2])
    with col_btn1:
        if is_admin:
            run_scan = st.button("🚀 Full S&P Scan", type="primary", use_container_width=True)
        else:
            run_scan = False
            st.markdown(_RO, unsafe_allow_html=True)
    with col_btn2:
        if is_admin:
            run_fast_scan = st.button("⚡ Fast Watchlist", use_container_width=True, help="Scans your watchlist (≤25 tickers) in under 8 seconds using batch snapshots + parallel FTFC")
        else:
            run_fast_scan = False
    with col_btn3:
        if is_admin:
            clear_hist = st.button("🗑️ Clear History", use_container_width=True)
        else:
            clear_hist = False
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

        # Snapshot for Pulse Scan decay detection
        _snap_rows = results.get('all_qualified', [])
        st.session_state.active_tickers = [_s['ticker'] for _s in _snap_rows]
        st.session_state.alpha_snapshot = {
            _s['ticker']: {'ptr': _s.get('ptr_score', 0), 'price': _s.get('price'),
                           'dir': _s.get('direction', 'FLAT'), 'gap_type': _s.get('gap_type', '')}
            for _s in _snap_rows
        }
        st.session_state.pulse_decay = {}

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

    # ── Fast Watchlist Scan ───────────────────────────────────────────────────
    if run_fast_scan:
        from config import ALL_TICKERS
        _fp = st.progress(0); _fs = st.empty()

        def _fast_progress(stage, current, total, message):
            pct = int((current / max(1, total)) * 20) if stage == "snapshot" else (20 + int((current / max(1, total)) * 80))
            _fp.progress(min(pct, 100))
            _fs.markdown(f"<div style='color:#94a3b8;font-family:JetBrains Mono,monospace;font-size:13px;'>{message}</div>", unsafe_allow_html=True)

        _fr = run_watchlist_scan(tickers=ALL_TICKERS, timeout_secs=7.5, progress_callback=_fast_progress)
        st.session_state.fast_scan_results = _fr
        _fp.progress(100)
        # Snapshot for Pulse Scan decay detection
        st.session_state.active_tickers = [_r['ticker'] for _r in _fr.get('results', [])]
        st.session_state.alpha_snapshot  = {
            _r['ticker']: {'ptr': _r.get('ptr_score', 0), 'price': _r.get('price'),
                           'dir': _r.get('direction', 'FLAT'), 'gap_type': _r.get('gap_type', '')}
            for _r in _fr.get('results', [])
        }
        st.session_state.pulse_decay = {}
        _warn = f" ⚠️ {_fr['warning']}" if _fr.get('truncated') else ""
        _fs.markdown(
            f"<div style='color:#22c55e;font-family:JetBrains Mono,monospace;font-size:13px;'>"
            f"⚡ Done in {_fr['elapsed_secs']}s — {_fr['completed']}/{_fr['total_scanned']} tickers, "
            f"ranked by FTFC alignment{_warn}</div>",
            unsafe_allow_html=True,
        )

    # ── Pulse Scan: re-check active tickers for decay ────────────────────────
    if st.session_state.active_tickers:
        if _eod_hard_close:
            # After 15:50 the Kill Switch has already liquidated — pulse is meaningless
            _pulse_clicked = False
            st.markdown(
                '<div style="background:#1c0a0a;border:1px solid rgba(239,68,68,0.35);'
                'border-radius:8px;padding:10px;font-family:DM Mono,monospace;'
                'font-size:11px;color:#f87171;text-align:center;margin:8px 0;">'
                '&#9888; BATTLE STATIONS offline — EOD Kill Switch active. Positions liquidated.</div>',
                unsafe_allow_html=True,
            )
        elif is_admin:
            _pulse_clicked = st.button(
                f"⚡ BATTLE STATIONS: PULSE SCAN  ({len(st.session_state.active_tickers)} tickers)",
                type="primary",
                use_container_width=True,
                key="pulse_scan_btn",
            )
        else:
            _pulse_clicked = False
            st.markdown(
                '<div style="background:#0f172a;border:1px solid rgba(99,102,241,0.12);'
                'border-radius:8px;padding:10px;font-family:DM Mono,monospace;'
                'font-size:11px;color:#374151;text-align:center;margin:8px 0;">'
                '🔒 BATTLE STATIONS locked — Read-Only Mode</div>',
                unsafe_allow_html=True,
            )

        if _pulse_clicked:
            _pb = st.progress(0); _ps = st.empty()

            def _pulse_prog(stage, current, total, message):
                _pb.progress(min(int((current / max(1, total)) * 100), 100))
                _ps.markdown(
                    f"<div style='color:#94a3b8;font-family:DM Mono,monospace;font-size:13px;'>"
                    f"⚡ {message}</div>", unsafe_allow_html=True,
                )

            _pr = run_watchlist_scan(
                tickers=st.session_state.active_tickers,
                timeout_secs=10,
                progress_callback=_pulse_prog,
            )
            _pb.progress(100)

            # ── Decay detection ─────────────────────────────────────────────
            _snap       = st.session_state.alpha_snapshot
            _grade_rank = {'A+': 3, 'A': 2, 'B': 1, 'C': 0}

            def _ptr_to_grade(ptr):
                if ptr >= 75: return 'A+'
                if ptr >= 60: return 'A'
                if ptr >= 45: return 'B'
                return 'C'

            _new_decay = {}
            for _pr_row in _pr.get('results', []):
                _t  = _pr_row['ticker']
                _sn = _snap.get(_t, {})
                if not _sn:
                    continue
                _snap_ptr   = _sn.get('ptr', 0)
                _snap_price = _sn.get('price') or _pr_row.get('price')
                _snap_dir   = _sn.get('dir', 'FLAT')
                _new_ptr    = _pr_row.get('ptr_score', 0)
                _new_price  = _pr_row.get('price') or _snap_price
                _sg  = _ptr_to_grade(_snap_ptr)
                _ng  = _ptr_to_grade(_new_ptr)
                _grade_dropped = _grade_rank.get(_ng, 0) < _grade_rank.get(_sg, 0)
                _mag_decayed = False
                try:
                    if _snap_price and _new_price:
                        _sp, _np = float(_snap_price), float(_new_price)
                        if _snap_dir == 'LONG':
                            _prog = (_np - _sp) / max(_sp * 0.05, 0.01)
                        elif _snap_dir == 'SHORT':
                            _prog = (_sp - _np) / max(_sp * 0.05, 0.01)
                        else:
                            _prog = 0
                        _mag_decayed = _prog >= 0.30
                except Exception:
                    pass
                _reasons = []
                if _grade_dropped:  _reasons.append(f"Grade {_sg}→{_ng}")
                if _mag_decayed:    _reasons.append("≥30% move captured")
                _new_decay[_t] = {
                    'is_decayed': _grade_dropped or _mag_decayed,
                    'reasons':    _reasons,
                    'snap_grade': _sg, 'new_grade': _ng,
                }

            st.session_state.pulse_decay = _new_decay
            _n_dec = sum(1 for v in _new_decay.values() if v['is_decayed'])
            _ps.markdown(
                f"<div style='color:#22c55e;font-family:DM Mono,monospace;font-size:13px;'>"
                f"✅ Pulse complete — {len(st.session_state.active_tickers)} tickers checked · "
                f"<span style='color:#ef4444;'>{_n_dec} DECAYED</span> · "
                f"{len(st.session_state.active_tickers) - _n_dec} still live</div>",
                unsafe_allow_html=True,
            )

    # ── Fast Scan Results Display ─────────────────────────────────────────────
    _fsr = st.session_state.get('fast_scan_results')
    if _fsr and _fsr.get('results'):
        _tf_names = ['Monthly', 'Weekly', 'Daily', '4H', '60min', '15min', '5min']
        _tf_short  = ['M', 'W', 'D', '4H', '60', '15', '5']

        st.markdown(
            f"<div class='section-header' style='margin-top:24px;'>⚡ Fast Watchlist — "
            f"{_fsr['completed']}/{_fsr['total_scanned']} tickers · {_fsr['elapsed_secs']}s · "
            f"ranked by FTFC alignment</div>",
            unsafe_allow_html=True,
        )
        if _fsr.get('truncated'):
            st.warning(_fsr['warning'])

        # ── Regime context (fetched once for the whole results block) ─────────
        _scan_regime_data: dict = {}
        if REGIME_ENGINE_AVAILABLE:
            try:
                _scan_regime_data = _hmm_get_regime()
            except Exception:
                pass
        _scan_regime    = _scan_regime_data.get('regime', '')
        _scan_regime_meta = _scan_regime_data.get('meta') or _REGIME_META.get(_scan_regime, {})
        # Risk multiplier: Volatile → 50%, everything else → 100%
        _regime_risk_mult = 0.5 if 'Volatile' in _scan_regime else 1.0

        # ── Pokémon-style trade card builder (Pass 1 skeleton → Pass 2 chart) ──
        def _make_card_html(cv, art_html='', decayed=False):
            d         = cv['dir']
            ptr       = cv['ptr']
            hp_color  = '#22c55e' if ptr >= 70 else ('#f59e0b' if ptr >= 40 else '#ef4444')
            hp_w      = str(max(0, min(100, ptr))) + '%'
            dir_color = '#4ade80' if d == 'LONG' else ('#f87171' if d == 'SHORT' else '#6b7280')
            dir_arrow = '▲' if d == 'LONG' else ('▼' if d == 'SHORT' else '—')
            ps        = ('$' + str(cv['price'])) if cv['price'] else '—'
            if decayed:
                card_cls     = 'trade-card decayed-card'
                decay_banner = (
                    '<div class="tc-decay-banner">'
                    '<div style="font-family:\'DM Mono\',monospace;font-size:10px;font-weight:700;'
                    'color:#ef4444;letter-spacing:2px;text-transform:uppercase;">'
                    '&#9888; DECAYED ALPHA | SUPER EFFECTIVE (-)</div>'
                    '<div style="font-family:\'DM Mono\',monospace;font-size:9px;color:#7f1d1d;'
                    'margin-top:2px;letter-spacing:1px;">MOVE PASSED &#8212; DO NOT CHASE</div>'
                    '</div>'
                )
                static_overlay = '<div class="tc-static-overlay"></div>'
            else:
                card_cls       = 'trade-card alpha-glow' if cv.get('alpha_setup') else 'trade-card'
                decay_banner   = ''
                static_overlay = ''

            # Compact sector badge
            sec_etf = cv.get('sector_etf', '')
            sec_dir = cv.get('sector_dir', 'neutral')
            if sec_etf:
                sec_c = '#4ade80' if sec_dir == 'up' else ('#f87171' if sec_dir == 'down' else '#6b7280')
                sec_a = '▲' if sec_dir == 'up' else ('▼' if sec_dir == 'down' else '—')
                sec_badge = (
                    '<span style="background:#111827;color:' + sec_c + ';border-radius:4px;'
                    'padding:1px 6px;font-size:9px;font-family:\'DM Mono\',monospace;font-weight:600;">'
                    + sec_etf + ' ' + sec_a + '</span>'
                )
            else:
                sec_badge = ''

            div_tag = (
                '<span style="color:#f59e0b;font-size:9px;font-family:\'DM Mono\',monospace;">'
                '⚠ Div</span>'
            ) if cv.get('sector_div') else ''

            # FTFC energy dots
            tf_names = ['Monthly', 'Weekly', 'Daily', '4H', '60min', '15min', '5min']
            tf_short = ['M', 'W', 'D', '4H', '60', '15', '5']
            stack    = cv.get('ftfc_stack', [])
            tf_map   = {t.get('tf'): t for t in stack}
            e_html   = '<div class="tc-energy-row"><span class="tc-energy-label">FTFC</span>'
            for tfn, tfs in zip(tf_names, tf_short):
                tfd   = tf_map.get(tfn, {}).get('direction', 'neutral')
                dot_c = '#22c55e' if tfd == 'up' else ('#ef4444' if tfd == 'down' else '#1e2433')
                bdr_c = '#166534' if tfd == 'up' else ('#7f1d1d' if tfd == 'down' else '#374151')
                e_html += (
                    '<div class="tc-energy-dot">'
                    '<div style="width:12px;height:12px;border-radius:50%;background:' + dot_c + ';'
                    'border:1.5px solid ' + bdr_c + ';"></div>'
                    '<span style="font-size:7px;color:#374151;font-family:\'DM Mono\',monospace;">'
                    + tfs + '</span></div>'
                )
            e_html += '</div>'

            # Moves section
            gap_type    = cv.get('gap_type', 'No Gap')
            gap_pct     = cv.get('gap_pct', 0.0)
            gap_c       = '#4ade80' if 'Up' in gap_type else ('#f87171' if 'Down' in gap_type else '#6b7280')
            gap_icon    = '⚡' if 'Up' in gap_type else ('⬇' if 'Down' in gap_type else '◦')
            gap_pct_str = (('+' if gap_pct > 0 else '') + str(round(gap_pct, 1)) + '%') if abs(gap_pct) > 0.01 else ''
            exec_str    = 'BUY CALL' if d == 'LONG' else ('BUY PUT' if d == 'SHORT' else '—')
            ss          = (' +' + str(cv['sentinel_bonus']) + ' Sentinel') if cv.get('sentinel_bonus') else ''

            alpha_badge = (
                '<div style="text-align:center;font-size:11px;color:#fb923c;'
                'font-family:\'DM Mono\',monospace;font-weight:700;letter-spacing:1px;'
                'padding:5px 0 1px 0;">\U0001f525 ALPHA SETUP</div>'
            ) if (cv.get('alpha_setup') and not decayed) else ''

            return (
                '<div class="' + card_cls + '">'
                + static_overlay + decay_banner +
                '<div class="tc-glare"></div>'
                '<div class="tc-foil"></div>'
                '<div class="tc-inner">'
                '<div class="tc-header">'
                '<div>'
                '<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px;">'
                '<span style="font-size:18px;color:' + dir_color + ';">' + dir_arrow + '</span>'
                '<span class="tc-name">' + cv['sym'] + '</span>'
                '</div>'
                '<div style="display:flex;gap:5px;align-items:center;flex-wrap:wrap;">'
                + sec_badge + cv.get('gap_badge_html', '') + div_tag +
                '<span style="color:#6b7280;font-family:\'DM Mono\',monospace;font-size:11px;">'
                + ps + ' ' + cv['chg_str'] + '</span>'
                '</div></div>'
                '<div class="tc-hp-col">'
                '<div style="font-size:9px;color:#4b5563;font-family:\'DM Mono\',monospace;'
                'letter-spacing:1px;text-align:right;margin-bottom:2px;">HP</div>'
                '<div class="tc-hp-num" style="color:' + hp_color + ';">' + str(ptr) + '</div>'
                '<div style="background:#0d1117;border-radius:3px;width:56px;height:4px;'
                'overflow:hidden;margin:3px 0 0 auto;">'
                '<div style="width:' + hp_w + ';height:100%;background:' + hp_color + ';border-radius:3px;"></div>'
                '</div>'
                '<div style="font-size:8px;color:#374151;font-family:\'DM Mono\',monospace;'
                'text-align:right;margin-top:3px;">' + cv.get('scan_time', '') + '</div>'
                '</div></div>'
                '<div class="tc-art-box">' + art_html + '</div>'
                + e_html +
                '<div class="tc-moves">'
                '<div class="tc-move-line">'
                '<span style="color:' + gap_c + ';">' + gap_icon + '</span>'
                '<span style="color:' + gap_c + ';font-weight:700;">' + gap_type + '</span>'
                + ('<span style="color:#6b7280;"> ' + gap_pct_str + '</span>' if gap_pct_str else '') +
                '</div>'
                '<div class="tc-move-line">'
                '<span style="color:#818cf8;">⚔</span>'
                '<span style="color:#c4b5fd;font-weight:700;">' + exec_str + '</span>'
                + ('<span style="color:#fb923c;font-size:9px;"> ' + ss + '</span>' if ss else '') +
                '</div></div>'
                '<div class="tc-footer">'
                '<span class="tc-footer-tag">Paper Trade</span>'
                '<span class="tc-footer-tag" style="color:#6366f1;">33% Guard ☑</span>'
                '</div>'
                + alpha_badge +
                '</div></div>'
            )

        _SKELETON_ART = (
            '<div style="width:100%;height:72px;position:relative;'
            'background:linear-gradient(90deg,#0d1117 25%,#1a2332 50%,#0d1117 75%);'
            'background-size:200% 100%;animation:tr-shimmer 1.5s ease-in-out infinite;">'
            '<div style="position:absolute;inset:0;display:flex;align-items:center;'
            'justify-content:center;">'
            '<span style="font-family:\'DM Mono\',monospace;font-size:9px;color:#374151;'
            'letter-spacing:2px;">LOADING CHART...</span>'
            '</div></div>'
        )

        _card_slots = []  # [(placeholder, cv_dict)] — filled by Pass 1, updated by Pass 2
        _cols = st.columns(5)

        for _ci, _r in enumerate(_fsr['results']):
            _sym     = _r['ticker']
            _price   = _r.get('price')
            _chg     = _r.get('change_pct')
            _dir     = _r.get('direction', 'FLAT')
            _ptr     = _r.get('ptr_score', 0)
            _comp    = _r.get('composite', 0)
            _stack   = _r.get('ftfc_stack', [])
            _au       = _r.get('aligned_up', 0)
            _ad       = _r.get('aligned_down', 0)
            _tot      = _r.get('total_tfs', 0)
            _gap_type         = _r.get('gap_type', 'No Gap')
            _gap_pct          = _r.get('gap_pct', 0.0)
            _sector_etf       = _r.get('sector_etf', '')
            _sector_dir       = _r.get('sector_dir', 'neutral')
            _sentinel_bonus   = _r.get('sentinel_bonus', 0)
            _sector_div       = _r.get('sector_divergence', False)
            _alpha_setup      = _r.get('alpha_setup', False)

            _arrow_color = '#4ade80' if _dir == 'LONG' else ('#f87171' if _dir == 'SHORT' else '#6b7280')
            _arrow       = '▲' if _dir == 'LONG' else ('▼' if _dir == 'SHORT' else '—')
            _chg_str     = f"{'+' if _chg and _chg > 0 else ''}{_chg:.2f}%" if _chg is not None else '—'
            _chg_color   = '#4ade80' if _chg and _chg > 0 else ('#f87171' if _chg and _chg < 0 else '#6b7280')

            # Gap badge
            _gap_badge_map = {
                'Full Up':    ('FG-UP', '#4ade80', '#14532d'),
                'Partial Up': ('PG-UP', '#86efac', '#1a2e1a'),
                'Full Down':  ('FG-DN', '#f87171', '#7f1d1d'),
                'Partial Down': ('PG-DN', '#fca5a5', '#3a1212'),
            }
            if _gap_type in _gap_badge_map:
                _gl, _gc, _gbg = _gap_badge_map[_gap_type]
                _gap_sign = '+' if _gap_pct > 0 else ''
                _gap_badge_html = (
                    f'<span style="background:{_gbg};color:{_gc};border-radius:6px;'
                    f'padding:2px 8px;font-size:10px;font-family:\'DM Mono\',monospace;'
                    f'font-weight:700;">{_gl} {_gap_sign}{_gap_pct:.1f}%</span>'
                )
            else:
                _gap_badge_html = ''

            # Sector badge
            if _sector_etf and _sector_dir != 'neutral':
                _sec_arrow = '▲' if _sector_dir == 'up' else '▼'
                _sec_color = '#4ade80' if _sector_dir == 'up' else '#f87171'
                _sec_bg    = '#14532d' if _sector_dir == 'up' else '#7f1d1d'
                _sector_badge_html = (
                    f'<span style="background:{_sec_bg};color:{_sec_color};border-radius:6px;'
                    f'padding:2px 8px;font-size:10px;font-family:\'DM Mono\',monospace;'
                    f'font-weight:700;">{_sector_etf} {_sec_arrow}</span>'
                )
            elif _sector_etf:
                _sector_badge_html = (
                    f'<span style="background:#1f2937;color:#6b7280;border-radius:6px;'
                    f'padding:2px 8px;font-size:10px;font-family:\'DM Mono\',monospace;">'
                    f'{_sector_etf} —</span>'
                )
            else:
                _sector_badge_html = ''

            # Divergence / Alpha overlays
            _divergence_html = (
                '<div style="color:#f59e0b;font-size:11px;font-family:\'DM Mono\',monospace;'
                'margin-top:6px;">⚠️ Sector Divergence — trade with caution</div>'
                if _sector_div else ''
            )
            _alpha_html = (
                '<div style="color:#fb923c;font-size:12px;font-family:\'DM Mono\',monospace;'
                'font-weight:700;letter-spacing:1px;margin-top:6px;">🔥 ALPHA SETUP</div>'
                if _alpha_setup else ''
            )
            _card_border = 'rgba(251,146,60,0.6)' if _alpha_setup else 'rgba(99,102,241,0.15)'
            _card_bg     = ('linear-gradient(135deg,#1a0f00,#0f0a00)'
                            if _alpha_setup else
                            'linear-gradient(135deg,#111827,#0f172a)')

            # Build FTFC heat strip
            _strip_html = ''
            _tf_map = {tf.get('tf'): tf for tf in _stack}
            for _tname, _tshort in zip(_tf_names, _tf_short):
                _tf_data  = _tf_map.get(_tname, {})
                _tf_dir   = _tf_data.get('direction', 'neutral')
                _bg = '#166534' if _tf_dir == 'up' else ('#7f1d1d' if _tf_dir == 'down' else '#1f2937')
                _fc = '#4ade80' if _tf_dir == 'up' else ('#f87171' if _tf_dir == 'down' else '#374151')
                _strip_html += (
                    f'<div style="display:inline-flex;flex-direction:column;align-items:center;'
                    f'background:{_bg};border-radius:5px;padding:3px 6px;min-width:28px;">'
                    f'<span style="font-size:9px;color:#6b7280;">{_tshort}</span>'
                    f'<span style="font-size:10px;font-weight:700;color:{_fc};">'
                    f'{"▲" if _tf_dir=="up" else ("▼" if _tf_dir=="down" else "—")}</span>'
                    f'</div>'
                )

            # PTR bar fill
            _ptr_bar_w  = max(0, min(100, _ptr))
            _ptr_color  = '#22c55e' if _ptr >= 70 else ('#f59e0b' if _ptr >= 40 else '#ef4444')

            # Capture card vars for Pass 2 chart update
            _cv = {
                'sym': _sym, 'price': _price, 'dir': _dir, 'ptr': _ptr,
                'au': _au, 'ad': _ad, 'tot': _tot,
                'arrow': _arrow, 'arrow_color': _arrow_color,
                'chg_str': _chg_str, 'chg_color': _chg_color,
                'gap_badge_html': _gap_badge_html,
                'sector_badge_html': _sector_badge_html,
                'alpha_html': _alpha_html,
                'divergence_html': _divergence_html,
                'card_border': _card_border, 'card_bg': _card_bg,
                'strip_html': _strip_html,
                'ptr_bar_w': _ptr_bar_w, 'ptr_color': _ptr_color,
                'sentinel_bonus': _sentinel_bonus,
                'scan_time': _r.get('scan_time', ''),
                # Pokémon card fields
                'alpha_setup': _alpha_setup,
                'sector_etf':  _sector_etf,
                'sector_dir':  _sector_dir,
                'sector_div':  _sector_div,
                'gap_type':    _gap_type,
                'gap_pct':     _gap_pct,
                'ftfc_stack':  _stack,
            }
            _decay_info = st.session_state.get('pulse_decay', {}).get(_sym, {})
            _is_decayed = _decay_info.get('is_decayed', False)
            _cv['_is_decayed'] = _is_decayed

            _col = _cols[_ci % 5]
            with _col:
                # Pass 1 — instant render with shimmer skeleton where chart will go
                _ph = st.empty()
                _ph.markdown(_make_card_html(_cv, _SKELETON_ART, decayed=_is_decayed), unsafe_allow_html=True)
                _card_slots.append((_ph, _cv))

                # ── Trade Setup expander ──────────────────────────────────────────
                # Alpha Kill Switch: full gap + sector confirmed BUT regime is Chop
                # → downgrade to information-only, no execution button
                _alpha_chop_killed = _alpha_setup and _scan_regime == 'Chop'

                if TRADE_SETUP_AVAILABLE and _price and _dir != 'FLAT':
                    if _alpha_chop_killed:
                        # Show information-only card instead of expander
                        st.markdown(f"""
<div style="background:#1c1508;border:1px solid rgba(251,191,36,0.35);border-radius:12px;
    padding:10px 14px;margin:4px 0 8px 0;">
  <div style="font-size:11px;color:#f59e0b;font-family:'DM Mono',monospace;font-weight:700;
      letter-spacing:1px;">ALPHA SETUP — INFORMATION ONLY</div>
  <div style="font-size:10px;color:#6b7280;margin-top:4px;font-family:'DM Sans',sans-serif;">
    Full gap + sector alignment detected, but HMM regime is <b style="color:#f59e0b;">Chop</b>.
    Monitor this setup — execution disabled until regime clears.
  </div>
</div>""", unsafe_allow_html=True)
                    else:
                        # Build regime sizing badge for the card header
                        if _scan_regime:
                            _rsz_label = _scan_regime_meta.get('sizing', '')
                            _rsz_color = _scan_regime_meta.get('color', '#6b7280')
                            _rsz_pct   = '50%' if _regime_risk_mult < 1.0 else '100%'
                            _regime_sizing_badge = (
                                f'<span style="background:{_scan_regime_meta.get("bg","#1f2937")};'
                                f'color:{_rsz_color};border-radius:4px;padding:1px 7px;'
                                f'font-size:9px;font-family:\'DM Mono\',monospace;">'
                                f'HMM {_scan_regime} · {_rsz_pct} size</span>'
                            )
                        else:
                            _regime_sizing_badge = ''

                        _expander_label = (
                            f"Suggested Execution — {_sym}"
                            + (" [RUNNER]" if any(
                                (lambda ts: ts.get('runner_active', False))(
                                    build_trade_setup(_sym, _dir, tt, float(_price), _gap_type,
                                                      risk_multiplier=_regime_risk_mult)
                                )
                                for tt in ('intraday',)
                            ) else "")
                        )
                        with st.expander(_expander_label, expanded=False):
                            if _regime_sizing_badge:
                                st.markdown(_regime_sizing_badge, unsafe_allow_html=True)
                            _ts_cols = st.columns(2)
                            for _ts_col, _ts_type in zip(_ts_cols, ('intraday', 'swing')):
                                try:
                                    _ts = build_trade_setup(
                                        ticker=_sym, direction=_dir,
                                        trade_type=_ts_type, spot_price=float(_price),
                                        gap_type=_gap_type,
                                        risk_multiplier=_regime_risk_mult,
                                    )
                                    with _ts_col:
                                        # Error: budget too small
                                        if _ts.get('error'):
                                            st.markdown(f"""
<div style="background:#0f0a0a;border:1px solid #7f1d1d;border-radius:12px;padding:14px 16px;">
  <div style="font-size:10px;color:#6b7280;text-transform:uppercase;margin-bottom:8px;">
    {'⚡ Intraday' if _ts_type=='intraday' else '📈 Swing'}
  </div>
  <div style="color:#f87171;font-size:12px;font-family:'DM Mono',monospace;">⚠️ {_ts['error']}</div>
  <div style="font-size:11px;color:#4b5563;margin-top:6px;">
    Min cost: ${_ts['cost_per_contract']:.0f} · Budget: ${_ts['risk_amount']}
  </div>
</div>""", unsafe_allow_html=True)
                                            continue

                                        _ts_clr   = '#4ade80' if _dir == 'LONG' else '#f87171'
                                        _cost_clr = '#22c55e' if _ts['total_cost'] <= _ts['risk_amount'] else '#f59e0b'
                                        _src_badge = (
                                            '<span style="background:#1e3a5f;color:#60a5fa;border-radius:4px;padding:1px 6px;font-size:9px;">LIVE</span>'
                                            if _ts['source'] == 'live' else
                                            '<span style="background:#1f2937;color:#6b7280;border-radius:4px;padding:1px 6px;font-size:9px;">EST.</span>'
                                        )
                                        _runner_badge = (
                                            '<span style="background:#1e1b4b;color:#a5b4fc;border-radius:4px;padding:1px 7px;font-size:9px;">RUNNER</span>'
                                            if _ts['runner_active'] else ''
                                        )
                                        _warn_html = (
                                            f'<div style="color:#f59e0b;font-size:10px;margin-top:6px;">⚠️ {_ts["budget_warning"]}</div>'
                                            if _ts.get('budget_warning') else ''
                                        )

                                        st.markdown(f"""
<div style="background:#07080d;border:1px solid rgba(99,102,241,0.2);border-radius:12px;padding:14px 16px;">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
    <span style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;">
      {'⚡ Intraday (0DTE)' if _ts_type=='intraday' else '📈 Swing (7-14d)'}
    </span>
    {_src_badge} {_runner_badge}
  </div>
  <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;">
    Strategy: <span style="color:#e2e8f0;font-weight:600;">{_ts['strategy']}</span>
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:15px;font-weight:700;color:{_ts_clr};margin-bottom:10px;">
    Action: {_ts['action_str']}
  </div>
  <div style="display:flex;flex-direction:column;gap:5px;font-family:'DM Mono',monospace;font-size:11px;">
    <div style="display:flex;justify-content:space-between;">
      <span style="color:#4b5563;">Strike (ITM ~.70d)</span>
      <span style="color:#e2e8f0;">${_ts['strike']:.2f}</span>
    </div>
    <div style="display:flex;justify-content:space-between;">
      <span style="color:#4b5563;">Expiry</span>
      <span style="color:#e2e8f0;">{_ts['exp_display']} ({_ts['dte']}d)</span>
    </div>
    <div style="display:flex;justify-content:space-between;">
      <span style="color:#4b5563;">Ask / Mid</span>
      <span style="color:#e2e8f0;">${_ts['premium_ask']:.2f} / ${_ts['premium_mid']:.2f}</span>
    </div>
    <div style="border-top:1px solid #1f2937;margin:4px 0;"></div>
    <div style="display:flex;justify-content:space-between;">
      <span style="color:#4b5563;">Size</span>
      <span style="color:{_ts_clr};font-weight:700;">{_ts['size_str']} contract{'s' if _ts['contracts']!=1 else ''}</span>
    </div>
    {'<div style="display:flex;justify-content:space-between;"><span style="color:#4b5563;">Scale-Out (T1)</span><span style="color:#fbbf24;">' + str(_ts['scale_qty']) + ' @ Prior Day High</span></div>' if _ts['runner_active'] else ''}
    {'<div style="display:flex;justify-content:space-between;"><span style="color:#4b5563;">Runner (33% Guard)</span><span style="color:#a5b4fc;">' + str(_ts['runner_qty']) + ' — trail from HOD</span></div>' if _ts['runner_active'] else ''}
    <div style="display:flex;justify-content:space-between;">
      <span style="color:#4b5563;">Est. Cost</span>
      <span style="color:{_cost_clr};font-weight:700;">${_ts['total_cost']:.2f}</span>
    </div>
  </div>
  {_warn_html}
  <div style="margin-top:8px;padding-top:8px;border-top:1px solid #111827;font-size:10px;color:#374151;font-family:'DM Mono',monospace;">
    OCC: {_ts['occ_symbol']}
  </div>
</div>""", unsafe_allow_html=True)

                                        # Register runner in session state
                                        if _ts['runner_active']:
                                            if 'pending_runners' not in st.session_state:
                                                st.session_state.pending_runners = []
                                            _runner_key = _ts['occ_symbol']
                                            if is_admin and _ts_type == 'intraday' and _eod_entry_locked:
                                                st.markdown(
                                                    '<div style="background:#1c0a0a;border:1px solid rgba(239,68,68,0.35);'
                                                    'border-radius:6px;padding:5px 8px;font-family:DM Mono,monospace;'
                                                    'font-size:9px;color:#f87171;text-align:center;">'
                                                    '&#128274; Runner locked after 3:30 PM ET</div>',
                                                    unsafe_allow_html=True,
                                                )
                                            elif is_admin:
                                                if st.button(
                                                    f"Activate Runner — {_sym}",
                                                    key=f"runner_{_runner_key}_{_ts_type}",
                                                    use_container_width=True,
                                                ):
                                                    if 'runner_manager' not in st.session_state:
                                                        from lib.runner_manager import RunnerManager
                                                        st.session_state.runner_manager = RunnerManager()
                                                    st.session_state.runner_manager.add_runner(
                                                        occ_symbol    = _runner_key,
                                                        ticker        = _sym,
                                                        direction     = _ts['contract_type'],
                                                        entry_premium = _ts['premium_ask'],
                                                        qty           = _ts['runner_qty'],
                                                    )
                                                    # Log to Supabase
                                                    if TRADE_LOG_AVAILABLE:
                                                        _entry_ctx = {
                                                            'regime':             _scan_regime,
                                                            'regime_confidence':  _scan_regime_data.get('confidence'),
                                                            'ftfc_aligned':       _r.get('aligned_up') if _r.get('direction') == 'LONG' else _r.get('aligned_down'),
                                                            'ftfc_total':         _r.get('total_tfs'),
                                                            'gap_type':           _gap_type,
                                                            'alpha_setup':        _alpha_setup,
                                                            'sector_etf':         _sector_etf,
                                                            'sentinel_bonus':     _sentinel_bonus,
                                                        }
                                                        log_trade_to_supabase(_ts, _entry_ctx)
                                                    st.success(f"Runner registered: {_runner_key} ({_ts['runner_qty']} contracts)")
                                            else:
                                                st.markdown(
                                                    '<div style="background:#0f172a;border:1px solid rgba(99,102,241,0.12);'
                                                    'border-radius:8px;padding:7px 10px;font-family:DM Mono,monospace;'
                                                    'font-size:10px;color:#374151;text-align:center;">'
                                                    '🔒 Execution Disabled — Read-Only Mode</div>',
                                                    unsafe_allow_html=True,
                                                )
                                except Exception as _ts_err:
                                    _ts_col.caption(f"{_ts_type} unavailable: {_ts_err}")

        # ── Pass 2: generate sparklines and update each placeholder ──────────
        for _ph, _cv in _card_slots:
            _b64 = generate_sparkline_base64(_cv['sym'])
            if _b64:
                _chart_art = (
                    '<img src="data:image/png;base64,' + _b64 + '" '
                    'style="width:100%;height:80px;object-fit:fill;display:block;" />'
                )
                _ph.markdown(_make_card_html(_cv, _chart_art, decayed=_cv.get('_is_decayed', False)), unsafe_allow_html=True)

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
                    if is_admin:
                        ecols = st.columns([1, 1, 2])
                        with ecols[0]:
                            if is_executable_grade(gi):
                                if _eod_entry_locked:
                                    st.markdown(
                                        '<div style="background:#1c0a0a;border:1px solid rgba(239,68,68,0.35);'
                                        'border-radius:6px;padding:5px 8px;font-family:DM Mono,monospace;'
                                        'font-size:9px;color:#f87171;text-align:center;">'
                                        '&#128274; 0DTE LOCKED<br>'
                                        '<span style="color:#7f1d1d;font-size:8px;">Entry closed after 3:30 PM ET</span>'
                                        '</div>',
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    btn_label = f"⚡ 0DTE {direction_r} {sym}"
                                    if st.button(btn_label, key=f"exec_0dte_{sym}", type="primary", use_container_width=True):
                                        st.session_state[f"confirm_0dte_{sym}"] = True
                        with ecols[1]:
                            if is_executable_grade(gs):
                                btn_label = f"📊 SWING {direction_r} {sym}"
                                if st.button(btn_label, key=f"exec_swing_{sym}", use_container_width=True):
                                    st.session_state[f"confirm_swing_{sym}"] = True

                        # 0DTE confirm — hard-blocked if entry window has closed
                        if st.session_state.get(f"confirm_0dte_{sym}"):
                            if _eod_entry_locked:
                                st.session_state[f"confirm_0dte_{sym}"] = False
                                st.error("0DTE entry window closed at 3:30 PM ET — order cancelled.", icon="🔒")
                            else:
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
                    else:
                        st.markdown(
                            '<div style="background:#0f172a;border:1px solid rgba(99,102,241,0.12);'
                            'border-radius:8px;padding:8px 14px;font-family:DM Mono,monospace;'
                            'font-size:11px;color:#374151;text-align:center;margin:4px 0;">'
                            '🔒 Execution Disabled — Read-Only Mode</div>',
                            unsafe_allow_html=True,
                        )

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
        if is_admin:
            if st.button("🚨 KILL SWITCH — CLOSE ALL", type="primary", use_container_width=True):
                st.session_state['confirm_kill'] = True
        else:
            st.markdown(
                '<div style="background:#0f172a;border:1px solid rgba(99,102,241,0.12);'
                'border-radius:8px;padding:8px 10px;font-family:DM Mono,monospace;'
                'font-size:10px;color:#374151;text-align:center;">'
                '🔒 Execution Disabled — Read-Only Mode</div>',
                unsafe_allow_html=True,
            )
    if is_admin and st.session_state.get('confirm_kill'):
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

    # ── Runner Monitor ────────────────────────────────────────────────────────
    if 'runner_manager' not in st.session_state:
        from lib.runner_manager import RunnerManager
        st.session_state.runner_manager = RunnerManager()

    _rm = st.session_state.runner_manager
    _active_runners = _rm.active_runners()
    _closed_runners = _rm.closed_runners()

    if _active_runners or _closed_runners:
        st.markdown("<div class='section-header' style='margin-top:24px;'>🏃 Runner Monitor — 33% Profit Guard</div>", unsafe_allow_html=True)

        if _active_runners:
            _rm_refresh = st.button("🔄 Refresh Runner Prices", use_container_width=False)

            for _run in _active_runners:
                _run_sym   = _run['symbol']
                _run_entry = _run['entry']
                _run_hod   = _run['hod']
                _run_trig  = _run['trigger']
                _run_dir   = _run['direction']

                # Try to get current premium from Alpaca if refresh clicked
                _run_current = _run_hod   # default to HOD if no live price
                _run_action  = None
                if _rm_refresh:
                    try:
                        from alpaca.data.historical.option import OptionHistoricalDataClient
                        from alpaca.data.requests import OptionSnapshotRequest
                        from config import get_alpaca_keys
                        _rk, _rs = get_alpaca_keys()
                        _oc = OptionHistoricalDataClient(api_key=_rk, secret_key=_rs)
                        _snaps = _oc.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=_run_sym))
                        _snap  = (_snaps or {}).get(_run_sym)
                        if _snap and _snap.latest_trade:
                            _run_current = float(_snap.latest_trade.price)
                        _run_action = _rm.update(_run_sym, _run_current)
                    except Exception:
                        pass

                _gain_pct    = round((_run_hod / _run_entry - 1) * 100, 1)
                _cushion_pct = round((_run_current / _run_hod * 100 - 67), 1) if _run_hod else 0
                _cushion_clr = '#22c55e' if _cushion_pct > 15 else ('#f59e0b' if _cushion_pct > 5 else '#ef4444')

                if _run_action and _run_action.get('action') == 'close':
                    st.error(f"CLOSE RUNNER NOW — {_run['ticker']} | {_run_action['reason']}")
                    log_msg = _run_action.get('log_msg', '')
                    if log_msg and 'order_log' in st.session_state:
                        st.session_state.order_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'runner_exit', 'symbol': _run_sym,
                            'message': log_msg,
                        })
                        if len(st.session_state.order_log) > 500:
                            st.session_state.order_log = st.session_state.order_log[-500:]
                    # Log exit to Supabase — include peak_premium so we can measure "meat on the bone"
                    if TRADE_LOG_AVAILABLE:
                        log_trade_exit(
                            occ_symbol    = _run_sym,
                            exit_premium  = float(_run_action.get('current', _run_entry)),
                            exit_reason   = _run_action.get('reason', '33% Guard'),
                            peak_premium  = float(_run_action.get('peak', _run_hod)),
                            entry_premium = float(_run_entry),
                        )

                st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1117,#0f172a);border:1px solid rgba(165,180,252,0.2);
    border-radius:14px;padding:16px 20px;margin:6px 0;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
    <span style="font-size:18px;">🏃</span>
    <span style="font-family:'DM Mono',monospace;font-size:16px;font-weight:700;color:#e2e8f0;">
      {_run['ticker']} {_run_dir}
    </span>
    <span style="font-size:11px;color:#6b7280;font-family:'DM Mono',monospace;">{_run_sym}</span>
    <span style="background:#1e1b4b;color:#a5b4fc;border-radius:6px;padding:2px 8px;font-size:10px;margin-left:auto;">
      {_run['qty']} contract{'s' if _run['qty']!=1 else ''} · entered {_run['entry_time']}
    </span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
    <div style="text-align:center;background:#07080d;border-radius:8px;padding:10px;">
      <div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:1px;">Entry</div>
      <div style="font-family:'DM Mono',monospace;font-size:16px;color:#94a3b8;">${_run_entry:.2f}</div>
    </div>
    <div style="text-align:center;background:#07080d;border-radius:8px;padding:10px;">
      <div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:1px;">Current Peak (HOD)</div>
      <div style="font-family:'DM Mono',monospace;font-size:16px;color:#4ade80;">${_run_hod:.2f}</div>
      <div style="font-size:10px;color:#22c55e;">+{_gain_pct:.1f}% from entry</div>
    </div>
    <div style="text-align:center;background:#07080d;border-radius:8px;padding:10px;">
      <div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:1px;">33% Trigger</div>
      <div style="font-family:'DM Mono',monospace;font-size:16px;color:#f59e0b;">${_run_trig:.2f}</div>
      <div style="font-size:10px;color:#6b7280;">= HOD × 0.67</div>
    </div>
    <div style="text-align:center;background:#07080d;border-radius:8px;padding:10px;">
      <div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:1px;">Cushion</div>
      <div style="font-family:'DM Mono',monospace;font-size:16px;color:{_cushion_clr};">{_cushion_pct:.1f}%</div>
      <div style="font-size:10px;color:#6b7280;">above trigger</div>
    </div>
  </div>
  {'<div style="margin-top:10px;background:#1a1505;border-radius:6px;padding:8px 12px;font-size:11px;color:#fbbf24;font-family:DM Mono,monospace;">✅ T1 Hit — Breakeven Stop Active at $' + f"{_run_entry:.2f}" + '</div>' if _run['t1_hit'] else ''}
</div>""", unsafe_allow_html=True)

                if is_admin and st.button(f"Close Runner — {_run['ticker']}", key=f"close_runner_{_run_sym}"):
                    if exe:
                        exe.close_position(_run_sym)
                    _rm.runners[_run_sym].closed = True
                    _rm.runners[_run_sym].close_reason = "Manual Close"
                    if 'order_log' in st.session_state:
                        st.session_state.order_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'runner_exit', 'symbol': _run_sym,
                            'message': f"Runner Closed: Manual (Peak ${_run_hod:.2f})",
                        })
                        if len(st.session_state.order_log) > 500:
                            st.session_state.order_log = st.session_state.order_log[-500:]
                    if TRADE_LOG_AVAILABLE:
                        log_trade_exit(
                            occ_symbol    = _run_sym,
                            exit_premium  = float(_run_current),
                            exit_reason   = 'Manual Close',
                            peak_premium  = float(_run_hod),
                            entry_premium = float(_run_entry),
                        )
                    st.success("Runner closed.")
                    st.rerun()

        if _closed_runners:
            with st.expander(f"📋 Closed Runners ({len(_closed_runners)})", expanded=False):
                for _cr in _closed_runners:
                    st.markdown(
                        f"**{_cr['ticker']}** `{_cr['symbol']}` — "
                        f"Entry: ${_cr['entry']:.2f} · HOD: ${_cr['hod']:.2f} · "
                        f"Trigger: ${_cr['trigger']:.2f} | _{_cr['close_reason']}_"
                    )

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
            if is_admin:
                if st.button(f"Close {p['symbol']}", key=f"close_{p['symbol']}", use_container_width=False):
                    if exe.close_position(p['symbol']):
                        st.success(f"✅ {p['symbol']} position closed.")
                        st.session_state['_acct_cache_time'] = 0
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to close {p['symbol']}")
            else:
                st.markdown(
                    '<span style="background:#0f172a;border:1px solid rgba(99,102,241,0.12);'
                    'border-radius:6px;padding:5px 10px;font-family:DM Mono,monospace;'
                    'font-size:10px;color:#374151;">🔒 Read-Only</span>',
                    unsafe_allow_html=True,
                )
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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PERFORMANCE — Self-Improving Loop Analytics
# ══════════════════════════════════════════════════════════════════════════════

elif page == "performance":
    show_tab_header("performance", "Performance Analytics", datetime.now().strftime('%B %d, %Y'))

    # ── Migration guard ───────────────────────────────────────────────────────
    _raw_trades = []
    _tl_error   = None
    if not TRADE_LOG_AVAILABLE:
        st.error("Trade log unavailable — executor import failed.")
    else:
        try:
            _raw_trades = fetch_trade_log(limit=500)
        except Exception as _e:
            _tl_error = str(_e)

    if _tl_error and 'PGRST205' in str(_tl_error):
        st.markdown("""
<div style="background:#1c1508;border:1px solid rgba(251,191,36,0.4);border-radius:14px;
    padding:20px 24px;margin-bottom:16px;">
  <div style="font-size:14px;color:#f59e0b;font-weight:700;font-family:'DM Mono',monospace;
      margin-bottom:8px;">Database Migration Required</div>
  <div style="font-size:13px;color:#94a3b8;font-family:'DM Sans',sans-serif;line-height:1.6;">
    The <code>trade_log</code> table does not exist yet.<br>
    To activate trade logging:<br>
    1. Open your Supabase dashboard<br>
    2. Go to <b>SQL Editor</b> → New Query<br>
    3. Paste the contents of <code>supabase_migration.sql</code> → Run<br>
    4. Reload this page
  </div>
</div>""", unsafe_allow_html=True)
        st.code(open(
            os.path.join(os.path.dirname(__file__), 'supabase_migration.sql'),
            encoding='utf-8'
        ).read(), language='sql')
        st.stop()

    # ── No data yet ───────────────────────────────────────────────────────────
    if not _raw_trades:
        st.markdown("""
<div style="background:linear-gradient(135deg,#111827,#0f172a);border:1px solid rgba(99,102,241,0.15);
    border-radius:14px;padding:32px;text-align:center;margin-top:24px;">
  <div style="font-size:32px;margin-bottom:12px;">🏆</div>
  <div style="font-size:16px;color:#e2e8f0;font-family:'DM Mono',monospace;margin-bottom:8px;">
    No trades logged yet
  </div>
  <div style="font-size:13px;color:#4b5563;font-family:'DM Sans',sans-serif;">
    Execute your first paper trade from the Scanner tab to begin the self-improving loop.
  </div>
</div>""", unsafe_allow_html=True)
        st.stop()

    import pandas as _pd_perf

    _df = _pd_perf.DataFrame(_raw_trades)
    _df['entered_at'] = _pd_perf.to_datetime(_df['entered_at'], utc=True, errors='coerce')
    _df['exited_at']  = _pd_perf.to_datetime(_df.get('exited_at'), utc=True, errors='coerce')

    _closed = _df[_df['status'] == 'closed'].copy()
    _open   = _df[_df['status'] == 'open'].copy()

    # ── Summary row ───────────────────────────────────────────────────────────
    _total  = len(_df)
    _n_cl   = len(_closed)
    _n_op   = len(_open)

    if _n_cl > 0:
        _wins       = (_closed['pnl_dollars'] > 0).sum()
        _win_rate   = round(_wins / _n_cl * 100, 1)
        _gross_gain = _closed.loc[_closed['pnl_dollars'] > 0, 'pnl_dollars'].sum()
        _gross_loss = abs(_closed.loc[_closed['pnl_dollars'] < 0, 'pnl_dollars'].sum())
        _pf         = round(_gross_gain / _gross_loss, 2) if _gross_loss > 0 else float('inf')
        _net_pnl    = round(_closed['pnl_dollars'].sum(), 2)
        _avg_peak   = round(_closed['peak_premium'].dropna().mean(), 4) if 'peak_premium' in _closed.columns else None
    else:
        _win_rate = _pf = _net_pnl = _gross_gain = _gross_loss = 0.0
        _avg_peak = None

    _pnl_color  = '#4ade80' if _net_pnl >= 0 else '#f87171'
    _pf_color   = '#4ade80' if _pf >= 1.5 else ('#f59e0b' if _pf >= 1.0 else '#ef4444')
    _wr_color   = '#4ade80' if _win_rate >= 55 else ('#f59e0b' if _win_rate >= 40 else '#ef4444')
    _pf_display = f"{_pf:.2f}x" if _pf != float('inf') else "inf"

    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:24px;">
  <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px;text-align:center;">
    <div style="font-size:9px;color:#4b5563;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">Trades Logged</div>
    <div style="font-family:'DM Mono',monospace;font-size:24px;font-weight:700;color:#e2e8f0;">{_total}</div>
    <div style="font-size:10px;color:#4b5563;">{_n_op} open · {_n_cl} closed</div>
  </div>
  <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px;text-align:center;">
    <div style="font-size:9px;color:#4b5563;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">Win Rate</div>
    <div style="font-family:'DM Mono',monospace;font-size:24px;font-weight:700;color:{_wr_color};">{_win_rate}%</div>
    <div style="font-size:10px;color:#4b5563;">{int(_wins) if _n_cl>0 else 0}W / {_n_cl - int(_wins) if _n_cl>0 else 0}L</div>
  </div>
  <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px;text-align:center;">
    <div style="font-size:9px;color:#4b5563;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">Profit Factor</div>
    <div style="font-family:'DM Mono',monospace;font-size:24px;font-weight:700;color:{_pf_color};">{_pf_display}</div>
    <div style="font-size:10px;color:#4b5563;">Gains / Losses</div>
  </div>
  <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px;text-align:center;">
    <div style="font-size:9px;color:#4b5563;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">Net P&L</div>
    <div style="font-family:'DM Mono',monospace;font-size:24px;font-weight:700;color:{_pnl_color};">{"+" if _net_pnl>=0 else ""}${_net_pnl:.2f}</div>
    <div style="font-size:10px;color:#4b5563;">paper trades</div>
  </div>
  <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px;text-align:center;">
    <div style="font-size:9px;color:#4b5563;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">Avg Peak</div>
    <div style="font-family:'DM Mono',monospace;font-size:24px;font-weight:700;color:#a5b4fc;">${f"{_avg_peak:.4f}" if _avg_peak else "—"}</div>
    <div style="font-size:10px;color:#4b5563;">HOD per runner</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Win Rate by Regime ────────────────────────────────────────────────────
    st.markdown("<div class='section-header' style='margin-top:8px;'>Win Rate by Regime</div>",
                unsafe_allow_html=True)

    _REGIME_ORDER  = ['Bull Quiet', 'Bull Volatile', 'Bear Quiet', 'Bear Volatile', 'Chop']
    _REGIME_COLORS = {
        'Bull Quiet':    '#4ade80', 'Bull Volatile':  '#86efac',
        'Bear Quiet':    '#f87171', 'Bear Volatile':  '#fca5a5',
        'Chop':          '#f59e0b',
    }

    if _n_cl > 0 and 'regime' in _closed.columns:
        _by_regime = (
            _closed.groupby('regime', dropna=False)
            .agg(
                total    = ('pnl_dollars', 'count'),
                wins     = ('pnl_dollars', lambda x: (x > 0).sum()),
                net_pnl  = ('pnl_dollars', 'sum'),
                avg_peak = ('peak_premium', 'mean'),
            )
            .reset_index()
        )
        _by_regime['win_rate'] = (_by_regime['wins'] / _by_regime['total'] * 100).round(1)

        _r_cols = st.columns(min(len(_by_regime), 5))
        for _i, (_rc, (__, _row)) in enumerate(zip(_r_cols, _by_regime.iterrows())):
            _rname   = str(_row['regime']) if _row['regime'] else 'Unknown'
            _rcolor  = _REGIME_COLORS.get(_rname, '#6b7280')
            _rwr     = _row['win_rate']
            _rwr_clr = '#4ade80' if _rwr >= 55 else ('#f59e0b' if _rwr >= 40 else '#ef4444')
            _rnpnl   = round(float(_row['net_pnl']), 2)
            _rnpnl_c = '#4ade80' if _rnpnl >= 0 else '#f87171'
            _rpeak   = f"${float(_row['avg_peak']):.4f}" if not _pd_perf.isna(_row['avg_peak']) else '—'
            with _rc:
                st.markdown(f"""
<div style="background:#0f172a;border:1px solid {_rcolor}44;border-radius:12px;padding:14px;text-align:center;">
  <div style="font-size:10px;color:{_rcolor};font-weight:700;font-family:'DM Mono',monospace;
      margin-bottom:8px;">{_rname}</div>
  <div style="font-size:22px;font-weight:700;color:{_rwr_clr};font-family:'DM Mono',monospace;">{_rwr}%</div>
  <div style="font-size:10px;color:#4b5563;margin-top:2px;">{int(_row['wins'])}W / {int(_row['total']-_row['wins'])}L</div>
  <div style="border-top:1px solid #1f2937;margin:8px 0;"></div>
  <div style="font-size:10px;color:{_rnpnl_c};font-family:'DM Mono',monospace;">
    {"+" if _rnpnl>=0 else ""}${_rnpnl:.2f} net
  </div>
  <div style="font-size:9px;color:#4b5563;margin-top:2px;">Avg Peak: {_rpeak}</div>
</div>""", unsafe_allow_html=True)
    else:
        st.info("No closed trades yet — win rate will populate after your first exit.")

    # ── Alpha Setup performance ───────────────────────────────────────────────
    if _n_cl > 0 and 'alpha_setup' in _closed.columns:
        _alpha_cl = _closed[_closed['alpha_setup'] == True]
        _non_alpha = _closed[_closed['alpha_setup'] != True]
        if len(_alpha_cl) > 0 or len(_non_alpha) > 0:
            st.markdown("<div class='section-header' style='margin-top:24px;'>Alpha Setup vs Standard</div>",
                        unsafe_allow_html=True)
            _ac1, _ac2 = st.columns(2)
            for _ac, _subset, _label, _clr in [
                (_ac1, _alpha_cl,  "Alpha Setups",   '#fb923c'),
                (_ac2, _non_alpha, "Standard Setups", '#6366f1'),
            ]:
                with _ac:
                    if len(_subset) == 0:
                        st.markdown(f"<div style='color:#4b5563;text-align:center;padding:20px;'>{_label}: no data</div>",
                                    unsafe_allow_html=True)
                        continue
                    _aw   = (_subset['pnl_dollars'] > 0).sum()
                    _awr  = round(_aw / len(_subset) * 100, 1)
                    _anpnl = round(_subset['pnl_dollars'].sum(), 2)
                    _aw_c = '#4ade80' if _awr >= 55 else ('#f59e0b' if _awr >= 40 else '#ef4444')
                    st.markdown(f"""
<div style="background:#0f172a;border:1px solid {_clr}44;border-radius:12px;padding:16px;text-align:center;">
  <div style="font-size:10px;color:{_clr};font-weight:700;font-family:'DM Mono',monospace;margin-bottom:8px;">{_label}</div>
  <div style="font-size:26px;font-weight:700;color:{_aw_c};font-family:'DM Mono',monospace;">{_awr}%</div>
  <div style="font-size:10px;color:#4b5563;">{int(_aw)}W / {int(len(_subset)-_aw)}L · {len(_subset)} total</div>
  <div style="font-size:11px;color:{'#4ade80' if _anpnl>=0 else '#f87171'};margin-top:6px;font-family:'DM Mono',monospace;">
    Net: {"+" if _anpnl>=0 else ""}${_anpnl:.2f}
  </div>
</div>""", unsafe_allow_html=True)

    # ── Peak vs Exit ("Meat on the Bone") ─────────────────────────────────────
    _guard_trades = _closed[
        (_closed['peak_premium'].notna()) & (_closed['exit_premium'].notna()) &
        (_closed['entry_premium'].notna())
    ].copy() if _n_cl > 0 else _pd_perf.DataFrame()

    if len(_guard_trades) > 0:
        st.markdown("<div class='section-header' style='margin-top:24px;'>Profit Guard — Meat on the Bone</div>",
                    unsafe_allow_html=True)
        st.caption("Compares HOD (peak) vs actual exit to show how much was left behind by the 33% pullback rule.")
        _guard_trades['captured_pct'] = (
            (_guard_trades['exit_premium'] - _guard_trades['entry_premium']) /
            (_guard_trades['peak_premium'] - _guard_trades['entry_premium'])
            * 100
        ).clip(0, 200).round(1)
        _avg_cap = round(_guard_trades['captured_pct'].mean(), 1)
        st.markdown(f"""
<div style="background:#111827;border:1px solid rgba(165,180,252,0.2);border-radius:12px;
    padding:14px 18px;margin-bottom:12px;">
  <div style="font-size:10px;color:#6b7280;margin-bottom:4px;">Average Move Captured</div>
  <div style="font-family:'DM Mono',monospace;font-size:20px;font-weight:700;color:#a5b4fc;">{_avg_cap}%</div>
  <div style="font-size:10px;color:#4b5563;margin-top:2px;">of peak-to-entry move retained at exit</div>
</div>""", unsafe_allow_html=True)
        _gt_display = _guard_trades[['ticker', 'entry_premium', 'peak_premium', 'exit_premium', 'captured_pct', 'exit_reason']].copy()
        _gt_display.columns = ['Ticker', 'Entry $', 'Peak $', 'Exit $', 'Captured %', 'Reason']
        st.dataframe(_gt_display.style.format({
            'Entry $': '{:.4f}', 'Peak $': '{:.4f}', 'Exit $': '{:.4f}', 'Captured %': '{:.1f}',
        }), use_container_width=True, hide_index=True)

    # ── Recent trades log ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header' style='margin-top:24px;'>Recent Trades</div>",
                unsafe_allow_html=True)

    _display_cols = [c for c in [
        'entered_at', 'ticker', 'contract_type', 'strike', 'trade_type',
        'regime', 'ftfc_aligned', 'ftfc_total', 'gap_type', 'alpha_setup',
        'entry_premium', 'peak_premium', 'exit_premium', 'pnl_dollars', 'pnl_pct',
        'exit_reason', 'status',
    ] if c in _df.columns]

    _df_display = _df[_display_cols].copy()
    if 'entered_at' in _df_display.columns:
        _df_display['entered_at'] = _df_display['entered_at'].dt.strftime('%m/%d %H:%M').fillna('—')

    # Color the pnl_dollars column
    def _color_pnl(val):
        if val is None or (isinstance(val, float) and _pd_perf.isna(val)):
            return ''
        return 'color: #4ade80' if val > 0 else ('color: #f87171' if val < 0 else '')

    st.dataframe(
        _df_display.style.map(_color_pnl, subset=['pnl_dollars'] if 'pnl_dollars' in _df_display.columns else []),
        use_container_width=True,
        hide_index=True,
    )

    # ── Clear log ─────────────────────────────────────────────────────────────
    with st.expander("Danger Zone", expanded=False):
        if is_admin:
            if st.button("Clear All Trade Logs", type="secondary"):
                try:
                    from config import supabase as _sb
                    _sb.table('trade_log').delete().gte('id', 0).execute()
                    st.success("Trade log cleared.")
                    st.rerun()
                except Exception as _ce:
                    st.error(f"Clear failed: {_ce}")
        else:
            st.markdown(
                '<div style="background:#0f172a;border:1px solid rgba(99,102,241,0.12);'
                'border-radius:8px;padding:8px 14px;font-family:DM Mono,monospace;'
                'font-size:11px;color:#374151;text-align:center;">'
                '🔒 Execution Disabled — Read-Only Mode</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATABASE CONNECTION TEST
# ══════════════════════════════════════════════════════════════════════════════

elif page == "db_test":
    show_tab_header("db_test", "🗄️ Database Connection Test", "Supabase Live Read")

    from lib.db import fetch_recent_trades, get_db

    # ── Mock data definition ──────────────────────────────────────────────────
    def _mock_pnl_rows() -> list[dict]:
        now = datetime.now()
        def _row(ticker, direction, trade_type, grade, regime, entry, exit_, qty, note, days_ago, hour):
            # entry/exit are option premiums; P&L is purely premium-in vs premium-out
            pnl_pct    = round((exit_ - entry) / entry * 100, 3)
            pnl_dollars = round((exit_ - entry) * qty * 100, 2)
            ts = (now - timedelta(days=days_ago)).replace(hour=hour, minute=15, second=0, microsecond=0)
            return {
                "ticker": ticker, "direction": direction, "trade_type": trade_type,
                "grade": grade, "regime": regime,
                "entry_price": entry, "exit_price": exit_, "qty": qty,
                "pnl_pct": pnl_pct, "pnl_dollars": pnl_dollars,
                "win": pnl_pct > 0, "note": note,
                "timestamp": ts.isoformat(), "date": ts.date().isoformat(),
                "hour": hour, "day_of_week": ts.strftime("%A"),
            }
        return [
            _row("NVDA", "LONG",  "0DTE_CALL",  "A+", "Bull_Quiet",    3.20,  8.75, 2, "Breakout on vol surge — rode to target",  1, 10),
            _row("SPY",  "SHORT", "0DTE_PUT",   "A",  "Bear_Volatile", 2.15,  0.60, 3, "SPY reclaimed VWAP, put collapsed",       2, 11),
            _row("TSLA", "LONG",  "SWING_LONG", "A",  "Bull_Quiet",    6.40, 11.25, 2, "Held through catalyst, 2-1-2 confirmed",  3,  9),
            _row("QQQ",  "SHORT", "0DTE_PUT",   "B",  "Bear_Quiet",    1.80,  3.95, 2, "Clean 3-2 continuation breakdown",        4, 14),
            _row("AAPL", "LONG",  "0DTE_CALL",  "A",  "Bull_Volatile", 4.50,  1.80, 2, "Stopped out on macro spike reversal",     5, 10),
        ]

    def _mock_tracker_rows() -> list[dict]:
        now = datetime.now()
        def _row(ticker, grade, direction, regime, score, entry, current, trade_type, patterns, note, mins_ago):
            # entry/current are option premiums
            pnl_pct    = round((current - entry) / entry * 100, 3)
            pnl_dollars = round((current - entry) * 100, 2)
            added = (now - timedelta(minutes=mins_ago)).isoformat()
            return {
                "ticker": ticker, "grade": grade, "direction": direction,
                "regime": regime, "signal_score": score,
                "entry_price": entry, "trade_type": trade_type,
                "patterns": patterns, "note": note,
                "added_at": added, "active": True,
                "current_price": current,
                "pnl_dollars": pnl_dollars, "pnl_pct": pnl_pct,
                "last_updated": now.isoformat(),
            }
        return [
            _row("NVDA", "A+", "LONG",  "Bull_Quiet",    85, 4.50, 6.80, "0DTE",  [{"name": "2-1-2 Cont", "direction": "up",   "grade": "A+"}], "Running — at +51%, watching $7",  42),
            _row("SPY",  "A",  "SHORT", "Bear_Volatile", 72, 3.80, 5.15, "SWING", [{"name": "3-2 Cont",   "direction": "down", "grade": "A"}],  "Put gaining as SPY breaks down",  18),
            _row("MU",   "A",  "LONG",  "Bull_Quiet",    68, 1.95, 1.30, "0DTE",  [{"name": "2-1-2 Rev",  "direction": "up",   "grade": "A"}],  "Gone against — monitoring stop",  95),
            _row("QQQ",  "B",  "LONG",  "Bull_Volatile", 54, 7.20, 7.85, "SWING", [{"name": "2-2 Rev",    "direction": "up",   "grade": "B"}],  "Small size in chop — up slightly", 130),
        ]

    # ── Generate Mock Data button ─────────────────────────────────────────────
    st.markdown("#### Seed Test Data")
    mock_col1, mock_col2, mock_col3 = st.columns([2, 2, 4])

    if is_admin:
        with mock_col1:
            gen_clicked = st.button("⚡ Generate Mock Data", use_container_width=True, type="primary")
        with mock_col2:
            clear_clicked = st.button("🗑️ Clear Mock Data", use_container_width=True, type="secondary")
    else:
        gen_clicked = False
        clear_clicked = False
        with mock_col1:
            st.markdown(
                '<div style="background:#0f172a;border:1px solid rgba(99,102,241,0.12);'
                'border-radius:8px;padding:8px 14px;font-family:DM Mono,monospace;'
                'font-size:11px;color:#374151;text-align:center;">'
                '🔒 Execution Disabled — Read-Only Mode</div>',
                unsafe_allow_html=True,
            )

    if gen_clicked:
        db = get_db()
        errors = []
        try:
            db.table("pnl_history").insert(_mock_pnl_rows()).execute()
        except Exception as e:
            errors.append(f"pnl_history: {e}")
        try:
            db.table("tracker_positions").insert(_mock_tracker_rows()).execute()
        except Exception as e:
            errors.append(f"tracker_positions: {e}")

        if errors:
            st.error("Insert failed: " + " | ".join(errors))
        else:
            st.success("✅ Inserted 5 rows → pnl_history  and  4 rows → tracker_positions. Select a table below to view.")
            st.rerun()

    if clear_clicked:
        db = get_db()
        errors = []
        try:
            db.table("pnl_history").delete().neq("id", 0).execute()
        except Exception as e:
            errors.append(f"pnl_history: {e}")
        try:
            db.table("tracker_positions").delete().neq("id", 0).execute()
        except Exception as e:
            errors.append(f"tracker_positions: {e}")

        if errors:
            st.error("Clear failed: " + " | ".join(errors))
        else:
            st.success("🗑️ All rows cleared from pnl_history and tracker_positions.")
            st.rerun()

    st.markdown("---")

    # ── Table selector ────────────────────────────────────────────────────────
    TABLE_META = {
        "pnl_history":       {"date_col": "date",      "order_col": "timestamp",  "label": "Trade Log (P&L History)"},
        "tracker_positions": {"date_col": None,         "order_col": "added_at",   "label": "Tracker Positions"},
        "alerts":            {"date_col": None,         "order_col": "timestamp",  "label": "Alerts"},
        "scan_history":      {"date_col": "scan_date",  "order_col": "scan_date",  "label": "Scan History"},
        "universe_members":  {"date_col": None,         "order_col": "added_at",   "label": "Universe Members"},
    }

    selected_table = st.selectbox(
        "Select table to inspect",
        list(TABLE_META.keys()),
        format_func=lambda k: TABLE_META[k]["label"],
    )
    row_limit = st.slider("Rows to fetch", min_value=5, max_value=100, value=20, step=5)

    meta = TABLE_META[selected_table]

    with st.spinner(f"Reading from `{selected_table}`..."):
        rows, source_label = fetch_recent_trades(
            table=selected_table,
            limit=row_limit,
            date_col=meta["date_col"],
            order_col=meta["order_col"],
        )

    # ── Status banner ────────────────────────────────────────────────────────
    if rows:
        source_color = "#22c55e" if source_label == "today" else "#f59e0b"
        source_icon  = "✅" if source_label == "today" else "📅"
        st.markdown(
            f"<div style='background:#0d1f12;border:1px solid {source_color}33;"
            f"border-radius:10px;padding:14px 18px;margin-bottom:16px;'>"
            f"<span style='color:{source_color};font-weight:700;font-size:14px;'>"
            f"{source_icon} Supabase connected — showing <code style='color:{source_color}'>{source_label}</code> "
            f"data &nbsp;·&nbsp; {len(rows)} row{'s' if len(rows) != 1 else ''} from "
            f"<code style='color:{source_color}'>{selected_table}</code></span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:#1f0d0d;border:1px solid #ef444433;"
            "border-radius:10px;padding:14px 18px;margin-bottom:16px;'>"
            "<span style='color:#ef4444;font-weight:700;font-size:14px;'>"
            "⚠️ No data found — table may be empty or RLS is blocking reads.</span></div>",
            unsafe_allow_html=True,
        )

    # ── Data table ───────────────────────────────────────────────────────────
    if rows:
        df = pd.DataFrame(rows)

        # Number formatting
        fmt = {}
        for col in ["entry_price", "exit_price", "current_price", "pnl_dollars"]:
            if col in df.columns:
                fmt[col] = "${:.2f}"
        if "pnl_pct" in df.columns:
            fmt["pnl_pct"] = "{:+.2f}%"

        styled = df.style.format(fmt, na_rep="—")

        # Green / red colouring on P&L columns
        def _pnl_colour(val):
            try:
                v = float(str(val).replace("$", "").replace("%", "").replace("+", ""))
            except (ValueError, TypeError):
                return ""
            return "color: #22c55e; font-weight: 600" if v > 0 else "color: #ef4444; font-weight: 600"

        for col in ["pnl_dollars", "pnl_pct"]:
            if col in df.columns:
                styled = styled.map(_pnl_colour, subset=[col])

        if "win" in df.columns:
            styled = styled.map(
                lambda v: "color: #22c55e" if v is True else ("color: #ef4444" if v is False else ""),
                subset=["win"],
            )

        # st.dataframe is interactive by default — columns are drag-to-reorder
        st.dataframe(styled, use_container_width=True, height=min(600, 40 + len(df) * 35))
        st.caption(f"{len(df)} rows · {len(df.columns)} columns · drag column headers to reorder · source: {source_label}")
    else:
        st.info("Nothing to display. Click ⚡ Generate Mock Data above, then select a table.")
