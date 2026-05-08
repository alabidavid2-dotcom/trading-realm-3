# ================================================
# WALKFORWARD.PY - Walk-Forward Analysis Engine
# Honest out-of-sample testing that exposes curve-fitting
#
# Method:
#   1. Train on 1-year window — optimize for best Sharpe
#   2. Test blindly on next window (intraday=3mo, swing=18mo)
#   3. Slide forward and repeat
#   4. Stitch only blind test results for honest performance
#
# Transaction Costs:
#   Exchange fee: 0.10% per trade
#   Slippage:     0.05% per trade
#   Total:        0.15% per trade (each way = 0.30% round trip)
# ================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── COSTS ───
EXCHANGE_FEE = 0.0010   # 0.10%
SLIPPAGE     = 0.0005   # 0.05%
TOTAL_COST   = 0.0015   # 0.15% per trade (one way)
ROUND_TRIP   = 0.0030   # 0.30% entry + exit

# ─── WINDOW SIZES (trading days) ───
TRAIN_DAYS        = 252       # ~1 year
TEST_INTRADAY     = 63        # ~3 months
TEST_SWING        = 378       # ~18 months
STEP_INTRADAY     = 63        # Slide by test size
STEP_SWING        = 126       # Slide 6 months for swing


# ─────────────────────────────────────────────
# INDICATOR + SIGNAL GENERATION (same as main system)
# ─────────────────────────────────────────────

def add_indicators(df):
    """Compute all indicators needed for signal generation."""
    df = df.copy()

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).ewm(span=14, adjust=False).mean()
    loss = (-delta).where(delta < 0, 0.0).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)

    # ADX
    h, lo, c = df['High'], df['Low'], df['Close']
    pdm = h.diff(); mdm = -lo.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    tr = pd.concat([h-lo, (h-c.shift(1)).abs(), (lo-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    pdi = 100 * (pdm.ewm(span=14, adjust=False).mean() / atr.replace(0, np.nan))
    mdi = 100 * (mdm.ewm(span=14, adjust=False).mean() / atr.replace(0, np.nan))
    dx = (abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)) * 100
    df['adx'] = dx.ewm(span=14, adjust=False).mean()
    df['adx_rising'] = df['adx'] > df['adx'].shift(1)

    # MACD
    ef = df['Close'].ewm(span=12, adjust=False).mean()
    es = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ef - es
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))

    # Momentum
    df['momentum'] = df['Close'].pct_change(10) * 100

    # ATR
    df['atr'] = tr.rolling(14).mean()

    # Volume
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan)
    else:
        df['vol_ratio'] = 1.0

    # Regime proxy (quantile-based for speed in backtesting)
    df['daily_return'] = df['Close'].pct_change()
    df['vol_20'] = df['daily_return'].rolling(20).std() * np.sqrt(252)

    return df


def assign_regime(df):
    """Quick regime assignment for backtesting."""
    df = df.copy()
    ret_q = df['daily_return'].quantile([0.15, 0.85])
    vol_q = df['vol_20'].quantile(0.5)

    def classify(row):
        r, v = row['daily_return'], row['vol_20']
        if pd.isna(r) or pd.isna(v): return "Chop"
        if r > ret_q[0.85] and v < vol_q: return "Bull_Quiet"
        elif r > ret_q[0.85]: return "Bull_Volatile"
        elif r < ret_q[0.15] and v < vol_q: return "Bear_Quiet"
        elif r < ret_q[0.15]: return "Bear_Volatile"
        else: return "Chop"

    df['regime'] = df.apply(classify, axis=1)
    return df


def add_strat(df):
    """Add Strat candle classification."""
    df = df.copy()
    df['strat_type'] = 0
    df['strat_dir'] = 'neutral'
    for i in range(1, len(df)):
        prev, curr = df.iloc[i-1], df.iloc[i]
        bh = curr['High'] > prev['High']
        bl = curr['Low'] < prev['Low']
        inside = curr['Low'] >= prev['Low'] and curr['High'] <= prev['High']
        if inside:
            df.iloc[i, df.columns.get_loc('strat_type')] = 1
        elif bh and bl:
            df.iloc[i, df.columns.get_loc('strat_type')] = 3
            df.iloc[i, df.columns.get_loc('strat_dir')] = 'up' if curr['Close'] > curr['Open'] else 'down'
        elif bh:
            df.iloc[i, df.columns.get_loc('strat_type')] = 2
            df.iloc[i, df.columns.get_loc('strat_dir')] = 'up'
        elif bl:
            df.iloc[i, df.columns.get_loc('strat_type')] = 2
            df.iloc[i, df.columns.get_loc('strat_dir')] = 'down'
    return df


def generate_signals(df, rsi_ob=70, rsi_os=30, adx_thresh=25):
    """
    Generate signals with parameterizable thresholds (for optimization).
    Returns df with 'signal' column: +1 (long), -1 (short), 0 (flat).
    """
    df = df.copy()
    df['signal'] = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        regime = row.get('regime', 'Chop')
        rsi = row.get('rsi', 50)
        adx = row.get('adx', 0)
        adx_rising = row.get('adx_rising', False)
        macd_hist = row.get('macd_hist', 0)
        macd_cu = row.get('macd_cross_up', False)
        macd_cd = row.get('macd_cross_down', False)
        mom = row.get('momentum', 0)
        strat = row.get('strat_type', 0)
        sdir = row.get('strat_dir', 'neutral')

        score = 0

        # Regime
        if regime in ['Bull_Quiet', 'Bull_Volatile']:
            score += 30
            if rsi_os < rsi < rsi_ob: score += 10
            elif rsi >= rsi_ob: score -= 10
            if adx > adx_thresh and adx_rising: score += 15
            elif adx > adx_thresh: score += 8
            if macd_cu: score += 12
            elif macd_hist > 0: score += 5
            elif macd_cd: score -= 15
            if mom > 0: score += 5
        elif regime in ['Bear_Quiet', 'Bear_Volatile']:
            score -= 30
            if rsi_os < rsi < 50: score -= 10  # RSI 30-50 confirms bear momentum (score goes more negative)
            elif rsi <= rsi_os: score += 10     # Oversold — potential bounce, reduce bear conviction
            if adx > adx_thresh and adx_rising: score -= 15
            elif adx > adx_thresh: score -= 8
            if macd_cd: score -= 12
            elif macd_hist < 0: score -= 5
            elif macd_cu: score += 15
            if mom < 0: score -= 5
        else:
            # Chop: mean reversion only at extremes
            if rsi > rsi_ob: score -= 8
            elif rsi < rsi_os: score += 8
            if adx < 20: score = int(score * 0.5)

        # Strat boost
        if strat == 2 and sdir == 'up' and regime in ['Bull_Quiet', 'Bull_Volatile']:
            score += 10
        elif strat == 2 and sdir == 'down' and regime in ['Bear_Quiet', 'Bear_Volatile']:
            score -= 10

        # Threshold
        if score >= 30:
            df.iloc[i, df.columns.get_loc('signal')] = 1
        elif score <= -30:
            df.iloc[i, df.columns.get_loc('signal')] = -1

    return df


# ─────────────────────────────────────────────
# TRADE SIMULATOR
# ─────────────────────────────────────────────

def simulate_trades(df, mode='intraday', hold_bars_intra=5, hold_bars_swing=20,
                    sl_atr=2.0, tp_atr=3.0):
    """
    Simulate trades on a dataframe with signals.
    Applies transaction costs (0.15% per trade each way).

    mode: 'intraday' or 'swing'
    """
    hold = hold_bars_intra if mode == 'intraday' else hold_bars_swing
    df = df.copy().reset_index(drop=True)
    pos = None
    trades = []
    cooldown = 0

    for i in range(len(df)):
        row = df.iloc[i]

        # Manage open position
        if pos is not None:
            held = i - pos['bar']
            hit_sl = hit_tp = False
            ep = row['Close']

            if pos['dir'] == 1:  # Long
                if row['Low'] <= pos['sl']: hit_sl = True; ep = pos['sl']
                elif row['High'] >= pos['tp']: hit_tp = True; ep = pos['tp']
            else:  # Short
                if row['High'] >= pos['sl']: hit_sl = True; ep = pos['sl']
                elif row['Low'] <= pos['tp']: hit_tp = True; ep = pos['tp']

            if hit_sl or hit_tp or held >= hold:
                if not hit_sl and not hit_tp: ep = row['Close']

                # PnL with transaction costs
                if pos['dir'] == 1:
                    raw_return = (ep - pos['entry']) / pos['entry']
                else:
                    raw_return = (pos['entry'] - ep) / pos['entry']

                net_return = raw_return - ROUND_TRIP  # Subtract 0.30% round trip
                pnl_pct = net_return * 100
                pnl_dollar = net_return * pos['capital_at_risk']

                trades.append({
                    'entry_bar': pos['bar'],
                    'exit_bar': i,
                    'direction': 'LONG' if pos['dir'] == 1 else 'SHORT',
                    'entry_price': pos['entry'],
                    'exit_price': round(ep, 4),
                    'raw_return_pct': round(raw_return * 100, 3),
                    'costs_pct': round(ROUND_TRIP * 100, 2),
                    'net_return_pct': round(pnl_pct, 3),
                    'pnl_dollar': round(pnl_dollar, 2),
                    'bars_held': held,
                    'exit_reason': 'SL' if hit_sl else ('TP' if hit_tp else 'TIME'),
                    'regime': pos.get('regime', ''),
                })
                pos = None
                cooldown = i + (3 if mode == 'intraday' else 5)
                continue

        if i < cooldown: continue

        # Enter on signal
        sig = row.get('signal', 0)
        if sig == 0: continue

        entry = row['Close']
        atr = row.get('atr', entry * 0.01)
        if pd.isna(atr) or atr == 0: atr = entry * 0.01

        if sig == 1:
            sl = entry - atr * sl_atr
            tp = entry + atr * tp_atr
        else:
            sl = entry + atr * sl_atr
            tp = entry - atr * tp_atr

        # Risk 1% of hypothetical capital
        capital_at_risk = 10000 * 0.01  # $100 per trade

        pos = {
            'bar': i, 'entry': entry, 'sl': round(sl, 4), 'tp': round(tp, 4),
            'dir': sig, 'capital_at_risk': capital_at_risk,
            'regime': row.get('regime', ''),
        }

    # Close open position at end
    if pos:
        ep = df.iloc[-1]['Close']
        if pos['dir'] == 1:
            raw_return = (ep - pos['entry']) / pos['entry']
        else:
            raw_return = (pos['entry'] - ep) / pos['entry']
        net_return = raw_return - ROUND_TRIP
        trades.append({
            'entry_bar': pos['bar'], 'exit_bar': len(df)-1,
            'direction': 'LONG' if pos['dir'] == 1 else 'SHORT',
            'entry_price': pos['entry'], 'exit_price': round(ep, 4),
            'raw_return_pct': round(raw_return * 100, 3),
            'costs_pct': round(ROUND_TRIP * 100, 2),
            'net_return_pct': round(net_return * 100, 3),
            'pnl_dollar': round(net_return * pos['capital_at_risk'], 2),
            'bars_held': len(df)-1-pos['bar'],
            'exit_reason': 'END', 'regime': pos.get('regime', ''),
        })

    return trades


# ─────────────────────────────────────────────
# PARAMETER OPTIMIZATION (on training window)
# ─────────────────────────────────────────────

def optimize_on_train(df_train, mode='intraday'):
    """
    Optimize RSI/ADX thresholds on training data for best Sharpe.
    Returns best parameters.
    """
    best_sharpe = -999
    best_params = {'rsi_ob': 70, 'rsi_os': 30, 'adx_thresh': 25}

    # Parameter grid (kept small for speed)
    rsi_ob_range = [65, 70, 75]
    rsi_os_range = [25, 30, 35]
    adx_range = [20, 25, 30]

    for rob in rsi_ob_range:
        for ros in rsi_os_range:
            for adx_t in adx_range:
                df_sig = generate_signals(df_train, rsi_ob=rob, rsi_os=ros, adx_thresh=adx_t)
                trades = simulate_trades(df_sig, mode=mode)

                if len(trades) < 5:
                    continue

                pnls = np.array([t['net_return_pct'] for t in trades])
                if pnls.std() > 0:
                    sharpe = (pnls.mean() / pnls.std()) * np.sqrt(252)
                else:
                    sharpe = 0

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {'rsi_ob': rob, 'rsi_os': ros, 'adx_thresh': adx_t}

    best_params['train_sharpe'] = round(best_sharpe, 3)
    return best_params


# ─────────────────────────────────────────────
# WALK-FORWARD ENGINE
# ─────────────────────────────────────────────

def run_walkforward(ticker, mode='intraday', progress_callback=None):
    """
    Run walk-forward analysis:
      1. Train on 1-year window → optimize parameters
      2. Test blindly on next window (3mo intraday / 18mo swing)
      3. Slide forward and repeat
      4. Stitch blind test results only

    Returns comprehensive results dict.
    """
    # Fetch enough data
    total_years = 5 if mode == 'swing' else 3
    from lib.data_client import get_daily
    df = get_daily(ticker, days=total_years * 365)
    if len(df) < TRAIN_DAYS + 100:
        return {'error': f'Not enough data for {ticker}. Need {total_years}+ years.'}

    # Prepare full dataset
    df = add_indicators(df)
    df = assign_regime(df)
    df = add_strat(df)
    df = df.dropna().reset_index(drop=True)

    test_size = TEST_INTRADAY if mode == 'intraday' else TEST_SWING
    step_size = STEP_INTRADAY if mode == 'intraday' else STEP_SWING

    # Walk-forward windows
    windows = []
    i = 0
    while i + TRAIN_DAYS + test_size <= len(df):
        train_start = i
        train_end = i + TRAIN_DAYS
        test_start = train_end
        test_end = min(train_end + test_size, len(df))

        windows.append({
            'train': (train_start, train_end),
            'test': (test_start, test_end),
        })
        i += step_size

    if not windows:
        return {'error': 'Not enough data for even one walk-forward window.'}

    # Run each window
    all_blind_trades = []
    window_results = []

    for wi, w in enumerate(windows):
        ts, te = w['train']
        os, oe = w['test']

        df_train = df.iloc[ts:te].copy()
        df_test = df.iloc[os:oe].copy()

        if progress_callback:
            progress_callback(wi + 1, len(windows),
                f"Window {wi+1}/{len(windows)}: Train [{ts}-{te}] → Test [{os}-{oe}]")

        # 1. Optimize on train
        best_params = optimize_on_train(df_train, mode=mode)

        # 2. Generate signals on TEST with train-optimized params
        df_test_sig = generate_signals(
            df_test,
            rsi_ob=best_params['rsi_ob'],
            rsi_os=best_params['rsi_os'],
            adx_thresh=best_params['adx_thresh'],
        )

        # 3. Simulate trades on blind test data
        blind_trades = simulate_trades(df_test_sig, mode=mode)

        # Tag trades with window info
        for t in blind_trades:
            t['window'] = wi + 1
            t['params'] = best_params.copy()

        all_blind_trades.extend(blind_trades)

        # Window summary
        if blind_trades:
            pnls = np.array([t['net_return_pct'] for t in blind_trades])
            w_sharpe = (pnls.mean() / pnls.std()) * np.sqrt(252) if pnls.std() > 0 else 0
            w_wr = (pnls > 0).sum() / len(pnls) * 100
        else:
            w_sharpe = 0; w_wr = 0

        window_results.append({
            'window': wi + 1,
            'train_range': f"{ts}-{te}",
            'test_range': f"{os}-{oe}",
            'params': best_params,
            'train_sharpe': best_params['train_sharpe'],
            'test_trades': len(blind_trades),
            'test_sharpe': round(w_sharpe, 3),
            'test_win_rate': round(w_wr, 1),
        })

    # ─── STITCH BLIND RESULTS ───
    return compute_walkforward_metrics(all_blind_trades, window_results, ticker, mode, df)


def compute_walkforward_metrics(trades, windows, ticker, mode, df):
    """Compute final metrics from stitched blind test results."""
    if not trades:
        return {
            'error': 'No trades generated across all walk-forward windows.',
            'ticker': ticker, 'mode': mode, 'windows': windows,
        }

    tdf = pd.DataFrame(trades)
    pnls = tdf['net_return_pct'].values
    raw_pnls = tdf['raw_return_pct'].values
    w = pnls[pnls > 0]
    l = pnls[pnls < 0]

    # Core metrics
    win_rate = len(w) / len(pnls) * 100
    avg_win = w.mean() if len(w) > 0 else 0
    avg_loss = l.mean() if len(l) > 0 else 0
    pf = abs(w.sum() / l.sum()) if len(l) > 0 and l.sum() != 0 else 99.9
    sharpe = (pnls.mean() / pnls.std()) * np.sqrt(252) if pnls.std() > 0 else 0

    # Cumulative returns
    cum_net = (1 + pnls / 100).cumprod()
    cum_raw = (1 + raw_pnls / 100).cumprod()
    total_return_net = (cum_net[-1] - 1) * 100
    total_return_raw = (cum_raw[-1] - 1) * 100
    total_cost_drag = total_return_raw - total_return_net

    # Max drawdown on cumulative curve
    running_max = np.maximum.accumulate(cum_net)
    dd = (cum_net - running_max) / running_max * 100
    max_dd = dd.min()

    # Buy & hold comparison
    if len(df) > 0:
        bh_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    else:
        bh_return = 0
    alpha = total_return_net - bh_return

    # Regime breakdown
    regime_stats = tdf.groupby('regime').agg(
        count=('net_return_pct', 'count'),
        wr=('net_return_pct', lambda x: (x > 0).mean() * 100),
        avg_net=('net_return_pct', 'mean'),
        total=('net_return_pct', 'sum'),
    ).round(2)

    # Walk-forward efficiency (test sharpe / train sharpe)
    wf_efficiencies = []
    for wr in windows:
        if wr['train_sharpe'] > 0 and wr['test_sharpe'] != 0:
            wf_efficiencies.append(wr['test_sharpe'] / wr['train_sharpe'])
    wf_efficiency = np.mean(wf_efficiencies) * 100 if wf_efficiencies else 0

    return {
        'ticker': ticker,
        'mode': mode,
        'total_trades': len(pnls),
        'winners': len(w),
        'losers': len(l),
        'win_rate': round(win_rate, 1),
        'avg_win_pct': round(avg_win, 3),
        'avg_loss_pct': round(avg_loss, 3),
        'profit_factor': round(pf, 2),
        'sharpe': round(sharpe, 2),
        'total_return_net': round(total_return_net, 2),
        'total_return_raw': round(total_return_raw, 2),
        'total_cost_drag': round(total_cost_drag, 2),
        'max_drawdown': round(max_dd, 2),
        'alpha_vs_bh': round(alpha, 2),
        'buy_hold_return': round(bh_return, 2),
        'avg_bars_held': round(tdf['bars_held'].mean(), 1),
        'wf_efficiency': round(wf_efficiency, 1),
        'windows': windows,
        'regime_stats': regime_stats,
        'trades_df': tdf,
        'cumulative_curve': cum_net.tolist(),
        'costs': {
            'exchange_fee': f"{EXCHANGE_FEE*100:.2f}%",
            'slippage': f"{SLIPPAGE*100:.2f}%",
            'total_per_trade': f"{TOTAL_COST*100:.2f}%",
            'round_trip': f"{ROUND_TRIP*100:.2f}%",
        },
    }
