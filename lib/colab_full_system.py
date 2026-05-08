# ================================================
# THE STRAT + HMM REGIME TRADING SYSTEM
# Complete Google Colab Version — Single File
# ================================================
# HOW TO USE:
#   1. Open Google Colab (colab.research.google.com)
#   2. Paste this entire file into a single cell, OR
#      split at the "# === CELL X ===" markers into separate cells
#   3. Run all cells
#   4. Modify CONFIG section to customize
# ================================================

# === CELL 1: INSTALL & IMPORTS ===
# !pip install yfinance hmmlearn pandas numpy matplotlib -q

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
    print("✅ hmmlearn loaded — using real HMM regime detection")
except ImportError:
    HMM_AVAILABLE = False
    print("⚠️  hmmlearn not found. Using quantile-based fallback.")
    print("   Install with: pip install hmmlearn")

print("✅ All libraries ready!")


# === CELL 2: CONFIGURATION ===
# ──────────────────────────────────────────────────────
# CHANGE THESE TO CUSTOMIZE YOUR SYSTEM
# ──────────────────────────────────────────────────────

TICKERS = ['SPY', 'QQQ', 'IWM']
WATCHLIST = ['NVDA', 'AAPL', 'TSLA', 'MU', 'WMT', 'UNH', 'ELF', 'GM']

# HMM
HMM_N_REGIMES = 5
HMM_TRAIN_DAYS = 700
HMM_N_ITER = 200
HMM_SEED = 42

# Indicators
RSI_PERIOD = 14
RSI_OB = 70        # Overbought
RSI_OS = 30        # Oversold
ADX_PERIOD = 14
ADX_TREND = 25
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG = 9
MOM_PERIOD = 10

# Risk
RISK_0DTE = 75
RISK_SWING = 150
CONTRACTS_MIN = 2
CONTRACTS_MAX = 4

# Backtest
INITIAL_CAPITAL = 10000
COMMISSION = 1.50

REGIME_LABELS = {0: "Bear_Volatile", 1: "Bear_Quiet", 2: "Chop", 3: "Bull_Quiet", 4: "Bull_Volatile"}
REGIME_RISK_MULT = {"Bull_Quiet": 1.0, "Bull_Volatile": 0.75, "Bear_Quiet": 1.0, "Bear_Volatile": 0.5, "Chop": 0.4}

TODAY = datetime.now().strftime("%Y-%m-%d")
print(f"📅 System configured for {TODAY}")


# === CELL 3: DATA FETCHER ===

def fetch(ticker, days=HMM_TRAIN_DAYS, interval='1d'):
    end = datetime.today()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# === CELL 4: HMM REGIME DETECTION ===

def prepare_hmm_features(df):
    df = df.copy()
    df['daily_return'] = df['Close'].pct_change()
    df['vol_20'] = df['daily_return'].rolling(20).std() * np.sqrt(252)
    df['range_pct'] = (df['High'] - df['Low']) / df['Close']
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    else:
        df['volume_ratio'] = 1.0
    return df.dropna()


def train_hmm_model(df):
    df = prepare_hmm_features(df)
    features = ['daily_return', 'vol_20', 'volume_ratio', 'range_pct']
    X = df[features].values

    if not HMM_AVAILABLE:
        return None, df, features

    model = GaussianHMM(
        n_components=HMM_N_REGIMES,
        covariance_type='full',
        n_iter=HMM_N_ITER,
        random_state=HMM_SEED,
        verbose=False
    )
    model.fit(X)
    return model, df, features


def label_regimes(model, df, features):
    X = df[features].values

    if model is None:
        # Quantile fallback
        ret_q = df['daily_return'].quantile([0.15, 0.85])
        vol_q = df['vol_20'].quantile(0.5)
        def assign(row):
            r, v = row['daily_return'], row['vol_20']
            if r > ret_q[0.85] and v < vol_q: return "Bull_Quiet"
            elif r > ret_q[0.85]: return "Bull_Volatile"
            elif r < ret_q[0.15] and v < vol_q: return "Bear_Quiet"
            elif r < ret_q[0.15]: return "Bear_Volatile"
            else: return "Chop"
        df = df.copy()
        df['regime'] = df.apply(assign, axis=1)
        df['regime_id'] = df['regime'].factorize()[0]
        return df

    states = model.predict(X)
    df = df.copy()
    df['regime_id'] = states

    # Sort by mean return → label
    means = df.groupby('regime_id')['daily_return'].mean().sort_values()
    sorted_ids = means.index.tolist()
    lk = sorted(REGIME_LABELS.keys())[:len(sorted_ids)]
    id_map = {rid: REGIME_LABELS[lk[i]] for i, rid in enumerate(sorted_ids) if i < len(lk)}
    df['regime'] = df['regime_id'].map(id_map).fillna('Chop')
    return df


def get_regime_probs(model, df, features):
    if model is None or not HMM_AVAILABLE:
        return None
    X = df[features].values
    probs = model.predict_proba(X)[-1]
    states = model.predict(X)
    df_tmp = df.copy()
    df_tmp['regime_id'] = states
    means = df_tmp.groupby('regime_id')['daily_return'].mean().sort_values()
    sorted_ids = means.index.tolist()
    lk = sorted(REGIME_LABELS.keys())[:len(sorted_ids)]
    result = {}
    for i, rid in enumerate(sorted_ids):
        if i < len(lk):
            result[REGIME_LABELS[lk[i]]] = round(probs[rid] * 100, 1)
    return result


def full_regime_analysis(ticker):
    df = fetch(ticker)
    if df.empty:
        return None
    model, df, feats = train_hmm_model(df)
    df = label_regimes(model, df, feats)
    probs = get_regime_probs(model, df, feats)
    latest = df.iloc[-1]
    return {
        'ticker': ticker,
        'close': round(float(latest['Close']), 2),
        'regime': latest['regime'],
        'vol_20': round(float(latest['vol_20']) * 100, 2),
        'probs': probs,
        'model': model,
        'df': df,
        'features': feats,
        'hmm_active': model is not None,
    }


# === CELL 5: TECHNICAL INDICATORS ===

def add_rsi(df, p=RSI_PERIOD):
    d = df['Close'].diff()
    g = d.where(d > 0, 0.0)
    l = (-d).where(d < 0, 0.0)
    ag = g.ewm(span=p, adjust=False).mean()
    al = l.ewm(span=p, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
    return df

def add_adx(df, p=ADX_PERIOD):
    h, lo, c = df['High'], df['Low'], df['Close']
    pdm = h.diff(); mdm = -lo.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=p, adjust=False).mean()
    pdi = 100 * (pdm.ewm(span=p, adjust=False).mean() / atr.replace(0, np.nan))
    mdi = 100 * (mdm.ewm(span=p, adjust=False).mean() / atr.replace(0, np.nan))
    dx = (abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)) * 100
    df['adx'] = dx.ewm(span=p, adjust=False).mean()
    df['adx_rising'] = df['adx'] > df['adx'].shift(1)
    df['plus_di'] = pdi; df['minus_di'] = mdi
    return df

def add_macd(df):
    ef = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
    es = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd'] = ef - es
    df['macd_signal'] = df['macd'].ewm(span=MACD_SIG, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    return df

def add_momentum(df):
    df['momentum'] = df['Close'].pct_change(MOM_PERIOD) * 100
    return df

def add_bollinger(df, p=20, s=2.0):
    df['bb_mid'] = df['Close'].rolling(p).mean()
    std = df['Close'].rolling(p).std()
    df['bb_upper'] = df['bb_mid'] + std * s
    df['bb_lower'] = df['bb_mid'] - std * s
    df['bb_pct'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    df['bb_squeeze'] = std < std.rolling(50).mean()
    return df

def add_atr(df, p=14):
    h, lo, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(p).mean()
    df['atr_pct'] = df['atr'] / df['Close'] * 100
    return df

def add_volume_analysis(df, p=20):
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['vol_ma'] = df['Volume'].rolling(p).mean()
        df['vol_ratio_ind'] = df['Volume'] / df['vol_ma'].replace(0, np.nan)
        df['high_volume'] = df['vol_ratio_ind'] > 1.5
    else:
        df['vol_ratio_ind'] = 1.0; df['high_volume'] = False
    return df

def add_all_indicators(df):
    df = df.copy()
    for fn in [add_rsi, add_adx, add_macd, add_momentum, add_bollinger, add_atr, add_volume_analysis]:
        df = fn(df)
    return df

def indicator_snapshot(df):
    r = df.iloc[-1]
    return {
        'rsi': round(float(r.get('rsi', 50)), 1),
        'adx': round(float(r.get('adx', 0)), 1),
        'adx_rising': bool(r.get('adx_rising', False)),
        'macd_hist': round(float(r.get('macd_hist', 0)), 4),
        'macd_cross_up': bool(r.get('macd_cross_up', False)),
        'macd_cross_down': bool(r.get('macd_cross_down', False)),
        'momentum': round(float(r.get('momentum', 0)), 2),
        'bb_pct': round(float(r.get('bb_pct', 0.5)), 2),
        'bb_squeeze': bool(r.get('bb_squeeze', False)),
        'vol_ratio': round(float(r.get('vol_ratio_ind', 1.0)), 2),
        'high_volume': bool(r.get('high_volume', False)),
        'atr': round(float(r.get('atr', 0)), 2),
    }


# === CELL 6: THE STRAT CLASSIFIER ===

def classify_candle(prev, curr):
    breaks_high = curr['High'] > prev['High']
    breaks_low = curr['Low'] < prev['Low']
    inside = curr['Low'] >= prev['Low'] and curr['High'] <= prev['High']
    if inside:
        return 1, 'neutral'
    elif breaks_high and breaks_low:
        return 3, ('up' if curr['Close'] > curr['Open'] else 'down')
    elif breaks_high:
        return 2, 'up'
    elif breaks_low:
        return 2, 'down'
    return 1, 'neutral'

def add_strat(df):
    df = df.copy()
    df['strat_type'] = 0; df['strat_dir'] = 'neutral'
    for i in range(1, len(df)):
        t, d = classify_candle(df.iloc[i-1], df.iloc[i])
        df.iloc[i, df.columns.get_loc('strat_type')] = t
        df.iloc[i, df.columns.get_loc('strat_dir')] = d
    return df

def detect_patterns(df, lookback=5):
    if len(df) < lookback: return []
    recent = df.tail(lookback)
    types = recent['strat_type'].tolist()
    dirs = recent['strat_dir'].tolist()
    patterns = []

    if len(types) >= 3:
        t3, t2, t1 = types[-3], types[-2], types[-1]
        d3, d2, d1 = dirs[-3], dirs[-2], dirs[-1]
        if t3 == 2 and t2 == 1 and t1 == 2 and d3 != d1 and d1 != 'neutral':
            patterns.append({'name': '2-1-2 Reversal', 'direction': d1, 'grade': 'A+'})
        if t3 == 2 and t2 == 1 and t1 == 2 and d3 == d1 and d1 != 'neutral':
            patterns.append({'name': '2-1-2 Continuation', 'direction': d1, 'grade': 'A'})
        if t2 == 3 and t1 == 2 and d1 != 'neutral':
            patterns.append({'name': '3-2 Continuation', 'direction': d1, 'grade': 'A'})
        if t2 == 3 and t1 == 1:
            patterns.append({'name': '3-1 Compression', 'direction': 'pending', 'grade': 'B+'})
    if len(types) >= 2:
        if types[-2] == 2 and types[-1] == 2 and dirs[-2] != dirs[-1] and dirs[-1] != 'neutral':
            patterns.append({'name': '2-2 Reversal', 'direction': dirs[-1], 'grade': 'A'})
    if sum(1 for t in types[-3:] if t == 1) >= 2:
        patterns.append({'name': 'Inside Compression', 'direction': 'pending', 'grade': 'B'})
    return patterns


# === CELL 7: SIGNAL ENGINE (REGIME-GATED) ===

def score_regime(regime):
    return {"Bull_Quiet": 40, "Bull_Volatile": 25, "Chop": 0, "Bear_Quiet": -40, "Bear_Volatile": -25}.get(regime, 0)

def score_indicators(snap, regime):
    s = 0
    rsi, adx, adxr = snap['rsi'], snap['adx'], snap['adx_rising']
    mh, mcu, mcd = snap['macd_hist'], snap['macd_cross_up'], snap['macd_cross_down']
    mom, bbp, vr = snap['momentum'], snap['bb_pct'], snap['vol_ratio']

    if regime in ["Bull_Quiet", "Bull_Volatile"]:
        if 50 < rsi < RSI_OB: s += 15
        elif rsi >= RSI_OB: s -= 10
        elif rsi < 40: s -= 5
        if adx > ADX_TREND and adxr: s += 15
        elif adx > ADX_TREND: s += 8
        if mcu: s += 12
        elif mh > 0: s += 5
        elif mcd: s -= 15
        if mom > 0: s += 5
        if vr > 1.5: s += 5
    elif regime in ["Bear_Quiet", "Bear_Volatile"]:
        if RSI_OS < rsi < 50: s -= 15
        elif rsi <= RSI_OS: s += 10
        elif rsi > 60: s += 5
        if adx > ADX_TREND and adxr: s -= 15
        elif adx > ADX_TREND: s -= 8
        if mcd: s -= 12
        elif mh < 0: s -= 5
        elif mcu: s += 15
        if mom < 0: s -= 5
        if vr > 1.5: s -= 5
    else:  # Chop
        if rsi > RSI_OB and bbp > 0.95: s -= 10
        elif rsi < RSI_OS and bbp < 0.05: s += 10
        if adx < 20: s = int(s * 0.5)
    return round(s)

def score_strat(patterns, regime):
    s = 0
    for p in patterns:
        w = {'A+': 20, 'A': 15, 'B+': 8, 'B': 5}.get(p['grade'], 3)
        if p['direction'] == 'pending':
            s += w * 0.3; continue
        bull_pat = p['direction'] == 'up'
        if bull_pat and regime in ["Bull_Quiet", "Bull_Volatile"]: s += w
        elif not bull_pat and regime in ["Bear_Quiet", "Bear_Volatile"]: s -= w
        elif bull_pat and regime in ["Bear_Quiet", "Bear_Volatile"]: s += w * 0.3
        elif not bull_pat and regime in ["Bull_Quiet", "Bull_Volatile"]: s -= w * 0.3
        else: s += (1 if bull_pat else -1) * w * 0.5
    return round(s)

def generate_signal(regime, snap, patterns, probs=None):
    rs = score_regime(regime)
    ins = score_indicators(snap, regime)
    ss = score_strat(patterns, regime)
    comp = max(-100, min(100, rs + ins + ss))
    confidence = 50
    reasons = [f"Regime({regime}): {rs:+d}", f"Indicators: {ins:+d}", f"Strat: {ss:+d}"]

    if probs:
        top = max(probs.values())
        confidence = min(95, top)
        if top < 50:
            comp = int(comp * 0.7)
            reasons.append(f"Low regime confidence ({top}%) — dampened")

    a = abs(comp)
    strength = 'STRONG' if a >= 50 else ('MODERATE' if a >= 30 else ('WEAK' if a >= 15 else 'NO_TRADE'))
    direction = 'LONG' if comp > 15 else ('SHORT' if comp < -15 else 'FLAT')
    if strength == 'NO_TRADE' or direction == 'FLAT': tt = 'NO_TRADE'
    elif direction == 'LONG' and strength == 'STRONG': tt = '0DTE_CALL'
    elif direction == 'LONG': tt = 'SWING_LONG'
    elif direction == 'SHORT' and strength == 'STRONG': tt = '0DTE_PUT'
    else: tt = 'SWING_SHORT'

    return {
        'composite': comp, 'direction': direction, 'strength': strength,
        'trade_type': tt, 'confidence': confidence,
        'risk_mult': REGIME_RISK_MULT.get(regime, 0.5),
        'reasons': reasons, 'r_score': rs, 'i_score': ins, 's_score': ss,
    }

def is_grade_a(sig):
    return sig['strength'] in ['STRONG', 'MODERATE'] and sig['confidence'] >= 40


# === CELL 8: BACKTESTER ===

def run_backtest(regime_df, ticker='SPY', capital=INITIAL_CAPITAL,
                 hold_bars=5, sl_atr=2.0, tp_atr=3.0, min_bars=50):
    df = regime_df.copy()
    df = add_all_indicators(df)
    df = add_strat(df)
    df = df.dropna().reset_index(drop=True)

    cap = capital
    pos = None
    trades = []
    equity = []
    cooldown_until = 0

    for i in range(min_bars, len(df)):
        row = df.iloc[i]
        regime = row.get('regime', 'Chop')
        equity.append({'idx': i, 'equity': cap, 'regime': regime})

        # Manage open position
        if pos is not None:
            held = i - pos['bar']
            hit_sl = hit_tp = False
            if pos['dir'] == 'LONG':
                if row['Low'] <= pos['sl']: hit_sl = True; ep = pos['sl']
                elif row['High'] >= pos['tp']: hit_tp = True; ep = pos['tp']
            else:
                if row['High'] >= pos['sl']: hit_sl = True; ep = pos['sl']
                elif row['Low'] <= pos['tp']: hit_tp = True; ep = pos['tp']

            if hit_sl or hit_tp or held >= hold_bars:
                if not hit_sl and not hit_tp: ep = row['Close']
                pnl = ((ep - pos['entry']) if pos['dir'] == 'LONG' else (pos['entry'] - ep)) * pos['shares']
                pnl -= COMMISSION * 2
                trades.append({
                    'entry': pos['entry'], 'exit': round(ep, 2), 'dir': pos['dir'],
                    'pnl': round(pnl, 2), 'regime': pos['regime'], 'score': pos['score'],
                    'exit_reason': 'SL' if hit_sl else ('TP' if hit_tp else 'TIME'),
                    'bars': held, 'shares': pos['shares'],
                })
                cap += pnl; pos = None; cooldown_until = i + 3
                continue

        if i < cooldown_until: continue

        # Generate signal
        window = df.iloc[max(0, i-50):i+1]
        try:
            snap = indicator_snapshot(window)
        except: continue
        pats = detect_patterns(df.iloc[max(0, i-10):i+1])
        sig = generate_signal(regime, snap, pats)

        if not is_grade_a(sig): continue

        # Enter
        entry = row['Close']
        atr = row.get('atr', entry * 0.01)
        if sig['direction'] == 'LONG':
            sl = entry - atr * sl_atr; tp = entry + atr * tp_atr
        else:
            sl = entry + atr * sl_atr; tp = entry - atr * tp_atr

        risk_d = abs(entry - sl)
        if risk_d == 0: continue
        shares = max(1, int(cap * 0.01 * sig['risk_mult'] / risk_d))

        pos = {'bar': i, 'entry': entry, 'sl': round(sl, 2), 'tp': round(tp, 2),
               'dir': sig['direction'], 'shares': shares, 'regime': regime, 'score': sig['composite']}

    # Close open position
    if pos:
        ep = df.iloc[-1]['Close']
        pnl = ((ep - pos['entry']) if pos['dir'] == 'LONG' else (pos['entry'] - ep)) * pos['shares']
        pnl -= COMMISSION * 2
        trades.append({'entry': pos['entry'], 'exit': round(ep, 2), 'dir': pos['dir'],
                       'pnl': round(pnl, 2), 'regime': pos['regime'], 'score': pos['score'],
                       'exit_reason': 'END', 'bars': len(df)-1-pos['bar'], 'shares': pos['shares']})
        cap += pnl

    return compute_metrics(trades, equity, cap, capital, ticker)


def compute_metrics(trades, equity, final_cap, init_cap, ticker):
    if not trades:
        return {'error': 'No trades generated', 'ticker': ticker}

    tdf = pd.DataFrame(trades)
    pnls = tdf['pnl'].values
    w = pnls[pnls > 0]; l = pnls[pnls < 0]

    eq = pd.DataFrame(equity)
    dd = 0
    if len(eq) > 0:
        e = eq['equity']
        dd = ((e - e.cummax()) / e.cummax() * 100).min()

    sharpe = (pnls.mean() / pnls.std()) * np.sqrt(252) if pnls.std() > 0 else 0
    pf = abs(w.sum() / l.sum()) if len(l) > 0 and l.sum() != 0 else float('inf')

    regime_stats = tdf.groupby('regime').agg(
        count=('pnl', 'count'), wr=('pnl', lambda x: (x > 0).mean() * 100),
        avg_pnl=('pnl', 'mean'), total=('pnl', 'sum')).round(2)

    return {
        'ticker': ticker, 'trades': len(pnls), 'winners': len(w), 'losers': len(l),
        'win_rate': round(len(w)/len(pnls)*100, 1), 'avg_win': round(w.mean(), 2) if len(w) else 0,
        'avg_loss': round(l.mean(), 2) if len(l) else 0, 'profit_factor': round(pf, 2),
        'sharpe': round(sharpe, 2), 'return_pct': round((final_cap - init_cap) / init_cap * 100, 2),
        'return_dollar': round(final_cap - init_cap, 2), 'final_capital': round(final_cap, 2),
        'max_dd': round(dd, 2), 'avg_bars': round(tdf['bars'].mean(), 1),
        'regime_stats': regime_stats, 'trades_df': tdf, 'equity_df': pd.DataFrame(equity),
    }


# === CELL 9: VISUALIZATION ===

def plot_regime_chart(regime_data, title=None):
    df = regime_data['df'].copy()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

    # Price + regime background
    ax1 = axes[0]
    colors = {"Bull_Quiet": "#2ecc71", "Bull_Volatile": "#27ae60", "Chop": "#95a5a6",
              "Bear_Quiet": "#e74c3c", "Bear_Volatile": "#c0392b"}
    for regime, color in colors.items():
        mask = df['regime'] == regime
        ax1.fill_between(df.index, df['Close'].min() * 0.98, df['Close'].max() * 1.02,
                         where=mask, alpha=0.15, color=color, label=regime)
    ax1.plot(df.index, df['Close'], color='black', linewidth=1)
    ax1.set_title(title or f"{regime_data['ticker']} — HMM Regime Map")
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_ylabel('Price')

    # Volatility
    ax2 = axes[1]
    ax2.plot(df.index, df['vol_20'] * 100, color='purple', linewidth=0.8)
    ax2.set_ylabel('Vol 20d (%)')
    ax2.axhline(y=df['vol_20'].median() * 100, color='gray', linestyle='--', alpha=0.5)

    # Daily returns
    ax3 = axes[2]
    pos_ret = df['daily_return'].where(df['daily_return'] > 0)
    neg_ret = df['daily_return'].where(df['daily_return'] <= 0)
    ax3.bar(df.index, pos_ret * 100, color='green', alpha=0.6, width=1)
    ax3.bar(df.index, neg_ret * 100, color='red', alpha=0.6, width=1)
    ax3.set_ylabel('Return (%)')

    plt.tight_layout()
    plt.show()


def plot_backtest(metrics):
    if 'error' in metrics:
        print(metrics['error']); return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Equity curve
    eq = metrics['equity_df']
    axes[0, 0].plot(eq['idx'], eq['equity'], color='blue', linewidth=1)
    axes[0, 0].set_title('Equity Curve')
    axes[0, 0].set_ylabel('$')

    # Trade PnL
    tdf = metrics['trades_df']
    colors = ['green' if p > 0 else 'red' for p in tdf['pnl']]
    axes[0, 1].bar(range(len(tdf)), tdf['pnl'], color=colors, alpha=0.7)
    axes[0, 1].set_title('Trade PnL')
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)

    # Win rate by regime
    rs = metrics['regime_stats']
    if 'wr' in rs.columns:
        rs['wr'].plot(kind='barh', ax=axes[1, 0], color='steelblue')
        axes[1, 0].set_title('Win Rate by Regime')
        axes[1, 0].set_xlabel('%')

    # Cumulative PnL
    axes[1, 1].plot(tdf['pnl'].cumsum(), color='darkblue', linewidth=1.5)
    axes[1, 1].set_title('Cumulative PnL')
    axes[1, 1].axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.show()


# === CELL 10: DAILY PLAYBOOK (RUN THIS EVERY MORNING) ===

def run_playbook(run_bt=False):
    print(f"🚀 Generating Playbook for {TODAY}\n")

    results = {}
    # Core indices
    for t in TICKERS:
        print(f"  Analyzing {t}...")
        rd = full_regime_analysis(t)
        if not rd: continue
        df_ind = add_all_indicators(rd['df'])
        snap = indicator_snapshot(df_ind)
        df_strat = add_strat(df_ind)
        pats = detect_patterns(df_strat)
        sig = generate_signal(rd['regime'], snap, pats, rd['probs'])

        bt = None
        if run_bt:
            bt = run_backtest(rd['df'], ticker=t)

        results[t] = {'regime': rd, 'snap': snap, 'patterns': pats, 'signal': sig, 'bt': bt}

    # Watchlist
    for t in WATCHLIST:
        print(f"  Scanning {t}...")
        rd = full_regime_analysis(t)
        if not rd: continue
        df_ind = add_all_indicators(rd['df'])
        snap = indicator_snapshot(df_ind)
        df_strat = add_strat(df_ind)
        pats = detect_patterns(df_strat)
        sig = generate_signal(rd['regime'], snap, pats, rd['probs'])
        results[t] = {'regime': rd, 'snap': snap, 'patterns': pats, 'signal': sig, 'bt': None}

    # Print report
    spy = results.get('SPY')
    if not spy:
        print("❌ No SPY data"); return results

    regime = spy['regime']['regime']
    vol = spy['regime']['vol_20']
    probs = spy['regime'].get('probs', {})
    sig = spy['signal']
    rm = REGIME_RISK_MULT.get(regime, 0.5)
    risk = round(RISK_0DTE * rm, 0)

    print(f"\n{'='*65}")
    print(f"  📅 THE STRAT + HMM REGIME PLAYBOOK — {TODAY}")
    print(f"{'='*65}")
    print(f"\n  🏛️  SPY REGIME: {regime} | Vol: {vol}%")
    if probs:
        print(f"  📊 Regime Probabilities:")
        for r, p in sorted(probs.items(), key=lambda x: -x[1]):
            print(f"      {r:<18} {p:>5.1f}%  {'█' * int(p / 5)}")
    print(f"\n  💰 0DTE Risk: ${risk} | Swing Risk: ${round(RISK_SWING * rm)} | Mult: {rm}x")

    print(f"\n{'─'*65}")
    print(f"  CORE INDICES")
    print(f"{'─'*65}")
    for t in TICKERS:
        r = results.get(t)
        if not r: continue
        s = r['signal']
        icon = '🟢' if s['direction'] == 'LONG' else ('🔴' if s['direction'] == 'SHORT' else '⚪')
        grade = '✅' if is_grade_a(s) else '⬜'
        print(f"\n  {icon} {t} @ ${r['regime']['close']}")
        print(f"     Regime: {r['regime']['regime']} | Signal: {s['direction']} {s['strength']} ({s['composite']:+d}) {grade}")
        print(f"     RSI: {r['snap']['rsi']} | ADX: {r['snap']['adx']} | MACD: {r['snap']['macd_hist']}")
        for p in r['patterns']:
            print(f"     🎯 {p['name']} [{p['grade']}] → {p['direction']}")
        for reason in s['reasons']:
            print(f"        → {reason}")

    print(f"\n{'─'*65}")
    print(f"  WATCHLIST — Grade A/A+ Only")
    print(f"{'─'*65}")
    found = False
    for t in WATCHLIST:
        r = results.get(t)
        if not r: continue
        s = r['signal']
        if is_grade_a(s):
            found = True
            icon = '🟢' if s['direction'] == 'LONG' else '🔴'
            print(f"\n  {icon} {t} @ ${r['regime']['close']} | {s['direction']} {s['strength']} ({s['composite']:+d})")
            print(f"     Regime: {r['regime']['regime']} | RSI: {r['snap']['rsi']}")
            for p in r['patterns']:
                print(f"     🎯 {p['name']} [{p['grade']}] → {p['direction']}")
    if not found:
        print("\n  No Grade A+/A watchlist setups. Patience. 🧘")

    # Backtest
    if spy.get('bt') and 'error' not in spy['bt']:
        b = spy['bt']
        print(f"\n{'─'*65}")
        print(f"  BACKTEST (SPY)")
        print(f"{'─'*65}")
        print(f"  Trades: {b['trades']} | Win Rate: {b['win_rate']}% | PF: {b['profit_factor']}")
        print(f"  Sharpe: {b['sharpe']} | Return: {b['return_pct']}% | Max DD: {b['max_dd']}%")

    print(f"\n{'='*65}")
    print(f"  Only A+/A setups. Respect the regime. No revenge trades.")
    print(f"{'='*65}\n")

    return results


# === CELL 11: RUN IT ===
# Basic playbook (fast):
# results = run_playbook(run_bt=False)

# Full playbook with backtest (slower):
# results = run_playbook(run_bt=True)

# Visualize SPY regime map:
# plot_regime_chart(results['SPY']['regime'])

# Visualize backtest results (must run_bt=True first):
# plot_backtest(results['SPY']['bt'])

# === QUICK BACKTEST ON ANY TICKER ===
# rd = full_regime_analysis('QQQ')
# bt = run_backtest(rd['df'], ticker='QQQ')
# print(f"Win: {bt['win_rate']}% | Sharpe: {bt['sharpe']} | Return: {bt['return_pct']}%")
# plot_backtest(bt)

print("\n✅ System loaded! Uncomment the cells above to run.")
print("   Quick start: results = run_playbook()")
