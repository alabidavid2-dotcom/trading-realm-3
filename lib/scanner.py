# ================================================
# SCANNER.PY - S&P 500 Two-Tier Scanner
# Tier 1: Fast regime filter on all 500 (~60 sec)
# Tier 2: Full HMM + indicators + Strat on candidates (~3-4 min)
# ================================================

import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from lib.trade_grader import grade_ticker_full
    GRADER_AVAILABLE = True
except ImportError:
    GRADER_AVAILABLE = False

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SCAN_RESULTS_FILE = "scan_history.json"


def get_sp500_tickers():
    """Fetch S&P 500 ticker list. Falls back to top ~110 if Wikipedia fetch fails."""
    try:
        tables = pd.read_html(SP500_URL)
        df = tables[0]
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers
    except Exception:
        return [
            'AAPL','MSFT','AMZN','NVDA','GOOGL','GOOG','META','BRK-B','TSLA','UNH',
            'XOM','JNJ','JPM','V','PG','MA','HD','CVX','MRK','ABBV',
            'LLY','PEP','COST','KO','AVGO','WMT','MCD','CSCO','TMO','ACN',
            'ABT','DHR','LIN','NEE','CRM','AMD','ADBE','ORCL','TXN','PM',
            'UPS','NKE','MS','RTX','LOW','INTC','QCOM','HON','INTU','SPGI',
            'IBM','GE','CAT','BA','AMAT','GS','BLK','AXP','AMGN','SYK',
            'DE','MDLZ','BKNG','ADI','ISRG','GILD','CB','VRTX','LRCX','MMC',
            'REGN','SCHW','PLD','ZTS','MU','PYPL','PANW','SNPS','CDNS','KLAC',
            'CME','CI','SO','DUK','CL','EOG','SLB','BDX','FI','ICE',
            'CMG','MCO','MO','PNC','TGT','USB','APD','GM','F','ELF',
            'SPY','QQQ','IWM','DIA','XLF','XLK','XLE','XLV','XLI','ARKK',
        ]


def fetch_quick(ticker, days=100):
    try:
        from lib.data_client import get_daily
        df = get_daily(ticker, days=days)
        if len(df) < 30:
            return None
        return df
    except Exception:
        return None


def tier1_quick_screen(ticker):
    """
    Tier 1: Fast screen using simple metrics (~0.2 sec per ticker).
    Returns score dict or None if uninteresting.
    """
    df = fetch_quick(ticker, days=100)
    if df is None:
        return None

    try:
        df['ret'] = df['Close'].pct_change()
        df['vol20'] = df['ret'].rolling(20).std() * np.sqrt(252)

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0).ewm(span=14, adjust=False).mean()
        loss = (-delta).where(delta < 0, 0.0).ewm(span=14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)

        h, lo, c = df['High'], df['Low'], df['Close']
        pdm = h.diff(); mdm = -lo.diff()
        pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
        mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
        tr = pd.concat([h-lo, (h-c.shift(1)).abs(), (lo-c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()
        pdi = 100*(pdm.ewm(span=14, adjust=False).mean()/atr.replace(0, np.nan))
        mdi = 100*(mdm.ewm(span=14, adjust=False).mean()/atr.replace(0, np.nan))
        dx = (abs(pdi-mdi)/(pdi+mdi).replace(0, np.nan))*100
        df['adx'] = dx.ewm(span=14, adjust=False).mean()

        latest = df.iloc[-1]
        ret_20 = df['ret'].tail(20).mean()
        vol = float(latest.get('vol20', 0.5))
        rsi = float(latest.get('rsi', 50))
        adx = float(latest.get('adx', 0))
        close = float(latest['Close'])

        if ret_20 > 0.001 and vol < 0.25: regime_guess = "Bull_Quiet"
        elif ret_20 > 0.001: regime_guess = "Bull_Volatile"
        elif ret_20 < -0.001 and vol < 0.25: regime_guess = "Bear_Quiet"
        elif ret_20 < -0.001: regime_guess = "Bear_Volatile"
        else: regime_guess = "Chop"

        score = 0
        if regime_guess in ["Bull_Quiet", "Bull_Volatile"]:
            score += 30
            if 50 < rsi < 70: score += 15
            if adx > 25: score += 10
        elif regime_guess in ["Bear_Quiet", "Bear_Volatile"]:
            score -= 30
            if 30 < rsi < 50: score -= 15
            if adx > 25: score -= 10

        if abs(score) < 20 or regime_guess == "Chop":
            return None

        return {
            'ticker': ticker, 'close': round(close, 2),
            'regime_guess': regime_guess, 'quick_score': score,
            'rsi': round(rsi, 1), 'adx': round(adx, 1),
            'vol20': round(vol * 100, 1),
        }
    except Exception:
        return None


def tier2_full_analysis(ticker, train_days=730):
    """Tier 2: Full HMM + indicators + Strat. Returns complete analysis or None."""
    try:
        from lib.data_client import get_daily
        df = get_daily(ticker, days=train_days)
        if len(df) < 60:
            return None

        df['daily_return'] = df['Close'].pct_change()
        df['vol_20'] = df['daily_return'].rolling(20).std() * np.sqrt(252)
        df['range_pct'] = (df['High'] - df['Low']) / df['Close']
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        else:
            df['volume_ratio'] = 1.0
        df = df.dropna()

        features = ['daily_return', 'vol_20', 'volume_ratio', 'range_pct']
        X = df[features].values
        regime_probs = None
        regime = "Chop"
        confidence = 0

        try:
            from hmmlearn.hmm import GaussianHMM
            model = GaussianHMM(n_components=5, covariance_type='full', n_iter=200, random_state=42)
            model.fit(X)
            states = model.predict(X)
            df['regime_id'] = states

            labels = {0:"Bear_Volatile",1:"Bear_Quiet",2:"Chop",3:"Bull_Quiet",4:"Bull_Volatile"}
            means = df.groupby('regime_id')['daily_return'].mean().sort_values()
            sorted_ids = means.index.tolist()
            all_lk = sorted(labels.keys())
            n_states = len(sorted_ids)
            n_labels = len(all_lk)
            id_map = {}
            for i, rid in enumerate(sorted_ids):
                if n_states <= n_labels:
                    id_map[rid] = labels[all_lk[i]]
                else:
                    idx = round(i * (n_labels - 1) / (n_states - 1)) if n_states > 1 else 0
                    id_map[rid] = labels[all_lk[idx]]
            df['regime'] = df['regime_id'].map(id_map).fillna('Chop')

            posteriors = model.predict_proba(X)
            latest_probs = posteriors[-1]
            regime_probs = {}
            for i, rid in enumerate(sorted_ids):
                regime_probs[id_map[rid]] = round(float(latest_probs[rid]) * 100, 1)

            regime = df.iloc[-1]['regime']
            confidence = max(regime_probs.values()) if regime_probs else 0
        except ImportError:
            ret_q = df['daily_return'].quantile([0.15, 0.85])
            vol_q = df['vol_20'].quantile(0.5)
            def assign(row):
                r, v = row['daily_return'], row['vol_20']
                if r > ret_q[0.85] and v < vol_q: return "Bull_Quiet"
                elif r > ret_q[0.85]: return "Bull_Volatile"
                elif r < ret_q[0.15] and v < vol_q: return "Bear_Quiet"
                elif r < ret_q[0.15]: return "Bear_Volatile"
                else: return "Chop"
            df['regime'] = df.apply(assign, axis=1)
            regime = df.iloc[-1]['regime']
            confidence = 75

        # Indicators
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0).ewm(span=14, adjust=False).mean()
        loss = (-delta).where(delta < 0, 0.0).ewm(span=14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)

        h, lo, c = df['High'], df['Low'], df['Close']
        pdm = h.diff(); mdm = -lo.diff()
        pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
        mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
        tr = pd.concat([h-lo,(h-c.shift(1)).abs(),(lo-c.shift(1)).abs()], axis=1).max(axis=1)
        atr_s = tr.ewm(span=14, adjust=False).mean()
        pdi = 100*(pdm.ewm(span=14, adjust=False).mean()/atr_s.replace(0, np.nan))
        mdi = 100*(mdm.ewm(span=14, adjust=False).mean()/atr_s.replace(0, np.nan))
        dx = (abs(pdi-mdi)/(pdi+mdi).replace(0, np.nan))*100
        df['adx'] = dx.ewm(span=14, adjust=False).mean()
        df['adx_rising'] = df['adx'] > df['adx'].shift(1)

        ef = df['Close'].ewm(span=12, adjust=False).mean()
        es = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ef - es
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_cross_up'] = (df['macd']>df['macd_signal'])&(df['macd'].shift(1)<=df['macd_signal'].shift(1))
        df['macd_cross_down'] = (df['macd']<df['macd_signal'])&(df['macd'].shift(1)>=df['macd_signal'].shift(1))
        df['momentum'] = df['Close'].pct_change(10) * 100

        latest = df.iloc[-1]
        rsi = float(latest.get('rsi', 50))
        adx = float(latest.get('adx', 0))
        adx_rising = bool(latest.get('adx_rising', False))
        macd_hist = float(latest.get('macd_hist', 0))
        macd_cross_up = bool(latest.get('macd_cross_up', False))
        macd_cross_down = bool(latest.get('macd_cross_down', False))
        momentum = float(latest.get('momentum', 0))

        # Signal scoring
        regime_scores = {"Bull_Quiet":40,"Bull_Volatile":25,"Chop":0,"Bear_Quiet":-40,"Bear_Volatile":-25}
        r_score = regime_scores.get(regime, 0)
        i_score = 0
        if regime in ["Bull_Quiet","Bull_Volatile"]:
            if 50<rsi<70: i_score+=15
            elif rsi>=70: i_score-=10
            if adx>25 and adx_rising: i_score+=15
            elif adx>25: i_score+=8
            if macd_cross_up: i_score+=12
            elif macd_hist>0: i_score+=5
            elif macd_cross_down: i_score-=15
            if momentum>0: i_score+=5
        elif regime in ["Bear_Quiet","Bear_Volatile"]:
            if 30<rsi<50: i_score-=15
            elif rsi<=30: i_score+=10
            if adx>25 and adx_rising: i_score-=15
            elif adx>25: i_score-=8
            if macd_cross_down: i_score-=12
            elif macd_hist<0: i_score-=5
            elif macd_cross_up: i_score+=15
            if momentum<0: i_score-=5
        else:
            if adx<20: i_score=int(i_score*0.5)

        # Strat
        df['strat_type']=0; df['strat_dir']='neutral'
        for idx in range(1,len(df)):
            prev,curr = df.iloc[idx-1],df.iloc[idx]
            bh=curr['High']>prev['High']; bl=curr['Low']<prev['Low']
            inside=curr['Low']>=prev['Low'] and curr['High']<=prev['High']
            if inside: df.iloc[idx,df.columns.get_loc('strat_type')]=1
            elif bh and bl:
                df.iloc[idx,df.columns.get_loc('strat_type')]=3
                df.iloc[idx,df.columns.get_loc('strat_dir')]='up' if curr['Close']>curr['Open'] else 'down'
            elif bh:
                df.iloc[idx,df.columns.get_loc('strat_type')]=2
                df.iloc[idx,df.columns.get_loc('strat_dir')]='up'
            elif bl:
                df.iloc[idx,df.columns.get_loc('strat_type')]=2
                df.iloc[idx,df.columns.get_loc('strat_dir')]='down'

        types=df['strat_type'].tail(5).tolist()
        dirs=df['strat_dir'].tail(5).tolist()
        patterns=[]
        if len(types)>=3:
            t3,t2,t1=types[-3],types[-2],types[-1]
            d3,d2,d1=dirs[-3],dirs[-2],dirs[-1]
            if t3==2 and t2==1 and t1==2 and d3!=d1 and d1!='neutral':
                patterns.append({'name':'2-1-2 Rev','direction':d1,'grade':'A+'})
            if t3==2 and t2==1 and t1==2 and d3==d1 and d1!='neutral':
                patterns.append({'name':'2-1-2 Cont','direction':d1,'grade':'A'})
            if t2==3 and t1==2 and d1!='neutral':
                patterns.append({'name':'3-2 Cont','direction':d1,'grade':'A'})
        if len(types)>=2:
            if types[-2]==2 and types[-1]==2 and dirs[-2]!=dirs[-1] and dirs[-1]!='neutral':
                patterns.append({'name':'2-2 Rev','direction':dirs[-1],'grade':'A'})

        s_score=0
        for p in patterns:
            w={'A+':20,'A':15,'B+':8,'B':5}.get(p['grade'],3)
            if p['direction']=='pending': s_score+=w*0.3; continue
            bull_pat=p['direction']=='up'
            if bull_pat and regime in ["Bull_Quiet","Bull_Volatile"]: s_score+=w
            elif not bull_pat and regime in ["Bear_Quiet","Bear_Volatile"]: s_score-=w
            else: s_score+=(1 if bull_pat else -1)*w*0.5

        composite = max(-100, min(100, r_score+i_score+int(s_score)))
        a=abs(composite)
        strength='STRONG' if a>=50 else ('MODERATE' if a>=30 else ('WEAK' if a>=15 else 'NO_TRADE'))
        direction='LONG' if composite>15 else ('SHORT' if composite<-15 else 'FLAT')
        if strength=='NO_TRADE' or direction=='FLAT': trade_type='CASH'
        elif direction=='LONG' and strength=='STRONG': trade_type='0DTE CALL'
        elif direction=='LONG': trade_type='SWING LONG'
        elif direction=='SHORT' and strength=='STRONG': trade_type='0DTE PUT'
        else: trade_type='SWING SHORT'

        # Build indicators dict for grader
        ind_dict = {
            'rsi':round(rsi,1),'adx':round(adx,1),'adx_rising':adx_rising,
            'macd_hist':round(macd_hist,4),
            'macd_cross_up':macd_cross_up,'macd_cross_down':macd_cross_down,
            'momentum':round(momentum,2),
            'high_volume': float(latest.get('volume_ratio',1.0)) > 1.5,
        }

        return {
            'ticker':ticker,'close':round(float(latest['Close']),2),
            'regime':regime,'confidence':round(confidence,1),
            'regime_probs':regime_probs,
            'composite':composite,'direction':direction,
            'strength':strength,'trade_type':trade_type,
            'rsi':round(rsi,1),'adx':round(adx,1),'adx_rising':adx_rising,
            'macd_hist':round(macd_hist,4),'momentum':round(momentum,2),
            'vol20':round(float(latest.get('vol_20',0))*100,1),
            'patterns':patterns,'indicators':ind_dict,
            'r_score':r_score,'i_score':i_score,'s_score':int(s_score),
            'scan_time':datetime.now().strftime("%I:%M %p"),
            '_df': df,  # Pass dataframe for grading
        }
    except Exception:
        return None


def run_full_scan(progress_callback=None, max_workers=10):
    """
    Full two-tier scan. Returns top 10 + all qualified results.
    """
    tickers = get_sp500_tickers()
    total = len(tickers)

    if progress_callback:
        progress_callback("tier1", 0, total, f"Tier 1: Screening {total} tickers...")

    tier1_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(tier1_quick_screen, t): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                tier1_results.append(result)
            if progress_callback and done % 25 == 0:
                progress_callback("tier1", done, total,
                    f"Tier 1: {done}/{total} | {len(tier1_results)} candidates")

    tier1_results.sort(key=lambda x: abs(x['quick_score']), reverse=True)
    candidates = tier1_results[:80]

    if progress_callback:
        progress_callback("tier2", 0, len(candidates),
            f"Tier 2: Deep analysis on {len(candidates)} candidates...")

    tier2_results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(tier2_full_analysis, c['ticker']): c['ticker'] for c in candidates}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                tier2_results.append(result)
            if progress_callback and done % 10 == 0:
                progress_callback("tier2", done, len(candidates),
                    f"Tier 2: {done}/{len(candidates)} | {len(tier2_results)} analyzed")

    qualified = [
        r for r in tier2_results
        if r['confidence'] >= 85 and abs(r['composite']) >= 30
    ]
    qualified.sort(key=lambda x: abs(x['composite']), reverse=True)

    # === TIER 3: Grade qualified results with full trade grader ===
    if GRADER_AVAILABLE and qualified:
        if progress_callback:
            progress_callback("tier3", 0, len(qualified),
                f"Tier 3: Grading {len(qualified)} setups (FTC + sector + ATR)...")
        for i, r in enumerate(qualified):
            try:
                grade = grade_ticker_full(
                    ticker=r['ticker'],
                    regime=r['regime'],
                    regime_confidence=r['confidence'],
                    composite_score=r['composite'],
                    direction=r['direction'],
                    indicators=r.get('indicators', {}),
                    strat_patterns=r.get('patterns', []),
                    df=r.get('_df', None),
                )
                r['grade'] = grade
            except Exception:
                r['grade'] = None
            if progress_callback and (i+1) % 5 == 0:
                progress_callback("tier3", i+1, len(qualified),
                    f"Tier 3: {i+1}/{len(qualified)} graded")

    # Remove internal dataframe before returning (not serializable)
    for r in tier2_results:
        r.pop('_df', None)
    for r in qualified:
        r.pop('_df', None)

    # Add gap badge to each qualified result
    try:
        from lib.indicators import detect_gap
        from lib.data_client import get_daily
        for r in qualified:
            try:
                df_gap = get_daily(r['ticker'], days=5)
                if not df_gap.empty:
                    r['gap'] = detect_gap(df_gap)
                else:
                    r['gap'] = {'gap_type': 'none', 'gap_pct': 0.0}
            except Exception:
                r['gap'] = {'gap_type': 'none', 'gap_pct': 0.0}
    except Exception:
        pass

    # Sort: A+ first, then A, then by composite
    grade_order = {'A+': 0, 'A': 1, 'B': 2, 'C': 3, 'NO_TRADE': 4, None: 5}
    def sort_key(r):
        g = r.get('grade') or {}
        gi = g.get('grade_intraday', 'NO_TRADE') if isinstance(g, dict) else 'NO_TRADE'
        gs = g.get('grade_swing',    'NO_TRADE') if isinstance(g, dict) else 'NO_TRADE'
        best = min(grade_order.get(gi, 5), grade_order.get(gs, 5))
        return (best, -abs(r['composite']))
    qualified.sort(key=sort_key)

    sector_breaks = detect_simultaneous_sector_breaks(qualified)

    return {
        'top10':          qualified[:10],
        'all_qualified':  qualified,
        'tier1_count':    len(tier1_results),
        'tier2_count':    len(tier2_results),
        'total_scanned':  total,
        'scan_time':      datetime.now().strftime("%I:%M %p"),
        'sector_breaks':  sector_breaks,
    }


def detect_simultaneous_sector_breaks(results: list) -> list:
    """
    Detect when 3+ tickers in the same sector break the same direction simultaneously.
    Returns list of sector break event dicts.
    """
    try:
        from lib.trade_grader import STOCK_TO_SECTOR
    except ImportError:
        return []

    sector_counts = {}
    for r in results:
        ticker    = r.get('ticker', '')
        direction = r.get('direction', 'FLAT')
        if direction == 'FLAT':
            continue
        sector = STOCK_TO_SECTOR.get(ticker)
        if not sector:
            continue
        key = (sector, direction)
        sector_counts.setdefault(key, []).append(ticker)

    breaks = []
    for (sector, direction), tickers in sector_counts.items():
        if len(tickers) >= 3:
            breaks.append({
                'sector':    sector,
                'direction': direction,
                'tickers':   tickers,
                'count':     len(tickers),
            })
    breaks.sort(key=lambda x: x['count'], reverse=True)
    return breaks


def load_scan_history():
    if os.path.exists(SCAN_RESULTS_FILE):
        try:
            with open(SCAN_RESULTS_FILE, 'r') as f:
                data = json.load(f)
            if data.get('date') == datetime.now().strftime("%Y-%m-%d"):
                return data.get('scans', [])
        except Exception:
            pass
    return []


def save_scan_history(scans):
    data = {'date': datetime.now().strftime("%Y-%m-%d"), 'scans': scans}
    try:
        with open(SCAN_RESULTS_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass


def merge_scan_results(new_results, history):
    current_time = datetime.now().strftime("%I:%M %p")
    current_tickers = set()
    new_entries = []
    for r in new_results:
        current_tickers.add(r['ticker'])
        entry = {k: v for k, v in r.items() if k not in ('regime_probs', '_df')}

        # Clean grade dict for JSON serialization
        if 'grade' in entry and entry['grade']:
            g = entry['grade']
            # Remove non-serializable items
            g.pop('session', None)
            # Simplify nested dicts
            if 'sector_corr' in g:
                g['sector_corr'] = {k: v for k, v in g['sector_corr'].items() if isinstance(v, (str, int, float, bool, type(None)))}
            if 'ftc' in g:
                g['ftc'] = {k: v for k, v in g['ftc'].items() if isinstance(v, (str, int, float, bool, type(None), dict))}
                g['ftc'].pop('details', None)  # Remove nested directions dict

        entry['status'] = 'current'
        entry['last_seen'] = current_time
        entry['first_seen'] = current_time
        for h in history:
            if h.get('ticker') == r['ticker']:
                entry['first_seen'] = h.get('first_seen', current_time)
                break
        new_entries.append(entry)

    old_entries = []
    for h in history:
        if h.get('ticker') not in current_tickers:
            h['status'] = 'previous'
            old_entries.append(h)

    return new_entries + old_entries
