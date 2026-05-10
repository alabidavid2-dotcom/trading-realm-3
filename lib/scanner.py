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
import warnings
warnings.filterwarnings('ignore')

try:
    from lib.trade_grader import grade_ticker_full
    GRADER_AVAILABLE = True
except ImportError:
    GRADER_AVAILABLE = False

from config import supabase

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


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


# ── Sector Sentinel ──────────────────────────────────────────────────────────
# Maps each universe ticker to its primary sector ETF.
# SMH used for semiconductor names (more precise than XLK).
SECTOR_MAP: dict = {
    # Technology / Semis
    'AAPL': 'XLK', 'MSFT': 'XLK', 'ORCL': 'XLK', 'CRM': 'XLK', 'ADBE': 'XLK',
    'CSCO': 'XLK', 'IBM': 'XLK', 'ACN': 'XLK', 'INTU': 'XLK', 'PANW': 'XLK',
    'FTNT': 'XLK', 'SNPS': 'SMH', 'CDNS': 'SMH', 'KLAC': 'SMH', 'AMAT': 'SMH',
    'LRCX': 'SMH', 'MCHP': 'SMH', 'ON': 'SMH', 'TER': 'SMH', 'ENTG': 'SMH',
    'NVDA': 'SMH', 'AMD': 'SMH', 'MU': 'SMH', 'AVGO': 'SMH', 'TXN': 'SMH',
    'QCOM': 'SMH', 'ADI': 'SMH', 'INTC': 'SMH',
    # Healthcare
    'JNJ': 'XLV', 'LLY': 'XLV', 'UNH': 'XLV', 'MRK': 'XLV', 'ABBV': 'XLV',
    'ABT': 'XLV', 'TMO': 'XLV', 'DHR': 'XLV', 'SYK': 'XLV', 'MDT': 'XLV',
    'ISRG': 'XLV', 'GILD': 'XLV', 'AMGN': 'XLV', 'VRTX': 'XLV', 'BDX': 'XLV',
    'ZTS': 'XLV', 'CI': 'XLV', 'REGN': 'XLV', 'HCA': 'XLV', 'HUM': 'XLV',
    'CVS': 'XLV', 'BMY': 'XLV', 'EW': 'XLV', 'MCK': 'XLV', 'ABC': 'XLV',
    # Financials
    'JPM': 'XLF', 'BAC': 'XLF', 'WFC': 'XLF', 'GS': 'XLF', 'MS': 'XLF',
    'AXP': 'XLF', 'BLK': 'XLF', 'SCHW': 'XLF', 'CB': 'XLF', 'ICE': 'XLF',
    'CME': 'XLF', 'SPGI': 'XLF', 'MCO': 'XLF', 'PNC': 'XLF', 'USB': 'XLF',
    'TFC': 'XLF', 'COF': 'XLF', 'AIG': 'XLF', 'MMC': 'XLF', 'BK': 'XLF',
    'FI': 'XLF', 'V': 'XLF', 'MA': 'XLF', 'PYPL': 'XLF',
    # Energy
    'XOM': 'XLE', 'CVX': 'XLE', 'SLB': 'XLE', 'EOG': 'XLE', 'MPC': 'XLE',
    'PSX': 'XLE', 'VLO': 'XLE', 'COP': 'XLE', 'OXY': 'XLE', 'HES': 'XLE',
    'DVN': 'XLE', 'BKR': 'XLE', 'HAL': 'XLE', 'APA': 'XLE',
    # Industrials
    'UPS': 'XLI', 'RTX': 'XLI', 'HON': 'XLI', 'BA': 'XLI', 'CAT': 'XLI',
    'DE': 'XLI', 'GE': 'XLI', 'LMT': 'XLI', 'EMR': 'XLI', 'ETN': 'XLI',
    'PH': 'XLI', 'FDX': 'XLI', 'WM': 'XLI', 'CARR': 'XLI', 'OTIS': 'XLI',
    'GWW': 'XLI', 'LHX': 'XLI', 'CMI': 'XLI', 'IR': 'XLI', 'TT': 'XLI',
    'AME': 'XLI',
    # Consumer Staples
    'PG': 'XLP', 'KO': 'XLP', 'PEP': 'XLP', 'WMT': 'XLP', 'COST': 'XLP',
    'MDLZ': 'XLP', 'CL': 'XLP', 'GIS': 'XLP', 'HSY': 'XLP', 'MO': 'XLP',
    'PM': 'XLP', 'STZ': 'XLP', 'ELF': 'XLP', 'MNST': 'XLP',
    # Consumer Discretionary
    'AMZN': 'XLY', 'TSLA': 'XLY', 'HD': 'XLY', 'MCD': 'XLY', 'NKE': 'XLY',
    'LOW': 'XLY', 'SBUX': 'XLY', 'TJX': 'XLY', 'BKNG': 'XLY', 'MAR': 'XLY',
    'GM': 'XLY', 'F': 'XLY', 'HLT': 'XLY', 'TGT': 'XLY', 'DHI': 'XLY',
    'LVS': 'XLY', 'MGM': 'XLY', 'PHM': 'XLY', 'CMG': 'XLY', 'APTV': 'XLY',
    # Communication Services
    'META': 'XLC', 'GOOGL': 'XLC', 'GOOG': 'XLC', 'T': 'XLC', 'VZ': 'XLC',
    'DIS': 'XLC', 'NFLX': 'XLC', 'CMCSA': 'XLC', 'EA': 'XLC', 'TTWO': 'XLC',
    'WBD': 'XLC', 'LYV': 'XLC',
    # Utilities
    'NEE': 'XLU', 'DUK': 'XLU', 'SO': 'XLU', 'D': 'XLU', 'SRE': 'XLU',
    'EXC': 'XLU', 'AEP': 'XLU', 'XEL': 'XLU', 'ED': 'XLU', 'WEC': 'XLU',
    # Real Estate
    'PLD': 'XLRE', 'AMT': 'XLRE', 'CCI': 'XLRE', 'EQIX': 'XLRE', 'PSA': 'XLRE',
    'SPG': 'XLRE', 'DLR': 'XLRE', 'O': 'XLRE',
    # Materials
    'LIN': 'XLB', 'APD': 'XLB', 'SHW': 'XLB', 'FCX': 'XLB', 'NEM': 'XLB',
    'ECL': 'XLB', 'DD': 'XLB', 'PPG': 'XLB', 'ALB': 'XLB', 'CF': 'XLB',
}


def _sector_direction(ftfc_list: list, min_aligned: int = 3) -> tuple:
    """Return (direction_str, aligned_count, total_tfs) for a sector ETF's FTFC stack."""
    up   = sum(1 for tf in ftfc_list if tf.get('direction') == 'up')
    down = sum(1 for tf in ftfc_list if tf.get('direction') == 'down')
    total = up + down
    if up >= min_aligned and up > down:
        return 'up', up, total
    if down >= min_aligned and down > up:
        return 'down', down, total
    return 'neutral', max(up, down), total


def _classify_gap(price, prev_close, prev_high, prev_low):
    """
    Classify gap type by comparing current price against yesterday's range.
    Returns (gap_type_str, gap_pct_float).

    Full Up   : price > prev_high  — gapped entirely above yesterday's range
    Partial Up: price > prev_close — opened above close but inside yesterday's range
    Full Down : price < prev_low   — gapped entirely below yesterday's range
    Partial Down: price < prev_close — opened below close but inside yesterday's range
    No Gap    : within prev_close tolerance
    """
    if None in (price, prev_close, prev_high, prev_low):
        return 'No Gap', 0.0
    gap_pct = round((price - prev_close) / prev_close * 100, 2)
    if price > prev_high:
        return 'Full Up', gap_pct
    if price > prev_close:
        return 'Partial Up', gap_pct
    if price < prev_low:
        return 'Full Down', gap_pct
    if price < prev_close:
        return 'Partial Down', gap_pct
    return 'No Gap', 0.0


def run_watchlist_scan(tickers: list = None, timeout_secs: float = 7.5, progress_callback=None) -> dict:
    """
    Fast scan targeting < 8 seconds for ≤ 25 tickers.

    Pipeline:
    1. Batch snapshot — one API call for all prices (< 1s)
    2. Parallel FTFC  — ThreadPoolExecutor(max_workers=10)
    3. PTR scoring    — ranked by FTFC alignment count
    4. Hard timeout   — 7.5s wall-clock; returns partial results with warning
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, wait as cf_wait
    from lib.data_client import get_batch_snapshots, get_ftfc_snapshot
    from config import ALL_TICKERS

    if tickers is None:
        tickers = ALL_TICKERS

    scan_start = time.monotonic()
    truncated  = False

    # ── Step 1: Batch snapshot (one API call) ─────────────────────────────────
    if progress_callback:
        progress_callback("snapshot", 0, len(tickers), f"Batch snapshot: {len(tickers)} tickers...")
    snapshots    = get_batch_snapshots(tickers)
    snap_elapsed = time.monotonic() - scan_start
    if progress_callback:
        progress_callback("snapshot", len(tickers), len(tickers),
            f"Snapshots done ({snap_elapsed:.1f}s) — {len(snapshots)}/{len(tickers)} returned")

    # ── Step 2: Parallel FTFC (tickers + sector ETFs, one pool) ──────────────
    ftfc_budget  = max(1.0, timeout_secs - snap_elapsed - 0.3)
    ftfc_results: dict = {}

    # Collect unique sector ETFs for this universe — no extra time cost (same pool)
    sector_etfs = set()
    for t in tickers:
        etf = SECTOR_MAP.get(t.upper())
        if etf and etf != t.upper():
            sector_etfs.add(etf)
    all_fetch_targets = list(tickers) + [e for e in sector_etfs if e not in tickers]

    if progress_callback:
        progress_callback("ftfc", 0, len(tickers),
            f"FTFC in parallel ({len(tickers)} tickers + {len(sector_etfs)} sector ETFs, {ftfc_budget:.1f}s budget)...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {executor.submit(get_ftfc_snapshot, t, 'intraday'): t for t in all_fetch_targets}
        done_futs, pending_futs = cf_wait(list(future_map), timeout=ftfc_budget)
        for fut in done_futs:
            t = future_map[fut]
            try:
                ftfc_results[t] = fut.result()
            except Exception:
                ftfc_results[t] = []
        if pending_futs:
            truncated = True
            for fut in pending_futs:
                ftfc_results[future_map[fut]] = []

    # ── Step 3: PTR scoring ───────────────────────────────────────────────────
    results = []
    for ticker in tickers:
        snap  = snapshots.get(ticker.upper(), {})
        ftfc  = ftfc_results.get(ticker, [])

        aligned_up   = sum(1 for tf in ftfc if tf.get('direction') == 'up')
        aligned_down = sum(1 for tf in ftfc if tf.get('direction') == 'down')
        valid_tfs    = aligned_up + aligned_down

        if aligned_up > aligned_down:
            ftfc_dir      = 'up'
            aligned_count = aligned_up
        elif aligned_down > aligned_up:
            ftfc_dir      = 'down'
            aligned_count = aligned_down
        else:
            ftfc_dir      = 'neutral'
            aligned_count = 0

        ptr_score = round((aligned_count / valid_tfs * 100) if valid_tfs else 0)
        snap_dir  = snap.get('direction', 'neutral')
        dir_match = (snap_dir == ftfc_dir) and ftfc_dir != 'neutral'
        composite = ptr_score + (20 if dir_match else 0)

        # Gap detection
        gap_type, gap_pct = _classify_gap(
            snap.get('price'), snap.get('prev_close'),
            snap.get('prev_high'), snap.get('prev_low'),
        )
        # High-conviction bonus: full gap aligned with FTFC direction
        if gap_type == 'Full Up'   and ftfc_dir == 'up':
            composite += 30
        elif gap_type == 'Full Down' and ftfc_dir == 'down':
            composite += 30

        # ── Sector Sentinel ──────────────────────────────────────────────────
        sector_etf      = SECTOR_MAP.get(ticker.upper(), '')
        sentinel_bonus  = 0
        sector_divergence = False
        sector_dir      = 'neutral'

        if sector_etf:
            sec_ftfc = ftfc_results.get(sector_etf, [])
            sector_dir, _sec_aligned, _sec_total = _sector_direction(sec_ftfc, min_aligned=3)

            if ftfc_dir != 'neutral' and sector_dir == ftfc_dir:
                sentinel_bonus = 15
                composite     += 15
            elif ftfc_dir != 'neutral' and sector_dir != 'neutral' and sector_dir != ftfc_dir:
                sector_divergence = True

        # Alpha Setup: Full Gap + FTFC aligned + Sector confirmed
        alpha_setup = (
            gap_type in ('Full Up', 'Full Down')
            and sentinel_bonus > 0
        )

        results.append({
            'ticker':             ticker,
            'price':              snap.get('price'),
            'change_pct':         snap.get('change_pct'),
            'snap_dir':           snap_dir,
            'direction':          'LONG' if ftfc_dir == 'up' else ('SHORT' if ftfc_dir == 'down' else 'FLAT'),
            'ftfc_stack':         ftfc,
            'aligned_up':         aligned_up,
            'aligned_down':       aligned_down,
            'total_tfs':          valid_tfs,
            'ptr_score':          ptr_score,
            'gap_type':           gap_type,
            'gap_pct':            gap_pct,
            'composite':          composite,
            'sector_etf':         sector_etf,
            'sector_dir':         sector_dir,
            'sentinel_bonus':     sentinel_bonus,
            'sector_divergence':  sector_divergence,
            'alpha_setup':        alpha_setup,
            'scan_time':          datetime.now().strftime("%I:%M %p"),
        })

    results.sort(key=lambda x: (x['ptr_score'], x['composite']), reverse=True)
    elapsed = round(time.monotonic() - scan_start, 2)

    return {
        'results':       results,
        'top10':         results[:10],
        'total_scanned': len(tickers),
        'completed':     len([r for r in results if r['total_tfs'] > 0]),
        'elapsed_secs':  elapsed,
        'truncated':     truncated,
        'warning':       'Optimization: Scan truncated to maintain 8s performance.' if truncated else None,
        'scan_time':     datetime.now().strftime("%I:%M %p"),
    }


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
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        res = supabase.table('scan_history').select('raw_data').eq('scan_date', today).execute()
        return [row['raw_data'] for row in (res.data or [])]
    except Exception:
        return []


def save_scan_history(scans):
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        supabase.table('scan_history').delete().eq('scan_date', today).execute()
        if scans:
            records = []
            for scan in scans:
                safe = json.loads(json.dumps(scan, default=str))
                records.append({
                    'scan_date': today,
                    'ticker':    scan.get('ticker', 'UNKNOWN'),
                    'raw_data':  safe,
                })
            supabase.table('scan_history').insert(records).execute()
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


def generate_sparkline_base64(ticker: str, days: int = 14) -> str:
    """Fetch recent closes and return a base64-encoded PNG sparkline. Empty string on failure."""
    try:
        import base64
        import io
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from lib.data_client import get_daily

        df = get_daily(ticker, days=days + 5)
        if df is None or len(df) < 5:
            return ''
        closes = df['Close'].tail(days).values
        if len(closes) < 2:
            return ''

        up = closes[-1] >= closes[0]
        line_color = '#4ade80' if up else '#f87171'
        fill_color = '#14532d' if up else '#7f1d1d'

        fig, ax = plt.subplots(figsize=(3.2, 1.2))
        fig.patch.set_facecolor('#0a0f1a')
        ax.set_facecolor('#0a0f1a')

        xs = list(range(len(closes)))
        ax.plot(xs, closes, color=line_color, linewidth=2.0, zorder=3)
        ax.fill_between(xs, closes, closes.min() * 0.999, alpha=0.25, color=fill_color, zorder=2)
        ax.set_xlim(0, len(closes) - 1)
        ax.margins(y=0.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=72, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('ascii')
    except Exception:
        return ''
