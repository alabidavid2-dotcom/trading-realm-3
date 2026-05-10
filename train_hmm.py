# ================================================
# TRAIN_HMM.PY  —  SPY Regime Model Training
# Run once (or weekly) to retrain the market brain.
#
# Usage:  python train_hmm.py
# Output: models/hmm_model.pkl
# ================================================

import os
import sys
import pickle
import numpy as np
from datetime import datetime


# ── Dependency guard ──────────────────────────────────────────────────────────
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    sys.exit("hmmlearn not found.  Run:  pip install hmmlearn")

# ── Config ────────────────────────────────────────────────────────────────────
TICKER        = 'SPY'
LOOKBACK_DAYS = 730          # ~2 years of trading data
N_COMPONENTS  = 5            # hidden states
COVARIANCE    = 'full'       # full covariance matrix (best for financial data)
N_ITER        = 500          # more iterations → tighter convergence
RANDOM_SEED   = 42
MODEL_PATH    = os.path.join(os.path.dirname(__file__), 'models', 'hmm_model.pkl')
FEATURES      = ['log_return', 'range']


# ── Step 1: Fetch data ────────────────────────────────────────────────────────
print(f"[1/4]  Fetching {LOOKBACK_DAYS}d of {TICKER} daily bars from Alpaca...")

from lib.data_client import get_daily
df = get_daily(TICKER, days=LOOKBACK_DAYS)

if df is None or len(df) < 100:
    sys.exit(f"Not enough data returned ({len(df) if df is not None else 0} rows). Check Alpaca keys.")

print(f"       {len(df)} daily bars loaded  ({df.index[0].date()} to {df.index[-1].date()})")


# ── Step 2: Feature engineering ───────────────────────────────────────────────
print("[2/4]  Computing features  (log_return, range)...")

import pandas as pd

df = df.copy()
# Log return: ln(close_t / close_{t-1})  — stationary, normally distributed
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
# Intraday range normalised by close  — proxy for volatility
df['range']      = (df['High'] - df['Low']) / df['Close']

df = df.dropna()
X = df[FEATURES].values
print(f"       Feature matrix: {X.shape[0]} obs x {X.shape[1]} features")


# ── Step 3: Train GaussianHMM ─────────────────────────────────────────────────
print(f"[3/4]  Training GaussianHMM  ({N_COMPONENTS} states, {N_ITER} iterations)...")

model = GaussianHMM(
    n_components=N_COMPONENTS,
    covariance_type=COVARIANCE,
    n_iter=N_ITER,
    random_state=RANDOM_SEED,
    verbose=False,
)
model.fit(X)

if not model.monitor_.converged:
    print("       ⚠  Model did not fully converge — consider increasing N_ITER")
else:
    print(f"       Converged after {model.monitor_.iter} iterations")


# ── Step 4: Label states ──────────────────────────────────────────────────────
print("[4/4]  Labelling regime states...")

# Per-state statistics from trained model
state_means = model.means_[:, 0]                               # mean log_return per state
state_vars  = np.array([model.covars_[i][0, 0]                 # variance of log_return
                         for i in range(N_COMPONENTS)])

var_median = np.median(state_vars)

# The state with the smallest |mean_return| is the sideways / Chop regime.
# The remaining four are classified by sign (positive/negative) and
# variance (below/above median) → 2 × 2 = 4 labels.
chop_idx = int(np.argmin(np.abs(state_means)))

state_labels: dict[int, str] = {}
for i in range(N_COMPONENTS):
    if i == chop_idx:
        state_labels[i] = 'Chop'
    elif state_means[i] > 0:
        state_labels[i] = 'Bull Quiet' if state_vars[i] < var_median else 'Bull Volatile'
    else:
        state_labels[i] = 'Bear Quiet' if state_vars[i] < var_median else 'Bear Volatile'

# Pretty-print state diagnostics
print()
print(f"  {'State':>6}  {'Mean Ret':>10}  {'Variance':>10}  {'Label'}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*20}")
for i in range(N_COMPONENTS):
    star = '  << Chop pivot' if i == chop_idx else ''
    print(f"  {i:>6}  {state_means[i]:>+.6f}  {state_vars[i]:>10.8f}  {state_labels[i]}{star}")
print()


# ── Save model artifact ───────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

artifact = {
    'model':             model,
    'state_labels':      state_labels,    # {state_id: label_str}
    'features':          FEATURES,
    'n_components':      N_COMPONENTS,
    'state_means':       state_means.tolist(),
    'state_vars':        state_vars.tolist(),
    'var_median':        float(var_median),
    'trained_on':        datetime.now().isoformat(),
    'trained_on_ticker': TICKER,
    'training_rows':     int(X.shape[0]),
    'spy_close_at_train': round(float(df['Close'].iloc[-1]), 2),
}

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(artifact, f)

print(f"Model saved  ->  {MODEL_PATH}")

# ── Quick sanity: predict last state ──────────────────────────────────────────
states = model.predict(X)
last_state = states[-1]
last_label = state_labels[last_state]
last_probs = model.predict_proba(X)[-1]
last_conf  = round(float(last_probs[last_state]) * 100, 1)

print()
print(f"Current SPY regime  ->  {last_label}  ({last_conf}% confidence)")
print(f"SPY close at train  ->  ${artifact['spy_close_at_train']}")
print()
print("Done.  Run  python -c \"from lib.regime_engine import get_current_regime; print(get_current_regime())\"  to verify.")
