# ================================================
# HMM_REGIME.PY - Hidden Markov Model Regime Detection
# Real probabilistic regime detection using hmmlearn
# ================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("⚠️  hmmlearn not installed. Run: pip install hmmlearn")
    print("   Falling back to quantile-based regime detection.")

from config import (
    HMM_N_REGIMES, HMM_TRAIN_DAYS, HMM_COVARIANCE,
    HMM_N_ITER, HMM_RANDOM_SEED, HMM_FEATURES, REGIME_LABELS
)


def prepare_features(df):
    """
    Build the feature matrix for HMM training.
    Features: daily return, 20d rolling vol, volume ratio, daily range %.
    """
    df = df.copy()
    df['daily_return'] = df['Close'].pct_change()
    df['vol_20'] = df['daily_return'].rolling(20).std() * np.sqrt(252)
    df['range_pct'] = (df['High'] - df['Low']) / df['Close']

    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    else:
        df['volume_ratio'] = 1.0  # Fallback if no volume data

    df = df.dropna()
    return df


def train_hmm(df, n_regimes=HMM_N_REGIMES):
    """
    Train a Gaussian HMM on the feature matrix.
    Returns the trained model and the feature-enriched dataframe.
    """
    df = prepare_features(df)
    feature_cols = [c for c in HMM_FEATURES if c in df.columns]
    X = df[feature_cols].values

    if not HMM_AVAILABLE:
        return None, df, feature_cols

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type=HMM_COVARIANCE,
        n_iter=HMM_N_ITER,
        random_state=HMM_RANDOM_SEED,
        verbose=False
    )
    model.fit(X)
    return model, df, feature_cols


def label_regimes(model, df, feature_cols):
    """
    Predict regimes and assign human-readable labels.
    Labels are sorted by the mean return of each regime:
      lowest mean return → Bear_Volatile
      highest mean return → Bull_Volatile
    """
    X = df[feature_cols].values

    if model is None:
        # Fallback: quantile-based proxy (matches original code logic)
        return _fallback_regime(df)

    hidden_states = model.predict(X)
    df = df.copy()
    df['regime_id'] = hidden_states

    # Sort regimes by mean return to assign meaningful labels
    regime_means = df.groupby('regime_id')['daily_return'].mean().sort_values()
    sorted_ids = regime_means.index.tolist()

    n = len(sorted_ids)
    all_label_keys = sorted(REGIME_LABELS.keys())
    n_labels = len(all_label_keys)

    id_to_label = {}
    for i, rid in enumerate(sorted_ids):
        if n_labels == 0:
            id_to_label[rid] = f"Regime_{rid}"
        elif n <= n_labels:
            # n states fit within labels: map by sorted position
            id_to_label[rid] = REGIME_LABELS[all_label_keys[i]]
        else:
            # more states than labels: space them evenly across label range
            label_idx = round(i * (n_labels - 1) / (n - 1)) if n > 1 else 0
            id_to_label[rid] = REGIME_LABELS[all_label_keys[label_idx]]

    df['regime'] = df['regime_id'].map(id_to_label).fillna('Chop')
    return df


def get_transition_matrix(model):
    """
    Extract the transition probability matrix from the trained HMM.
    Returns a DataFrame showing P(next_state | current_state).
    """
    if model is None or not HMM_AVAILABLE:
        return None

    trans = model.transmat_
    n = trans.shape[0]
    labels = [REGIME_LABELS.get(i, f"Regime_{i}") for i in range(n)]

    # Re-sort to match label_regimes ordering
    return pd.DataFrame(trans, index=labels, columns=labels)


def get_regime_probabilities(model, df, feature_cols):
    """
    Get the probability distribution over regimes for the most recent observation.
    This tells you confidence: "80% Bull_Quiet, 15% Chop, 5% Bear_Quiet"
    """
    if model is None or not HMM_AVAILABLE:
        return None

    X = df[feature_cols].values
    posteriors = model.predict_proba(X)
    latest_probs = posteriors[-1]

    # Sort to match label ordering
    regime_means = df.copy()
    regime_means['regime_id'] = model.predict(X)
    means = regime_means.groupby('regime_id')['daily_return'].mean().sort_values()
    sorted_ids = means.index.tolist()

    n = len(sorted_ids)
    all_label_keys = sorted(REGIME_LABELS.keys())
    n_labels = len(all_label_keys)

    prob_dict = {}
    for i, rid in enumerate(sorted_ids):
        if n_labels == 0:
            label = f"Regime_{rid}"
        elif n <= n_labels:
            label = REGIME_LABELS[all_label_keys[i]]
        else:
            label_idx = round(i * (n_labels - 1) / (n - 1)) if n > 1 else 0
            label = REGIME_LABELS[all_label_keys[label_idx]]
        prob_dict[label] = round(float(latest_probs[rid]) * 100, 1)

    return prob_dict


def _fallback_regime(df):
    """
    Quantile-based regime detection (no ML required).
    Used when hmmlearn is not available.
    """
    df = df.copy()
    ret_q = df['daily_return'].quantile([0.15, 0.85])
    vol_q = df['vol_20'].quantile(0.5)

    def assign(row):
        r, v = row['daily_return'], row['vol_20']
        if r > ret_q[0.85] and v < vol_q:
            return "Bull_Quiet"
        elif r > ret_q[0.85]:
            return "Bull_Volatile"
        elif r < ret_q[0.15] and v < vol_q:
            return "Bear_Quiet"
        elif r < ret_q[0.15]:
            return "Bear_Volatile"
        else:
            return "Chop"

    df['regime'] = df.apply(assign, axis=1)
    df['regime_id'] = df['regime'].factorize()[0]
    return df


def analyze_regime(ticker, df_raw):
    """
    Full regime analysis pipeline for a single ticker.
    Returns dict with regime info, probabilities, and transition matrix.
    """
    model, df, feature_cols = train_hmm(df_raw)
    df = label_regimes(model, df, feature_cols)
    probs = get_regime_probabilities(model, df, feature_cols)
    trans = get_transition_matrix(model)

    latest = df.iloc[-1]

    return {
        'ticker': ticker,
        'close': round(float(latest['Close']), 2),
        'regime': latest['regime'],
        'regime_id': int(latest['regime_id']),
        'vol_20': round(float(latest['vol_20']) * 100, 2),
        'regime_probabilities': probs,
        'transition_matrix': trans,
        'model': model,
        'df': df,
        'feature_cols': feature_cols,
        'hmm_active': model is not None,
    }
