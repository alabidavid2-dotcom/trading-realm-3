# ================================================
# REGIME_ENGINE.PY  —  SPY Market Regime Inference
# Loads the trained HMM from models/hmm_model.pkl
# and classifies the current market environment.
#
# Regimes:
#   Bull Quiet    — trending up, low noise       (best for swing longs)
#   Bull Volatile — trending up, high swings     (0DTE calls work; size down)
#   Bear Quiet    — slowly drifting down         (swing puts)
#   Bear Volatile — aggressive sell-off          (0DTE puts; size down)
#   Chop          — no trend, mean-reverting     (stay flat or reduce)
# ================================================

from __future__ import annotations

import os
import pickle
import numpy as np
from functools import lru_cache
from datetime import datetime

_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hmm_model.pkl')

# How many trading days to feed into the decoder.
# HMM Viterbi needs a sequence; 30 days gives stable state probabilities
# without dragging in stale regime history.
_DECODE_WINDOW = 30

# Regime metadata used by the UI (color, bias, sizing guidance)
REGIME_META: dict[str, dict] = {
    'Bull Quiet': {
        'color':    '#4ade80',   # green
        'bg':       '#14532d',
        'bias':     'LONG',
        'sizing':   'Full size — ideal conditions',
        'strategy': 'Swing longs, hold runners',
    },
    'Bull Volatile': {
        'color':    '#86efac',   # light green
        'bg':       '#1a2e1a',
        'bias':     'LONG',
        'sizing':   'Reduce size — elevated whipsaw risk',
        'strategy': '0DTE calls on dips, tight stops',
    },
    'Bear Quiet': {
        'color':    '#f87171',   # red
        'bg':       '#7f1d1d',
        'bias':     'SHORT',
        'sizing':   'Full size — clean downtrend',
        'strategy': 'Swing puts, hold runners',
    },
    'Bear Volatile': {
        'color':    '#fca5a5',   # light red
        'bg':       '#3a1212',
        'bias':     'SHORT',
        'sizing':   'Reduce size — crash-mode volatility',
        'strategy': '0DTE puts on bounces, tight stops',
    },
    'Chop': {
        'color':    '#f59e0b',   # amber
        'bg':       '#1c1508',
        'bias':     'FLAT',
        'sizing':   'Stay flat or minimal size',
        'strategy': 'Wait for regime break — no runners',
    },
}

# Fallback returned when the model file does not exist yet.
_NOT_TRAINED = {
    'regime':      'Not Trained',
    'confidence':  0.0,
    'state_id':    -1,
    'probabilities': {},
    'meta':        {},
    'spy_close':   None,
    'spy_log_return': None,
    'trained_on':  None,
    'error':       'Run  python train_hmm.py  to train the regime model.',
}


def _load_artifact() -> dict:
    """Load and return the pickled model artifact. Raises FileNotFoundError if missing."""
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {_MODEL_PATH}. Run train_hmm.py first.")
    with open(_MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def model_exists() -> bool:
    """Quick check — use this in the UI to show 'Train Model' prompt."""
    return os.path.exists(_MODEL_PATH)


def get_model_info() -> dict:
    """
    Return metadata about the trained model without running inference.
    Safe to call even when the model doesn't exist (returns error dict).
    """
    if not model_exists():
        return {'trained': False, 'error': 'Model not trained yet.'}
    try:
        artifact = _load_artifact()
        return {
            'trained':       True,
            'trained_on':    artifact.get('trained_on'),
            'training_rows': artifact.get('training_rows'),
            'n_components':  artifact.get('n_components'),
            'spy_close_at_train': artifact.get('spy_close_at_train'),
            'features':      artifact.get('features'),
            'state_labels':  artifact.get('state_labels'),
        }
    except Exception as e:
        return {'trained': False, 'error': str(e)}


def get_current_regime(window: int = _DECODE_WINDOW) -> dict:
    """
    Predict the current SPY market regime.

    Pipeline
    --------
    1. Load trained GaussianHMM from disk.
    2. Fetch the last `window` trading days of SPY daily bars.
    3. Build feature matrix  [log_return, range].
    4. Run Viterbi decoding → current hidden state.
    5. Run forward-backward  → posterior probability per state.
    6. Map state → human-readable regime label.

    Returns
    -------
    dict with keys:
        regime          – e.g. 'Bull Quiet'
        confidence      – posterior probability of the current state (%)
        state_id        – raw HMM state integer
        probabilities   – {label: pct} for all regimes (sums to ~100)
        meta            – color / sizing / strategy guidance from REGIME_META
        spy_close       – last SPY close used for inference
        spy_log_return  – last log return (%)
        trained_on      – ISO timestamp of when model was trained
        error           – populated only on failure
    """
    # ── Guard: model not trained ──────────────────────────────────────────────
    if not model_exists():
        return dict(_NOT_TRAINED)

    try:
        artifact     = _load_artifact()
        model        = artifact['model']
        state_labels = artifact['state_labels']        # {int: str}
        features     = artifact.get('features', ['log_return', 'range'])

        # ── Fetch SPY data ────────────────────────────────────────────────────
        from lib.data_client import get_daily
        # Fetch 2× window to guarantee enough rows after dropna
        df = get_daily('SPY', days=window * 2 + 20)

        if df is None or len(df) < 10:
            return {**dict(_NOT_TRAINED), 'error': 'Insufficient SPY data from Alpaca.'}

        # ── Feature engineering ───────────────────────────────────────────────
        import pandas as _pd
        df = df.copy()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['range']      = (df['High'] - df['Low']) / df['Close']
        df = df.dropna().tail(window)

        if len(df) < 5:
            return {**dict(_NOT_TRAINED), 'error': f'Only {len(df)} rows after dropna — need ≥ 5.'}

        X = df[features].values

        # ── Decode ────────────────────────────────────────────────────────────
        states      = model.predict(X)          # Viterbi path
        posteriors  = model.predict_proba(X)    # forward-backward probabilities

        current_state = int(states[-1])
        current_probs = posteriors[-1]          # shape: (n_components,)

        # ── Map state → label ─────────────────────────────────────────────────
        regime     = state_labels.get(current_state, f'State_{current_state}')
        confidence = round(float(current_probs[current_state]) * 100, 1)

        # Aggregate probabilities per label (two states could share a label
        # if the model learns redundant states)
        prob_by_label: dict[str, float] = {}
        for sid, prob in enumerate(current_probs):
            label = state_labels.get(sid, f'State_{sid}')
            prob_by_label[label] = round(
                prob_by_label.get(label, 0.0) + float(prob) * 100, 1
            )

        last_lr = float(df['log_return'].iloc[-1])

        return {
            'regime':         regime,
            'confidence':     confidence,
            'state_id':       current_state,
            'probabilities':  prob_by_label,
            'meta':           REGIME_META.get(regime, {}),
            'spy_close':      round(float(df['Close'].iloc[-1]), 2),
            'spy_log_return': round(last_lr * 100, 4),   # in percent
            'trained_on':     artifact.get('trained_on'),
            'decode_window':  len(df),
            'error':          None,
        }

    except Exception as exc:
        return {**dict(_NOT_TRAINED), 'error': str(exc)}


def get_regime_history(window: int = 90) -> list[dict]:
    """
    Return the day-by-day regime sequence for the last `window` trading days.
    Useful for plotting a regime timeline in the UI.

    Returns list of {date, regime, state_id, log_return, range} dicts.
    """
    if not model_exists():
        return []
    try:
        artifact     = _load_artifact()
        model        = artifact['model']
        state_labels = artifact['state_labels']
        features     = artifact.get('features', ['log_return', 'range'])

        from lib.data_client import get_daily
        df = get_daily('SPY', days=window + 30)
        if df is None or len(df) < 10:
            return []

        df = df.copy()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['range']      = (df['High'] - df['Low']) / df['Close']
        df = df.dropna().tail(window)

        X      = df[features].values
        states = model.predict(X)

        history = []
        for i, (idx, row) in enumerate(df.iterrows()):
            sid = int(states[i])
            history.append({
                'date':       str(idx.date()),
                'regime':     state_labels.get(sid, f'State_{sid}'),
                'state_id':   sid,
                'log_return': round(float(row['log_return']) * 100, 4),
                'range':      round(float(row['range']) * 100, 2),
                'close':      round(float(row['Close']), 2),
            })
        return history

    except Exception:
        return []
