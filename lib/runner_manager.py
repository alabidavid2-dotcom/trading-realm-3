# ================================================
# RUNNER_MANAGER.PY — Savage Runner Protocol
# Tracks active runner legs and enforces the
# 33% Profit Guard + Breakeven Stop lifecycle.
# ================================================

from __future__ import annotations   # enables dict | None on Python 3.9
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RunnerState:
    occ_symbol:    str
    ticker:        str
    direction:     str          # 'CALL' or 'PUT'
    entry_premium: float        # ask price at entry (per share)
    qty:           int          # runner contracts remaining
    contract_hod:  float        # highest premium since entry
    entry_time:    datetime = field(default_factory=datetime.now)
    t1_hit:        bool     = False
    closed:        bool     = False
    close_reason:  str      = ''

    @property
    def trigger_price(self) -> float:
        """33% pullback from HOD."""
        return round(self.contract_hod * 0.67, 2)

    @property
    def cushion_pct(self) -> float:
        """Current HOD vs trigger, as a percentage of HOD."""
        return round((self.contract_hod - self.trigger_price) / self.contract_hod * 100, 1)


class RunnerManager:
    """
    Tracks active runner contracts and enforces two exit rules (in priority order):

    1. 33% Profit Guard  — if current premium pulls back ≥33% from HOD, close immediately.
       This overrides the breakeven stop once the contract is significantly in profit.
    2. Breakeven Stop    — active only after T1 (scale-out) is hit. Closes at entry premium.

    Usage
    -----
    rm = RunnerManager()
    rm.add_runner(occ_symbol, ticker, 'CALL', entry_premium=2.50, qty=1)

    # In monitoring loop (per price update):
    result = rm.update(occ_symbol, current_premium=3.10)
    if result['action'] == 'close':
        executor.close_position(occ_symbol)
        log_order(f"Runner Closed: {result['reason']}")
    """

    def __init__(self):
        self.runners: dict[str, RunnerState] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def add_runner(
        self,
        occ_symbol:    str,
        ticker:        str,
        direction:     str,
        entry_premium: float,
        qty:           int,
    ) -> RunnerState:
        state = RunnerState(
            occ_symbol=occ_symbol,
            ticker=ticker,
            direction=direction,
            entry_premium=entry_premium,
            qty=qty,
            contract_hod=entry_premium,
        )
        self.runners[occ_symbol] = state
        return state

    def mark_t1_hit(self, occ_symbol: str) -> None:
        """Call when the scale-out limit order fills at T1. Activates breakeven stop."""
        r = self.runners.get(occ_symbol)
        if r:
            r.t1_hit = True

    # ── Price update + exit logic ─────────────────────────────────────────────

    def update(self, occ_symbol: str, current_premium: float) -> dict:
        """
        Feed each new premium price. Returns action dict.
        action == 'close'  → execute market close immediately.
        action == 'hold'   → no action needed.
        """
        r = self.runners.get(occ_symbol)
        if not r or r.closed:
            return {'action': 'hold', 'reason': 'no active runner'}

        # Track HOD
        if current_premium > r.contract_hod:
            r.contract_hod = current_premium

        trigger = r.trigger_price

        # ── Rule 1: 33% Profit Guard (highest priority) ───────────────────────
        # Only armed once the contract is at least 5% above entry (avoids noise at entry)
        if current_premium <= trigger and r.contract_hod > r.entry_premium * 1.05:
            r.closed = True
            r.close_reason = f'33% Pullback from Peak (${r.contract_hod:.2f})'
            return {
                'action':  'close',
                'reason':  r.close_reason,
                'peak':    r.contract_hod,
                'trigger': trigger,
                'current': current_premium,
                'log_msg': f"Runner Closed: {r.close_reason}",
            }

        # ── Rule 2: Breakeven Stop (only after T1 is cleared) ─────────────────
        if r.t1_hit and current_premium <= r.entry_premium:
            r.closed = True
            r.close_reason = f'Breakeven Stop (${r.entry_premium:.2f})'
            return {
                'action':  'close',
                'reason':  r.close_reason,
                'peak':    r.contract_hod,
                'trigger': r.entry_premium,
                'current': current_premium,
                'log_msg': f"Runner Closed: {r.close_reason}",
            }

        gain_pct = round((current_premium / r.entry_premium - 1) * 100, 1)
        return {
            'action':   'hold',
            'hod':      r.contract_hod,
            'trigger':  trigger,
            'current':  current_premium,
            'gain_pct': gain_pct,
        }

    # ── Display helpers ───────────────────────────────────────────────────────

    def get_display(self, occ_symbol: str) -> dict | None:
        r = self.runners.get(occ_symbol)
        if not r:
            return None
        return {
            'symbol':       r.occ_symbol,
            'ticker':       r.ticker,
            'direction':    r.direction,
            'qty':          r.qty,
            'entry':        r.entry_premium,
            'hod':          r.contract_hod,
            'trigger':      r.trigger_price,
            'closed':       r.closed,
            't1_hit':       r.t1_hit,
            'close_reason': r.close_reason,
            'entry_time':   r.entry_time.strftime('%I:%M %p'),
        }

    def active_runners(self) -> list[dict]:
        return [self.get_display(s) for s, r in self.runners.items() if not r.closed]

    def all_runners(self) -> list[dict]:
        return [self.get_display(s) for s in self.runners]

    def closed_runners(self) -> list[dict]:
        return [self.get_display(s) for s, r in self.runners.items() if r.closed]

    def has_active(self) -> bool:
        return any(not r.closed for r in self.runners.values())
