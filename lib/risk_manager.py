# ================================================
# RISK_MANAGER.PY - Position Sizing & Trade Management
# Regime-adjusted risk, cooldowns, max daily loss
# ================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import (
    RISK_BASE_0DTE, RISK_BASE_SWING, CONTRACTS_MIN, CONTRACTS_MAX,
    REGIME_RISK_MULTIPLIER, COOLDOWN_HOURS_0DTE, COOLDOWN_HOURS_SWING,
    COMMISSION_PER_TRADE
)


class RiskManager:
    """
    Manages risk per trade, daily limits, and cooldown enforcement.
    """

    def __init__(self, capital=10000):
        self.capital = capital
        self.daily_pnl = 0.0
        self.trades_today = []
        self.last_exit_time = None
        self.max_daily_loss = -300   # Stop trading if you lose this much in a day
        self.max_daily_trades = 6    # No overtrading

    def calculate_position_size(self, signal, option_price=None):
        """
        Calculate position size based on signal strength, regime, and risk budget.

        Args:
            signal: dict from signal_engine.generate_signal()
            option_price: current option premium (if known)

        Returns:
            dict with risk_amount, contracts, stop_loss_pct, etc.
        """
        trade_type = signal['trade_type']
        regime_mult = signal['risk_multiplier']

        # Base risk by trade type
        if '0DTE' in trade_type:
            base_risk = RISK_BASE_0DTE
        else:
            base_risk = RISK_BASE_SWING

        # Adjust for regime
        adjusted_risk = base_risk * regime_mult

        # Adjust for signal strength
        strength_mult = {
            'STRONG': 1.0,
            'MODERATE': 0.7,
            'WEAK': 0.4,
            'NO_TRADE': 0.0,
        }.get(signal['strength'], 0.5)
        adjusted_risk *= strength_mult

        # Round to nearest $5
        adjusted_risk = round(adjusted_risk / 5) * 5
        adjusted_risk = max(25, min(adjusted_risk, base_risk * 1.2))  # Floor and cap

        # Calculate contracts
        if option_price and option_price > 0:
            max_contracts_by_risk = int(adjusted_risk / (option_price * 100))
            contracts = max(CONTRACTS_MIN, min(max_contracts_by_risk, CONTRACTS_MAX))
        else:
            contracts = CONTRACTS_MIN

        return {
            'risk_amount': adjusted_risk,
            'contracts': contracts,
            'base_risk': base_risk,
            'regime_multiplier': regime_mult,
            'strength_multiplier': strength_mult,
            'trade_type': trade_type,
        }

    def check_cooldown(self, trade_type):
        """
        Check if cooldown period has elapsed since last exit.
        Returns (is_clear, time_remaining_str)
        """
        if self.last_exit_time is None:
            return True, "No recent exits"

        if '0DTE' in trade_type:
            cooldown = timedelta(hours=COOLDOWN_HOURS_0DTE)
        else:
            cooldown = timedelta(hours=COOLDOWN_HOURS_SWING)

        elapsed = datetime.now() - self.last_exit_time
        if elapsed >= cooldown:
            return True, "Cooldown clear"
        else:
            remaining = cooldown - elapsed
            mins = int(remaining.total_seconds() / 60)
            return False, f"Cooldown: {mins} min remaining"

    def check_daily_limits(self):
        """
        Check if daily loss limit or max trades reached.
        Returns (can_trade, reason)
        """
        if self.daily_pnl <= self.max_daily_loss:
            return False, f"DAILY LOSS LIMIT HIT (${self.daily_pnl:.0f}). Done for today."

        if len(self.trades_today) >= self.max_daily_trades:
            return False, f"MAX TRADES ({self.max_daily_trades}) reached. Done for today."

        return True, "Within daily limits"

    def pre_trade_check(self, signal):
        """
        Full pre-trade validation. Returns (approved, details_dict)
        """
        # Check if signal is tradeable
        if signal['strength'] == 'NO_TRADE' or signal['direction'] == 'FLAT':
            return False, {'reason': 'No actionable signal'}

        # Daily limits
        can_trade, limit_msg = self.check_daily_limits()
        if not can_trade:
            return False, {'reason': limit_msg}

        # Cooldown
        clear, cooldown_msg = self.check_cooldown(signal['trade_type'])
        if not clear:
            return False, {'reason': cooldown_msg}

        # Position sizing
        sizing = self.calculate_position_size(signal)

        return True, {
            'approved': True,
            'sizing': sizing,
            'daily_pnl': self.daily_pnl,
            'trades_today': len(self.trades_today),
        }

    def record_trade(self, entry_price, exit_price, contracts, trade_type, entry_time=None, exit_time=None):
        """Record a completed trade for daily tracking."""
        if '0DTE' in trade_type:
            # Options: PnL = (exit - entry) * 100 * contracts - commission
            pnl = (exit_price - entry_price) * 100 * contracts - (COMMISSION_PER_TRADE * contracts * 2)
        else:
            # Equity: commission is per-share (contracts = shares here)
            pnl = (exit_price - entry_price) * contracts - (COMMISSION_PER_TRADE * contracts * 2)

        trade = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'contracts': contracts,
            'trade_type': trade_type,
            'pnl': round(pnl, 2),
            'entry_time': entry_time or datetime.now(),
            'exit_time': exit_time or datetime.now(),
        }

        self.trades_today.append(trade)
        self.daily_pnl += pnl
        self.last_exit_time = trade['exit_time']
        self.capital += pnl

        return trade

    def reset_daily(self):
        """Call at start of each trading day."""
        self.daily_pnl = 0.0
        self.trades_today = []
        self.last_exit_time = None

    def summary(self):
        """Return current risk state."""
        return {
            'capital': round(self.capital, 2),
            'daily_pnl': round(self.daily_pnl, 2),
            'trades_today': len(self.trades_today),
            'max_daily_loss': self.max_daily_loss,
            'max_daily_trades': self.max_daily_trades,
        }
