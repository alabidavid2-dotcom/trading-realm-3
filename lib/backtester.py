# ================================================
# BACKTESTER.PY - Historical Backtest Engine
# Simulates the full signal pipeline on historical data
# Reports: Win rate, Sharpe, Max DD, Total Return, Alpha
# ================================================

import numpy as np
import pandas as pd
from lib.indicators import add_all_indicators, indicator_snapshot
from lib.strat_classifier import add_strat_columns, detect_strat_patterns
from lib.signal_engine import generate_signal, filter_grade_a_only
from lib.risk_manager import RiskManager
from config import INITIAL_CAPITAL, COMMISSION_PER_TRADE


class Backtester:
    """
    Walk-forward backtester that simulates the full system on historical data.
    Uses the same signal_engine logic that would run live.
    """

    def __init__(self, regime_df, ticker='SPY', capital=INITIAL_CAPITAL):
        """
        Args:
            regime_df: DataFrame with regime column already assigned (from hmm_regime)
            ticker: Symbol for labeling
            capital: Starting capital
        """
        self.ticker = ticker
        self.initial_capital = capital
        self.regime_df = regime_df.copy()
        self.trades = []
        self.equity_curve = []

    def run(self, hold_bars=5, stop_loss_atr=2.0, take_profit_atr=3.0, min_bars=50):
        """
        Run the backtest.

        Args:
            hold_bars: Default holding period (bars) if no SL/TP hit
            stop_loss_atr: Stop loss as multiple of ATR
            take_profit_atr: Take profit as multiple of ATR
            min_bars: Minimum bars before we start generating signals (warmup)
        """
        df = self.regime_df.copy()
        df = add_all_indicators(df)
        df = add_strat_columns(df)
        df = df.dropna().reset_index(drop=True)

        capital = self.initial_capital
        position = None  # None = flat, dict = active trade
        risk_mgr = RiskManager(capital)
        cooldown_until = 0

        for i in range(min_bars, len(df)):
            row = df.iloc[i]
            regime = row.get('regime', 'Chop')

            # Track equity
            self.equity_curve.append({
                'date': row.name if hasattr(row.name, 'strftime') else i,
                'equity': capital,
                'regime': regime,
            })

            # --- CHECK EXISTING POSITION ---
            if position is not None:
                bars_held = i - position['entry_bar']
                current_price = row['Close']

                # Stop loss
                hit_sl = False
                hit_tp = False
                if position['direction'] == 'LONG':
                    if row['Low'] <= position['stop_loss']:
                        hit_sl = True
                        exit_price = position['stop_loss']
                    elif row['High'] >= position['take_profit']:
                        hit_tp = True
                        exit_price = position['take_profit']
                else:  # SHORT
                    if row['High'] >= position['stop_loss']:
                        hit_sl = True
                        exit_price = position['stop_loss']
                    elif row['Low'] <= position['take_profit']:
                        hit_tp = True
                        exit_price = position['take_profit']

                # Time-based exit
                time_exit = bars_held >= hold_bars

                if hit_sl or hit_tp or time_exit:
                    if not hit_sl and not hit_tp:
                        exit_price = current_price

                    # Calculate PnL
                    if position['direction'] == 'LONG':
                        pnl = (exit_price - position['entry_price']) * position['shares']
                    else:
                        pnl = (position['entry_price'] - exit_price) * position['shares']

                    pnl -= COMMISSION_PER_TRADE * position['shares'] * 2  # per-share, entry + exit

                    trade_record = {
                        'entry_bar': position['entry_bar'],
                        'exit_bar': i,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': round(exit_price, 2),
                        'shares': position['shares'],
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl / position['risk_amount'] * 100, 1) if position['risk_amount'] else 0,
                        'regime': position['entry_regime'],
                        'signal_score': position['signal_score'],
                        'exit_reason': 'SL' if hit_sl else ('TP' if hit_tp else 'TIME'),
                        'bars_held': bars_held,
                    }
                    self.trades.append(trade_record)
                    capital += pnl
                    position = None
                    cooldown_until = i + 3  # 3-bar cooldown
                    continue

            # --- COOLDOWN CHECK ---
            if i < cooldown_until:
                continue

            # --- GENERATE SIGNAL ---
            # Build indicator snapshot from current bar
            window = df.iloc[max(0, i - 50):i + 1]
            try:
                ind_snap = indicator_snapshot(window)
            except Exception:
                continue

            # Strat patterns on recent bars
            strat_window = df.iloc[max(0, i - 10):i + 1]
            strat_pats = detect_strat_patterns(strat_window)

            signal = generate_signal(regime, ind_snap, strat_pats)

            # Filter: only trade Grade A+ / A setups
            if not filter_grade_a_only(signal):
                continue

            # --- ENTER POSITION ---
            entry_price = row['Close']
            atr = row.get('atr', entry_price * 0.01)  # Fallback 1% if no ATR

            if signal['direction'] == 'LONG':
                stop_loss = entry_price - (atr * stop_loss_atr)
                take_profit = entry_price + (atr * take_profit_atr)
            else:  # SHORT
                stop_loss = entry_price + (atr * stop_loss_atr)
                take_profit = entry_price - (atr * take_profit_atr)

            risk_amount = abs(entry_price - stop_loss)
            if risk_amount == 0:
                continue

            # Size: risk a % of capital
            risk_dollars = capital * 0.01 * signal['risk_multiplier']  # 1% base
            shares = max(1, int(risk_dollars / risk_amount))

            position = {
                'entry_bar': i,
                'entry_price': entry_price,
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'direction': signal['direction'],
                'shares': shares,
                'risk_amount': risk_dollars,
                'entry_regime': regime,
                'signal_score': signal['composite_score'],
            }

        # Close any open position at end
        if position is not None:
            exit_price = df.iloc[-1]['Close']
            if position['direction'] == 'LONG':
                pnl = (exit_price - position['entry_price']) * position['shares']
            else:
                pnl = (position['entry_price'] - exit_price) * position['shares']
            pnl -= COMMISSION_PER_TRADE * position['shares'] * 2
            self.trades.append({
                'entry_bar': position['entry_bar'],
                'exit_bar': len(df) - 1,
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': round(exit_price, 2),
                'shares': position['shares'],
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl / position['risk_amount'] * 100, 1) if position['risk_amount'] else 0,
                'regime': position['entry_regime'],
                'signal_score': position['signal_score'],
                'exit_reason': 'END',
                'bars_held': len(df) - 1 - position['entry_bar'],
            })
            capital += pnl

        self.final_capital = capital
        return self.compute_metrics()

    def compute_metrics(self):
        """Compute performance metrics from trade history."""
        if not self.trades:
            return {'error': 'No trades generated. Check signal thresholds.'}

        trades_df = pd.DataFrame(self.trades)
        pnls = trades_df['pnl'].values
        winners = pnls[pnls > 0]
        losers = pnls[pnls < 0]

        total_return = (self.final_capital - self.initial_capital) / self.initial_capital * 100
        win_rate = len(winners) / len(pnls) * 100 if len(pnls) > 0 else 0
        avg_win = winners.mean() if len(winners) > 0 else 0
        avg_loss = losers.mean() if len(losers) > 0 else 0
        profit_factor = abs(winners.sum() / losers.sum()) if len(losers) > 0 and losers.sum() != 0 else float('inf')

        # Sharpe-like ratio (on trade PnLs)
        if pnls.std() > 0:
            sharpe = (pnls.mean() / pnls.std()) * np.sqrt(252)  # Annualized approx
        else:
            sharpe = 0

        # Max drawdown from equity curve
        equity = pd.DataFrame(self.equity_curve)
        if len(equity) > 0:
            eq = equity['equity']
            running_max = eq.cummax()
            drawdown = (eq - running_max) / running_max * 100
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0

        # Regime breakdown
        regime_stats = trades_df.groupby('regime').agg(
            trades=('pnl', 'count'),
            win_rate=('pnl', lambda x: (x > 0).mean() * 100),
            avg_pnl=('pnl', 'mean'),
            total_pnl=('pnl', 'sum'),
        ).round(2)

        # Exit reason breakdown
        exit_stats = trades_df.groupby('exit_reason').agg(
            trades=('pnl', 'count'),
            avg_pnl=('pnl', 'mean'),
        ).round(2)

        metrics = {
            'ticker': self.ticker,
            'total_trades': len(pnls),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': round(win_rate, 1),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe, 2),
            'total_return_pct': round(total_return, 2),
            'total_return_dollars': round(self.final_capital - self.initial_capital, 2),
            'final_capital': round(self.final_capital, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'avg_bars_held': round(trades_df['bars_held'].mean(), 1),
            'regime_breakdown': regime_stats,
            'exit_breakdown': exit_stats,
        }

        self.metrics = metrics
        self.trades_df = trades_df
        return metrics

    def print_report(self):
        """Print a formatted backtest report."""
        m = self.metrics if hasattr(self, 'metrics') else self.run()

        print(f"\n{'='*60}")
        print(f"  BACKTEST REPORT: {self.ticker}")
        print(f"{'='*60}")
        print(f"  Total Trades:      {m['total_trades']}")
        print(f"  Win Rate:          {m['win_rate']}%")
        print(f"  Winners/Losers:    {m['winners']} / {m['losers']}")
        print(f"  Avg Win:           ${m['avg_win']}")
        print(f"  Avg Loss:          ${m['avg_loss']}")
        print(f"  Profit Factor:     {m['profit_factor']}")
        print(f"  Sharpe Ratio:      {m['sharpe_ratio']}")
        print(f"  Total Return:      {m['total_return_pct']}% (${m['total_return_dollars']})")
        print(f"  Max Drawdown:      {m['max_drawdown_pct']}%")
        print(f"  Final Capital:     ${m['final_capital']}")
        print(f"  Avg Bars Held:     {m['avg_bars_held']}")
        print(f"\n--- By Regime ---")
        print(m['regime_breakdown'].to_string())
        print(f"\n--- By Exit Type ---")
        print(m['exit_breakdown'].to_string())
        print(f"{'='*60}\n")
