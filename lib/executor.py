# ================================================
# EXECUTOR.PY - Alpaca Order Execution Engine
# Paper trading via Alpaca API
# Handles equity (swing) and options (0DTE) orders
# ================================================

import time
from datetime import datetime, date
from typing import Optional

from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER_TRADING,
    ALPACA_BASE_URL, ALPACA_DATA_URL,
    RISK_BASE_0DTE, RISK_BASE_SWING,
)
from lib.risk_manager import RiskManager

# --- Alpaca SDK import (alpaca-py) ---
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest,
    GetOrdersRequest, ClosePositionRequest,
    OptionLegRequest,
)
from alpaca.trading.enums import (
    OrderSide, TimeInForce, OrderStatus,
    AssetClass, ContractType,
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
ALPACA_AVAILABLE = True


class Executor:
    """
    Submits trades to Alpaca paper account based on signals from signal_engine.
    Works in tandem with RiskManager for sizing and daily limit enforcement.
    """

    def __init__(self, risk_manager: Optional[RiskManager] = None):
        self.paper = ALPACA_PAPER_TRADING
        self.rm = risk_manager or RiskManager()
        self.order_log = []     # In-session order history
        self.open_positions = {}  # {ticker: order_id}

        if not ALPACA_AVAILABLE:
            self.client = None
            self.data_client = None
            print("[Executor] Running in DRY-RUN mode (alpaca-py not installed).")
            return

        self.client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=self.paper,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
        )
        mode = "PAPER" if self.paper else "LIVE"
        print(f"[Executor] Connected to Alpaca ({mode})")

    # --------------------------------------------------
    # ACCOUNT INFO
    # --------------------------------------------------

    def get_account(self):
        """Return Alpaca account details."""
        if not self.client:
            return {"status": "DRY_RUN", "buying_power": 0}
        try:
            acct = self.client.get_account()
            return {
                "status": acct.status,
                "buying_power": float(acct.buying_power),
                "portfolio_value": float(acct.portfolio_value),
                "cash": float(acct.cash),
                "pattern_day_trader": acct.pattern_day_trader,
            }
        except Exception as e:
            print(f"[Executor] get_account error: {e}")
            return {}

    def get_positions(self):
        """Return all current open positions."""
        if not self.client:
            return []
        try:
            return self.client.get_all_positions()
        except Exception as e:
            print(f"[Executor] get_positions error: {e}")
            return []

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Fetch latest ask price for a ticker."""
        if not self.data_client:
            return None
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quote = self.data_client.get_stock_latest_quote(req)
            return float(quote[ticker].ask_price)
        except Exception as e:
            print(f"[Executor] get_latest_price({ticker}) error: {e}")
            return None

    # --------------------------------------------------
    # EQUITY ORDERS (Swing trades)
    # --------------------------------------------------

    def submit_equity_order(
        self,
        ticker: str,
        side: str,          # 'buy' or 'sell'
        qty: int,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        time_in_force: str = 'day',
    ) -> dict:
        """
        Submit an equity order (for swing long/short).
        Returns order dict with id, status, and details.
        """
        result = {
            "ticker": ticker,
            "side": side,
            "qty": qty,
            "order_type": order_type,
            "limit_price": limit_price,
            "timestamp": datetime.now().isoformat(),
            "paper": self.paper,
        }

        if not self.client:
            result["status"] = "DRY_RUN"
            result["order_id"] = f"DRY-{ticker}-{int(time.time())}"
            print(f"[Executor][DRY-RUN] {side.upper()} {qty} {ticker} @ {order_type.upper()}")
            self.order_log.append(result)
            return result

        try:
            tif = TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

            if order_type == 'market':
                req = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                )
            else:
                req = LimitOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=order_side,
                    limit_price=limit_price,
                    time_in_force=tif,
                )

            order = self.client.submit_order(req)
            result["status"] = str(order.status)
            result["order_id"] = str(order.id)
            result["filled_avg_price"] = float(order.filled_avg_price or 0)
            print(f"[Executor] {side.upper()} {qty} {ticker} — order {order.id} ({order.status})")

        except Exception as e:
            result["status"] = "ERROR"
            result["error"] = str(e)
            print(f"[Executor] Order error for {ticker}: {e}")

        self.order_log.append(result)
        return result

    # --------------------------------------------------
    # OPTIONS ORDERS (0DTE calls/puts)
    # --------------------------------------------------

    def find_0dte_contract(
        self,
        ticker: str,
        direction: str,     # 'CALL' or 'PUT'
        spot_price: float,
    ) -> Optional[str]:
        """
        Find an ATM 0DTE options contract symbol for the given ticker.
        Returns the Alpaca options symbol string or None.

        Alpaca options symbols follow OCC format:
            {TICKER}{YYMMDD}{C/P}{8-digit strike * 1000}
        Example: SPY240315C00510000
        """
        if not self.client:
            return None

        today = date.today().strftime("%y%m%d")
        contract_type = "C" if direction == "CALL" else "P"

        # Strike: nearest $1 increment to spot (ATM)
        strike = round(spot_price)
        strike_str = f"{int(strike * 1000):08d}"
        symbol = f"{ticker}{today}{contract_type}{strike_str}"
        return symbol

    def submit_options_order(
        self,
        ticker: str,
        direction: str,     # 'CALL' or 'PUT'
        contracts: int,
        spot_price: float,
    ) -> dict:
        """
        Submit a 0DTE options order (buy to open).
        Uses ATM strike expiring today.
        """
        result = {
            "ticker": ticker,
            "direction": direction,
            "contracts": contracts,
            "timestamp": datetime.now().isoformat(),
            "paper": self.paper,
            "trade_type": f"0DTE_{direction}",
        }

        symbol = self.find_0dte_contract(ticker, direction, spot_price)
        result["symbol"] = symbol

        if not self.client:
            result["status"] = "DRY_RUN"
            result["order_id"] = f"DRY-{symbol}-{int(time.time())}"
            print(f"[Executor][DRY-RUN] BUY {contracts} {symbol} (0DTE {direction})")
            self.order_log.append(result)
            return result

        try:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=contracts,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                asset_class=AssetClass.US_OPTION,
            )
            order = self.client.submit_order(req)
            result["status"] = str(order.status)
            result["order_id"] = str(order.id)
            result["filled_avg_price"] = float(order.filled_avg_price or 0)
            print(f"[Executor] BUY {contracts}x {symbol} — order {order.id} ({order.status})")
            self.open_positions[ticker] = str(order.id)

        except Exception as e:
            result["status"] = "ERROR"
            result["error"] = str(e)
            print(f"[Executor] Options order error for {symbol}: {e}")
            print(f"[Executor] Note: Ensure Alpaca account has options trading enabled.")

        self.order_log.append(result)
        return result

    # --------------------------------------------------
    # CLOSE POSITION
    # --------------------------------------------------

    def close_position(self, ticker: str) -> dict:
        """Close the entire position for a given ticker (equity or options)."""
        result = {"ticker": ticker, "action": "close", "timestamp": datetime.now().isoformat()}

        if not self.client:
            result["status"] = "DRY_RUN"
            print(f"[Executor][DRY-RUN] CLOSE position: {ticker}")
            return result

        try:
            self.client.close_position(ticker)
            result["status"] = "submitted"
            self.open_positions.pop(ticker, None)
            print(f"[Executor] Closed position: {ticker}")
        except Exception as e:
            result["status"] = "ERROR"
            result["error"] = str(e)
            print(f"[Executor] close_position error for {ticker}: {e}")

        return result

    def close_all_positions(self) -> list:
        """Close all open positions (e.g., EOD cleanup)."""
        if not self.client:
            print("[Executor][DRY-RUN] CLOSE ALL positions")
            return []
        try:
            self.client.close_all_positions(cancel_orders=True)
            self.open_positions.clear()
            print("[Executor] All positions closed.")
            return []
        except Exception as e:
            print(f"[Executor] close_all_positions error: {e}")
            return []

    # --------------------------------------------------
    # MAIN ENTRY POINT: Execute a signal
    # --------------------------------------------------

    def execute_signal(self, ticker: str, signal: dict) -> dict:
        """
        Main entry point. Takes a signal dict from signal_engine.generate_signal()
        and executes the appropriate order after risk checks.

        Args:
            ticker:  e.g. 'SPY'
            signal:  output of signal_engine.generate_signal()

        Returns:
            dict with execution result and risk details
        """
        trade_type = signal.get("trade_type", "NO_TRADE")
        direction = signal.get("direction", "FLAT")

        # 1. Pre-trade risk check
        approved, details = self.rm.pre_trade_check(signal)
        if not approved:
            reason = details.get("reason", "Blocked by risk manager")
            print(f"[Executor] BLOCKED — {ticker} | {reason}")
            return {"ticker": ticker, "status": "BLOCKED", "reason": reason}

        sizing = details["sizing"]
        contracts = sizing["contracts"]

        # 2. Fetch current price
        price = self.get_latest_price(ticker)
        if price is None:
            print(f"[Executor] Could not fetch price for {ticker} — skipping")
            return {"ticker": ticker, "status": "NO_PRICE"}

        print(f"[Executor] Signal: {ticker} | {trade_type} | {direction} | "
              f"{signal['strength']} | score={signal['composite_score']:+d} | "
              f"contracts={contracts} | price=${price:.2f}")

        # 3. Route to correct order type
        if trade_type == "NO_TRADE":
            return {"ticker": ticker, "status": "NO_TRADE"}

        elif trade_type in ("0DTE_CALL", "0DTE_PUT"):
            option_dir = "CALL" if trade_type == "0DTE_CALL" else "PUT"
            result = self.submit_options_order(
                ticker=ticker,
                direction=option_dir,
                contracts=contracts,
                spot_price=price,
            )

        elif trade_type == "SWING_LONG":
            # Calculate share qty from risk budget
            shares = max(1, int(sizing["risk_amount"] / price))
            result = self.submit_equity_order(
                ticker=ticker,
                side="buy",
                qty=shares,
            )

        elif trade_type == "SWING_SHORT":
            shares = max(1, int(sizing["risk_amount"] / price))
            result = self.submit_equity_order(
                ticker=ticker,
                side="sell",
                qty=shares,
            )

        else:
            return {"ticker": ticker, "status": "UNKNOWN_TRADE_TYPE", "trade_type": trade_type}

        result["sizing"] = sizing
        result["spot_price"] = price
        result["signal_score"] = signal["composite_score"]
        result["regime"] = signal.get("regime", "Unknown")
        return result

    # --------------------------------------------------
    # UTILITIES
    # --------------------------------------------------

    def order_status(self, order_id: str) -> Optional[str]:
        """Check the status of a submitted order."""
        if not self.client:
            return "DRY_RUN"
        try:
            order = self.client.get_order_by_id(order_id)
            return str(order.status)
        except Exception as e:
            print(f"[Executor] order_status error: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by ID."""
        if not self.client:
            print(f"[Executor][DRY-RUN] Cancel order {order_id}")
            return True
        try:
            self.client.cancel_order_by_id(order_id)
            print(f"[Executor] Cancelled order {order_id}")
            return True
        except Exception as e:
            print(f"[Executor] cancel_order error: {e}")
            return False

    def session_summary(self) -> dict:
        """Return a summary of all orders placed this session."""
        total = len(self.order_log)
        filled = sum(1 for o in self.order_log if o.get("status") not in ("ERROR", "DRY_RUN", "BLOCKED"))
        errors = sum(1 for o in self.order_log if o.get("status") == "ERROR")
        return {
            "orders_total": total,
            "orders_filled": filled,
            "orders_error": errors,
            "risk_summary": self.rm.summary(),
            "open_positions": list(self.open_positions.keys()),
        }


# --------------------------------------------------
# QUICK TEST / DEMO
# --------------------------------------------------

if __name__ == "__main__":
    from lib.signal_engine import generate_signal

    ex = Executor()

    # Print account info
    acct = ex.get_account()
    print("\n--- Account ---")
    for k, v in acct.items():
        print(f"  {k}: {v}")

    # Simulate a strong bull signal
    dummy_signal = {
        "composite_score": 65,
        "direction": "LONG",
        "strength": "STRONG",
        "trade_type": "0DTE_CALL",
        "confidence": 78,
        "risk_multiplier": 1.0,
        "regime_score": 40,
        "indicator_score": 15,
        "strat_score": 10,
        "reasoning": ["Test signal"],
    }

    print("\n--- Executing dummy LONG signal on SPY ---")
    result = ex.execute_signal("SPY", dummy_signal)
    print(f"  Result: {result}")

    print("\n--- Session Summary ---")
    summary = ex.session_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")
