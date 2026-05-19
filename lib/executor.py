# ================================================
# EXECUTOR.PY - Alpaca Order Execution Engine
# Paper trading via Alpaca API
# Handles equity (swing) and options (0DTE) orders
# ================================================

import time
from datetime import datetime, date
from typing import Optional

from config import (
    get_alpaca_keys,
    ALPACA_PAPER_TRADING, ALPACA_BASE_URL, ALPACA_DATA_URL,
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

        _key, _sec = get_alpaca_keys()   # stripped, validated
        self.client = TradingClient(
            api_key=_key,
            secret_key=_sec,
            paper=self.paper,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=_key,
            secret_key=_sec,
            url_override=ALPACA_DATA_URL,
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

    def kill_switch_0dte(self, swing_tickers: list = None) -> int:
        """
        EOD liquidation: close all 0DTE (options) positions, skip swing equity.
        Pre-checks live positions first — safe to call on an already-flat account
        (returns 0 without touching the broker API).
        Returns the number of positions closed.
        """
        from lib.kill_switch import close_0dte_positions

        # Pre-flight: if account is already flat, skip entirely
        try:
            live_positions = self.get_positions()
        except Exception as _e:
            print(f"[KillSwitch] Could not fetch positions: {_e}")
            return 0

        if not live_positions:
            print("[KillSwitch] Account already flat — no action needed.")
            return 0

        result = close_0dte_positions(self, swing_tickers=swing_tickers or [])
        n = len(result['closed'])
        if result['skipped']:
            print(f"[KillSwitch] Swing positions preserved: {result['skipped']}")
        if result['errors']:
            print(f"[KillSwitch] Errors during close: {result['errors']}")
        print(f"[KillSwitch] EOD complete — {n} closed, "
              f"{len(result['skipped'])} swing preserved, {len(result['errors'])} errors.")
        return n

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


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE OPTIONS HELPERS  — pure math, no API client required
# ══════════════════════════════════════════════════════════════════════════════

import math
from datetime import date as _date, timedelta as _td

_INTRADAY_RISK = 150   # PRD: max $ loss per intraday trade
_SWING_RISK    = 150   # PRD: max $ loss per swing trade


def get_expiration_date(trade_type: str) -> _date:
    """
    Intraday → next Friday from execution date (today if today IS Friday).
    Swing    → first Friday in 14-45 day window from execution date.
               Returns the Friday closest to 30 days out as default.
    """
    today        = _date.today()
    days_to_fri  = (4 - today.weekday()) % 7   # 0 when today IS Friday
    nearest_fri  = today + _td(days=days_to_fri)
    if trade_type == 'intraday':
        return nearest_fri
    # Swing: collect all Fridays between 14 and 45 days out, pick closest to 30
    swing_fridays = []
    candidate = nearest_fri
    while (candidate - today).days < 14:
        candidate += _td(weeks=1)
    while (candidate - today).days <= 45:
        swing_fridays.append(candidate)
        candidate += _td(weeks=1)
    if not swing_fridays:
        # Fallback: first Friday at least 14 days out
        candidate = nearest_fri
        while (candidate - today).days < 14:
            candidate += _td(weeks=1)
        return candidate
    # Pick Friday closest to 30 days
    target_days = 30
    return min(swing_fridays, key=lambda d: abs((d - today).days - target_days))


def get_swing_expiry_range() -> list:
    """
    Return all Fridays between 14 and 45 days from today.
    Used by build_trade_setup to search for the best swing contract.
    """
    today    = _date.today()
    days_fri = (4 - today.weekday()) % 7
    candidate = today + _td(days=days_fri)
    while (candidate - today).days < 14:
        candidate += _td(weeks=1)
    fridays = []
    while (candidate - today).days <= 45:
        fridays.append(candidate)
        candidate += _td(weeks=1)
    return fridays


def _strike_increment(price: float) -> float:
    if price < 25:   return 0.50
    if price < 100:  return 1.00
    if price < 200:  return 2.50
    if price < 500:  return 5.00
    return 10.0


def get_best_strike(spot_price: float, direction: str) -> float:
    """
    First ITM strike using standard strike increments.
    CALL → first strike below spot (ITM for long call).
    PUT  → first strike above spot (ITM for long put).
    """
    inc = _strike_increment(spot_price)
    if direction.upper() in ('CALL', 'LONG', 'UP'):
        return math.floor(spot_price / inc) * inc
    return math.ceil(spot_price / inc) * inc


def estimate_option_premium(spot_price: float, strike: float, dte: int, direction: str) -> float:
    """
    Premium ≈ intrinsic value + time value.
    Time value = spot × 1% × √(DTE/252) — rough but consistent.
    """
    if direction.upper() in ('CALL', 'LONG', 'UP'):
        intrinsic = max(0.0, spot_price - strike)
    else:
        intrinsic = max(0.0, strike - spot_price)
    time_val = spot_price * 0.01 * math.sqrt(max(1, dte) / 252)
    return max(0.05, round(intrinsic + time_val, 2))


def calculate_option_contracts(risk_amount: float, option_premium: float) -> int:
    """contracts = floor(risk_amount / (premium × 100)), minimum 1."""
    if option_premium <= 0:
        return 1
    return max(1, int(risk_amount / (option_premium * 100)))


# ── Live option chain fetch ───────────────────────────────────────────────────

def fetch_itm_contract(
    ticker:     str,
    direction:  str,    # 'CALL' or 'PUT'
    expiry:     _date,
    spot_price: float,
) -> Optional[dict]:
    """
    Fetch the nearest ITM contract from Alpaca's live option chain.
    Returns {symbol, strike, ask, bid, mid, source='live'} or None.

    CALL → highest strike strictly below spot (≤3 increments away).
    PUT  → lowest strike strictly above spot  (≤3 increments away).
    Falls back gracefully — callers always use math estimate on None.
    """
    try:
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import ContractType as CT

        key, sec  = get_alpaca_keys()
        tc        = TradingClient(api_key=key, secret_key=sec, paper=ALPACA_PAPER_TRADING)
        is_call   = direction.upper() == 'CALL'
        inc       = _strike_increment(spot_price)
        scan_width = inc * 4          # search ±4 strikes from spot

        if is_call:
            lower = round(spot_price - scan_width, 2)
            upper = round(spot_price, 2)
        else:
            lower = round(spot_price, 2)
            upper = round(spot_price + scan_width, 2)

        req  = GetOptionContractsRequest(
            underlying_symbols=[ticker.upper()],
            expiration_date=expiry.isoformat(),
            type=CT.CALL if is_call else CT.PUT,
            strike_price_gte=str(lower),
            strike_price_lte=str(upper),
            limit=10,
        )
        resp      = tc.get_option_contracts(req)
        contracts = getattr(resp, 'option_contracts', None) or []
        if not contracts:
            return None

        # Closest ITM: highest strike for calls, lowest for puts
        sorted_c = sorted(contracts, key=lambda c: float(c.strike_price), reverse=is_call)
        best     = sorted_c[0]

        best_strike = float(best.strike_price)

        # Live quote for the selected contract
        ask = bid = mid = None
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            from alpaca.data.requests import OptionSnapshotRequest
            oc    = OptionHistoricalDataClient(api_key=key, secret_key=sec)
            snaps = oc.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=best.symbol))
            snap  = (snaps or {}).get(best.symbol)
            if snap and snap.latest_quote:
                ask = float(snap.latest_quote.ask_price or 0) or None
                bid = float(snap.latest_quote.bid_price or 0) or None
                mid = round((ask + bid) / 2, 2) if ask and bid else (ask or bid)
        except Exception:
            pass

        # Sanity-check: reject stale/invalid quotes from Alpaca.
        # Compare against math estimate; if ask is > 3× what theory predicts,
        # the quote is from an old fill that no longer reflects the market.
        if ask is not None:
            _exp_dte  = max(0, (expiry - _date.today()).days)
            _math_est = estimate_option_premium(spot_price, best_strike, _exp_dte,
                                                'CALL' if is_call else 'PUT')
            _intrinsic = max(0.0, spot_price - best_strike) if is_call else max(0.0, best_strike - spot_price)
            _max_sane  = spot_price if is_call else best_strike
            if ask > _max_sane or ask > _math_est * 3 or ask < _intrinsic * 0.5:
                ask = bid = mid = None   # reject stale quote — math fallback used

        return {
            'symbol': best.symbol,
            'strike': best_strike,
            'expiry': str(best.expiration_date),
            'ask':    ask,
            'bid':    bid,
            'mid':    mid,
            'source': 'live',
        }

    except Exception as _e:
        fetch_itm_contract._last_error = str(_e)
        return None


def build_trade_setup(
    ticker:          str,
    direction:       str,
    trade_type:      str,           # 'intraday' | 'swing'
    spot_price:      float,
    gap_type:        str   = 'No Gap',
    risk_multiplier: float = 1.0,   # 0.5 in Volatile regimes, 1.0 in Quiet
) -> dict:
    """
    Full options trade recommendation with Savage Runner Protocol.

    Pipeline
    --------
    1. Try live Alpaca option chain → ITM contract + real ask price.
    2. Fall back to math estimate (intrinsic + time value).
    3. Budget check: if 1 contract > 2× risk budget → return error dict.
    4. Savage Runner split: scale_qty (60%) + runner_qty (40%).

    Returns dict with all fields needed by the UI and executor.
    """
    is_intraday   = trade_type.lower() == 'intraday'
    contract_type = 'CALL' if direction.upper() in ('LONG', 'UP', 'CALL') else 'PUT'
    _base_risk    = _INTRADAY_RISK if is_intraday else _SWING_RISK
    risk_amount   = max(10, round(_base_risk * max(0.1, min(1.0, risk_multiplier))))

    expiry = get_expiration_date('intraday' if is_intraday else 'swing')
    dte    = max(0, (expiry - _date.today()).days)

    # ── 1. Live chain (best-effort) ───────────────────────────────────────────
    live   = None
    source = 'estimate'

    if is_intraday:
        # 0DTE: single expiry — first Friday from execution date
        live = fetch_itm_contract(ticker, contract_type, expiry, spot_price)
        if live and live.get('ask'):
            source = 'live'
    else:
        # Swing: search 14-45 day window for best live contract
        swing_dates = get_swing_expiry_range()
        for _sw_date in swing_dates:
            _candidate = fetch_itm_contract(ticker, contract_type, _sw_date, spot_price)
            if _candidate and _candidate.get('ask'):
                live   = _candidate
                expiry = _sw_date
                dte    = max(0, (expiry - _date.today()).days)
                source = 'live'
                break
        if live is None:
            # No live contract found → use default expiry (closest to 30 days)
            expiry = get_expiration_date('swing')
            dte    = max(0, (expiry - _date.today()).days)

    if live and live.get('ask'):
        strike     = live['strike']
        premium    = live['ask']        # use ask for conservative sizing
        mid        = live.get('mid', premium)
        occ_symbol = live['symbol']
    else:
        # ── 2. Math fallback ─────────────────────────────────────────────────
        source     = 'estimate'
        strike     = get_best_strike(spot_price, contract_type)
        premium    = estimate_option_premium(spot_price, strike, dte, contract_type)
        mid        = premium
        cp         = 'C' if contract_type == 'CALL' else 'P'
        occ_symbol = f"{ticker.upper()}{expiry.strftime('%y%m%d')}{cp}{int(strike * 1000):08d}"

    cost_per_contract = round(premium * 100, 2)

    # ── 3. Budget check ───────────────────────────────────────────────────────
    if cost_per_contract > risk_amount * 2:
        exp_display = f"{expiry.strftime('%b')} {expiry.day}"
        return {
            'error':             'Insufficient Risk Budget for ITM Contract',
            'cost_per_contract': cost_per_contract,
            'risk_amount':       risk_amount,
            'ticker':            ticker.upper(),
            'direction':         direction,
            'trade_type':        trade_type,
            'occ_symbol':        occ_symbol,
            'strike':            strike,
            'exp_display':       exp_display,
            'source':            source,
        }

    contracts    = max(1, int(risk_amount / cost_per_contract))
    budget_warn  = (
        f"Min contract (${cost_per_contract:.0f}) exceeds ${risk_amount} budget — sizing to 1"
        if cost_per_contract > risk_amount else None
    )

    # ── 4. Savage Runner split (only when > 1 contract) ──────────────────────
    if contracts > 1:
        scale_qty  = math.ceil(contracts * 0.60)   # ~60% scale-out at T1
        runner_qty = contracts - scale_qty
    else:
        scale_qty  = 1
        runner_qty = 0

    runner_active = runner_qty > 0
    total_cost    = round(contracts * premium * 100, 2)

    if gap_type in ('Full Up', 'Full Down'):
        strategy = 'Gap and Go'
    elif gap_type in ('Partial Up', 'Partial Down'):
        strategy = 'Gap Fill / Continuation'
    else:
        strategy = 'Trend Continuation'

    exp_display = f"{expiry.strftime('%b')} {expiry.day}"
    size_str    = (
        f"{contracts} (Scale: {scale_qty} / Runner: {runner_qty})"
        if runner_active else str(contracts)
    )

    return {
        'strategy':      strategy,
        'contract_type': contract_type,
        'strike':        strike,
        'exp_display':   exp_display,
        'expiration':    expiry.isoformat(),
        'dte':           dte,
        'premium_ask':   premium,
        'premium_mid':   mid,
        'premium_est':   premium,
        'contracts':     contracts,
        'scale_qty':     scale_qty,
        'runner_qty':    runner_qty,
        'runner_active': runner_active,
        'risk_amount':   risk_amount,
        'total_cost':    total_cost,
        'budget_warning': budget_warn,
        'occ_symbol':    occ_symbol,
        'source':        source,
        'action_str':    f"BUY {int(strike)} {exp_display} {contract_type}",
        'size_str':      size_str,
        'trade_type':    'intraday' if is_intraday else 'swing',
        'ticker':        ticker.upper(),
        'spot_price':    spot_price,
    }


def execute_paper_order(trade_setup: dict, context: dict = None) -> dict:
    """
    One-shot paper execution from a build_trade_setup() output dict.
    Uses the pre-computed OCC symbol — no second chain lookup needed.

    Returns dict: {status, order_id, occ_symbol, contracts, cost, error?}
    """
    occ    = trade_setup.get('occ_symbol', '')
    qty    = trade_setup.get('contracts', 1)
    ticker = trade_setup.get('ticker', '')
    ct     = trade_setup.get('contract_type', 'CALL')

    result = {
        'timestamp':  datetime.now().isoformat(),
        'occ_symbol': occ,
        'ticker':     ticker,
        'contracts':  qty,
        'cost':       trade_setup.get('total_cost', 0),
        'trade_type': trade_setup.get('trade_type', ''),
        'direction':  'LONG' if ct == 'CALL' else 'SHORT',
        'paper':      True,
        'status':     'PENDING',
    }

    try:
        key, sec = get_alpaca_keys()
        if not key or not sec:
            result['status']   = 'DRY_RUN'
            result['order_id'] = f"DRY-{ticker}-{int(time.time())}"
            return result

        # Always paper=True for this function — it is intentionally paper-only
        tc = TradingClient(api_key=key, secret_key=sec, paper=True)
        req = MarketOrderRequest(
            symbol=occ,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = tc.submit_order(req)
        result['status']   = str(order.status)
        result['order_id'] = str(order.id)

    except Exception as exc:
        result['status'] = 'ERROR'
        result['error']  = str(exc)

    # Persist to Supabase if context provided and no error
    if result['status'] not in ('ERROR', 'DRY_RUN', 'PENDING'):
        try:
            log_trade_to_supabase(trade_setup, context or {})
        except Exception:
            pass

    return result


# ── Aliases so app.py can import these names ──────────────────────────────────
AlpacaExecutor = Executor


# ══════════════════════════════════════════════════════════════════════════════
# TRADE LOGGER — Supabase persistence layer
# ══════════════════════════════════════════════════════════════════════════════

def log_trade_to_supabase(
    trade_setup: dict,
    context: dict = None,
) -> Optional[int]:
    """
    Persist an entry-side options trade to the trade_log table.

    Parameters
    ----------
    trade_setup : dict
        Output of build_trade_setup().
    context : dict
        Regime + FTFC + scanner context at the moment of execution:
          regime, regime_confidence, ftfc_aligned, ftfc_total,
          gap_type, alpha_setup, sector_etf, sentinel_bonus

    Returns the new row id on success, None on failure.
    """
    from datetime import timezone as _tz
    context = context or {}

    row = {
        'ticker':             trade_setup.get('ticker', '').upper(),
        'occ_symbol':         trade_setup.get('occ_symbol'),
        'contract_type':      trade_setup.get('contract_type'),
        'strike':             trade_setup.get('strike'),
        'expiration':         trade_setup.get('expiration'),
        'dte':                trade_setup.get('dte'),
        'entry_premium':      trade_setup.get('premium_ask'),
        'spot_price':         trade_setup.get('spot_price'),
        'contracts':          trade_setup.get('contracts'),
        'total_cost':         trade_setup.get('total_cost'),
        'trade_type':         trade_setup.get('trade_type'),
        'direction':          'LONG' if trade_setup.get('contract_type') == 'CALL' else 'SHORT',
        'strategy':           trade_setup.get('strategy'),
        'source':             trade_setup.get('source'),
        # Brain context
        'regime':             context.get('regime', ''),
        'regime_confidence':  context.get('regime_confidence'),
        'ftfc_aligned':       context.get('ftfc_aligned'),
        'ftfc_total':         context.get('ftfc_total'),
        'gap_type':           context.get('gap_type', 'No Gap'),
        'alpha_setup':        bool(context.get('alpha_setup', False)),
        'sector_etf':         context.get('sector_etf', ''),
        'sentinel_bonus':     int(context.get('sentinel_bonus', 0)),
        # Runner split
        'scale_qty':          trade_setup.get('scale_qty'),
        'runner_qty':         trade_setup.get('runner_qty'),
        'runner_active':      bool(trade_setup.get('runner_active', False)),
        # Status
        'status':             'open',
        'paper':              True,
        'entered_at':         datetime.now(_tz.utc).isoformat(),
    }
    # Drop None values — Supabase doesn't mind, but keeps rows clean
    row = {k: v for k, v in row.items() if v is not None}

    try:
        from config import supabase
        res = supabase.table('trade_log').insert(row).execute()
        return res.data[0]['id'] if res.data else None
    except Exception as exc:
        print(f'[TradeLog] insert failed: {exc}')
        return None