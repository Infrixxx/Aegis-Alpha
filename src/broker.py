"""Binance Spot Testnet and Dry-Run broker execution module for Aegis Alpha.

Phase 5 module:
- Uses ccxt.binance with Spot Testnet sandbox mode when BINANCE_API_KEY and
  BINANCE_API_SECRET are configured via environment variables.
- Gracefully falls back to simulated dry-run order execution and structured
  logging whenever credentials are absent or dry_run=True.
- Computes dynamic probability-scaled order sizes with strict 2% Stop-Loss
  and 4% Take-Profit metadata.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    BINANCE_TESTNET,
    INITIAL_CAPITAL,
    MAX_CONCURRENT_POSITIONS,
    REQUEST_TIMEOUT_SEC,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TRADING_FEE_RATE,
)

logger = logging.getLogger("aegis_alpha.broker")


@dataclass
class OrderReceipt:
    """Standardized execution receipt returned by live, testnet, or dry-run orders."""

    order_id: str
    mode: str
    symbol: str
    side: str
    status: str
    price: float
    quantity: float
    notional_usd: float
    estimated_fee_usd: float
    stop_loss_price: float
    take_profit_price: float
    prediction_prob: float
    timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BinanceBroker:
    """Spot order execution client supporting Binance Testnet and Dry-Run fallback."""

    def __init__(
        self,
        api_key: str = BINANCE_API_KEY,
        api_secret: str = BINANCE_API_SECRET,
        testnet: bool = BINANCE_TESTNET,
        dry_run: bool = False,
    ) -> None:
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.testnet = testnet
        self.dry_run = dry_run or not (self.api_key and self.api_secret)
        self.exchange: Any = None

        if self.dry_run:
            logger.info(
                "BinanceBroker initialized in DRY-RUN mode "
                "(BINANCE_API_KEY/BINANCE_API_SECRET not set or dry_run=True)."
            )
        else:
            self._init_ccxt_client()

    def _init_ccxt_client(self) -> None:
        try:
            import ccxt
        except ImportError:
            logger.warning("ccxt is not installed; falling back to DRY-RUN broker mode.")
            self.dry_run = True
            return

        self.exchange = ccxt.binance(
            {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "timeout": REQUEST_TIMEOUT_SEC * 1000,
                "options": {"defaultType": "spot"},
            }
        )
        if self.testnet:
            self.exchange.set_sandbox_mode(True)
            logger.info("BinanceBroker initialized on Binance Spot TESTNET.")
        else:
            logger.warning("BinanceBroker initialized on LIVE Binance Spot exchange.")

    @property
    def mode_label(self) -> str:
        if self.dry_run:
            return "DRY_RUN"
        return "BINANCE_TESTNET" if self.testnet else "BINANCE_LIVE"

    def get_usdt_balance(self, default_equity: float = INITIAL_CAPITAL) -> float:
        """Return available USDT balance (or simulated capital in dry-run mode)."""
        if self.dry_run or self.exchange is None:
            return float(default_equity)
        try:
            balance = self.exchange.fetch_balance()
            free_usdt = float(balance.get("USDT", {}).get("free", 0.0))
            return free_usdt
        except Exception as exc:
            logger.warning("Failed to query exchange balance (%s); using default equity.", exc)
            return float(default_equity)

    def place_market_buy(
        self,
        symbol: str,
        notional_usd: float,
        reference_price: float,
        prediction_prob: float = 0.65,
        stop_loss_pct: float = STOP_LOSS_PCT,
        take_profit_pct: float = TAKE_PROFIT_PCT,
    ) -> OrderReceipt:
        """Place a market BUY order (or log a simulated order in dry-run mode)."""
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if reference_price <= 0.0 or notional_usd <= 0.0:
            raise ValueError("reference_price and notional_usd must be strictly positive.")

        stop_loss_price = reference_price * (1.0 - stop_loss_pct)
        take_profit_price = reference_price * (1.0 + take_profit_pct)

        if self.dry_run or self.exchange is None:
            qty = notional_usd / reference_price
            receipt = OrderReceipt(
                order_id=f"SIM-{uuid.uuid4().hex[:10].upper()}",
                mode=self.mode_label,
                symbol=symbol,
                side="BUY",
                status="SIMULATED_FILLED",
                price=round(reference_price, 6),
                quantity=round(qty, 8),
                notional_usd=round(notional_usd, 2),
                estimated_fee_usd=round(notional_usd * TRADING_FEE_RATE, 4),
                stop_loss_price=round(stop_loss_price, 6),
                take_profit_price=round(take_profit_price, 6),
                prediction_prob=round(prediction_prob, 5),
                timestamp_utc=now_utc,
            )
            logger.info(
                "[%s] BUY %s | qty=%.6f @ $%.4f | notional=$%.2f | SL=$%.4f | TP=$%.4f | prob=%.4f",
                receipt.mode,
                receipt.symbol,
                receipt.quantity,
                receipt.price,
                receipt.notional_usd,
                receipt.stop_loss_price,
                receipt.take_profit_price,
                receipt.prediction_prob,
            )
            return receipt

        # Testnet / Live execution via ccxt
        ticker = self.exchange.fetch_ticker(symbol)
        exec_price = float(ticker.get("last") or reference_price)
        raw_qty = notional_usd / exec_price
        precision_qty = float(self.exchange.amount_to_precision(symbol, raw_qty))
        order = self.exchange.create_market_buy_order(symbol, precision_qty)

        filled_price = float(order.get("average") or order.get("price") or exec_price)
        filled_qty = float(order.get("filled") or precision_qty)
        filled_notional = filled_price * filled_qty
        receipt = OrderReceipt(
            order_id=str(order.get("id", "UNKNOWN")),
            mode=self.mode_label,
            symbol=symbol,
            side="BUY",
            status=str(order.get("status", "SUBMITTED")).upper(),
            price=round(filled_price, 6),
            quantity=round(filled_qty, 8),
            notional_usd=round(filled_notional, 2),
            estimated_fee_usd=round(filled_notional * TRADING_FEE_RATE, 4),
            stop_loss_price=round(filled_price * (1.0 - stop_loss_pct), 6),
            take_profit_price=round(filled_price * (1.0 + take_profit_pct), 6),
            prediction_prob=round(prediction_prob, 5),
            timestamp_utc=now_utc,
        )
        logger.info(
            "[%s] Submitted order %s: BUY %s qty=%.6f @ $%.4f",
            receipt.mode,
            receipt.order_id,
            receipt.symbol,
            receipt.quantity,
            receipt.price,
        )
        return receipt

    def execute_latest_signals(
        self,
        predictions_df: pd.DataFrame,
        signal_col: str = "pred_custom_rule",
        prob_col: str = "pred_prob_ensemble",
        max_positions: int = MAX_CONCURRENT_POSITIONS,
        fallback_equity: float = INITIAL_CAPITAL,
    ) -> list[OrderReceipt]:
        """Inspect the latest bar per symbol and execute BUY orders for active signals."""
        if predictions_df.empty:
            return []

        latest_per_symbol = (
            predictions_df.sort_values("timestamp")
            .groupby("symbol", as_index=False)
            .tail(1)
            .sort_values(prob_col, ascending=False)
        )
        actionable = latest_per_symbol[latest_per_symbol[signal_col] == 1].head(max_positions)
        if actionable.empty:
            logger.info("No actionable BUY signals on the latest bar across monitored symbols.")
            return []

        available_usdt = self.get_usdt_balance(default_equity=fallback_equity)
        base_slot = available_usdt / float(max_positions)
        receipts: list[OrderReceipt] = []

        for row in actionable.itertuples(index=False):
            prob = float(getattr(row, prob_col))
            scale = float(np.clip((prob - 0.50) / 0.35, 0.35, 1.0))
            notional = min(base_slot * scale, available_usdt / (1.0 + TRADING_FEE_RATE))
            if notional < 10.0:
                continue
            receipt = self.place_market_buy(
                symbol=str(row.symbol),
                notional_usd=notional,
                reference_price=float(row.close),
                prediction_prob=prob,
            )
            receipts.append(receipt)
            available_usdt -= notional * (1.0 + TRADING_FEE_RATE)

        return receipts
