"""Telegram alerting and console fallback notification module for Aegis Alpha.

Phase 5 module:
- Sends clean, ASCII-only (zero emojis) trade alerts and portfolio metric summaries
  via the Telegram Bot HTTP API when TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set.
- Gracefully logs formatted alerts to the console if credentials are empty.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import requests

from src.broker import OrderReceipt
from src.config import REQUEST_TIMEOUT_SEC, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger("aegis_alpha.notifier")


class TelegramNotifier:
    """Notification dispatcher with Telegram Bot API support and console fallback."""

    def __init__(
        self,
        bot_token: str = TELEGRAM_BOT_TOKEN,
        chat_id: str = TELEGRAM_CHAT_ID,
        timeout: int = REQUEST_TIMEOUT_SEC,
    ) -> None:
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self.timeout = timeout
        self.enabled = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            logger.info(
                "TelegramNotifier running in CONSOLE FALLBACK mode "
                "(TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured)."
            )

    def send_message(self, text: str) -> bool:
        """Deliver a formatted text notification via Telegram or console fallback."""
        if not self.enabled:
            print("\n--- [AEGIS ALPHA NOTIFICATION - CONSOLE FALLBACK] ---")
            print(text)
            print("--- [END NOTIFICATION] ---\n")
            return True

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            logger.info("Sent Telegram notification to chat_id=%s.", self.chat_id)
            return True
        except requests.RequestException as exc:
            logger.warning(
                "Telegram delivery failed (%s); falling back to console output.", exc
            )
            print("\n--- [AEGIS ALPHA NOTIFICATION - FALLBACK AFTER ERROR] ---")
            print(text)
            print("--- [END NOTIFICATION] ---\n")
            return False

    @staticmethod
    def format_trade_alert(receipt: OrderReceipt) -> str:
        """Format a single trade order receipt in clean ASCII."""
        return "\n".join(
            [
                f"[AEGIS ALPHA TRADE ALERT | {receipt.mode}]",
                f"Order ID    : {receipt.order_id}",
                f"Symbol      : {receipt.symbol}",
                f"Action      : {receipt.side} ({receipt.status})",
                f"Entry Price : ${receipt.price:,.4f}",
                f"Quantity    : {receipt.quantity:.6f}",
                f"Notional    : ${receipt.notional_usd:,.2f} (Est. Fee: ${receipt.estimated_fee_usd:.4f})",
                f"Stop-Loss   : ${receipt.stop_loss_price:,.4f} (-2.0%)",
                f"Take-Profit : ${receipt.take_profit_price:,.4f} (+4.0%)",
                f"Model Prob  : {receipt.prediction_prob:.4f}",
                f"Timestamp   : {receipt.timestamp_utc}",
            ]
        )

    @staticmethod
    def format_portfolio_summary(
        exact_metrics: dict[str, Any],
        bench_metrics: dict[str, Any],
        run_mode: str,
    ) -> str:
        """Format portfolio simulation metrics vs. BTC Buy-and-Hold in clean ASCII."""
        return "\n".join(
            [
                f"[AEGIS ALPHA PORTFOLIO SUMMARY | MODE={run_mode.upper()}]",
                f"Strategy Final Equity : ${exact_metrics['final_equity']:,.2f} "
                f"(Total Return: {exact_metrics['total_return_pct']:+.2f}%)",
                f"Strategy CAGR / Sharpe: {exact_metrics['cagr_pct']:+.2f}% / {exact_metrics['sharpe_ratio']:.3f}",
                f"Strategy Max Drawdown : {exact_metrics['max_drawdown_pct']:.2f}%",
                f"Closed Trades / Win%  : {exact_metrics['total_trades']} / {exact_metrics['win_rate_pct']:.2f}%",
                f"BTC Buy-Hold Return   : {bench_metrics['total_return_pct']:+.2f}% "
                f"(Sharpe: {bench_metrics['sharpe_ratio']:.3f}, MaxDD: {bench_metrics['max_drawdown_pct']:.2f}%)",
            ]
        )

    def notify_trades(self, receipts: Sequence[OrderReceipt]) -> int:
        """Send alerts for all executed or simulated orders."""
        sent = 0
        for receipt in receipts:
            if self.send_message(self.format_trade_alert(receipt)):
                sent += 1
        return sent

    def notify_portfolio_summary(
        self,
        exact_metrics: dict[str, Any],
        bench_metrics: dict[str, Any],
        run_mode: str,
    ) -> bool:
        """Send formatted portfolio performance digest."""
        msg = self.format_portfolio_summary(
            exact_metrics=exact_metrics,
            bench_metrics=bench_metrics,
            run_mode=run_mode,
        )
        return self.send_message(msg)
