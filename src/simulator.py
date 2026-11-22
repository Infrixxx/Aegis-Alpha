"""Vectorized and exact iterative trading simulations plus diagnostic analytics.

Phase 4 module for Aegis Alpha:
- Vectorized simulations generating sim1_gross_rev, sim1_net_rev (baseline rule)
  and sim2_gross_rev, sim2_net_rev (custom rule + sentiment/volatility filter).
- Exact iterative portfolio simulation using itertuples:
  * $10,000 starting capital with dynamic capital reinvestment
  * Max 3 concurrent open positions (max 1 per symbol)
  * Dynamic position sizing scaled by model prediction probability
  * 0.1% exchange fee on entry and exit
  * Strict 2% Stop-Loss / 4% Take-Profit with intrabar OHLC checking
- Performance benchmarking (CAGR, Sharpe Ratio, Max Drawdown, 30d Rolling Returns)
  against BTC/USDT Buy-and-Hold over the identical evaluation period.
- Deep diagnostic slicing by ticker, month, sentiment regime, and volatility regime,
  including False Positive root-cause debugging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    INITIAL_CAPITAL,
    MAX_CONCURRENT_POSITIONS,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TRADING_FEE_RATE,
)
from src.features import TARGET_COL, TARGET_HORIZON_BARS

logger = logging.getLogger("aegis_alpha.simulator")


@dataclass
class OpenPosition:
    """State of an active long spot position during exact iterative simulation."""

    symbol: str
    entry_timestamp: int
    entry_datetime: str
    entry_price: float
    quantity: float
    entry_notional: float
    entry_fee: float
    stop_loss_price: float
    take_profit_price: float
    entry_prob: float
    sentiment_regime: str
    volatility_regime: str
    bars_held: int = 0


def run_vectorized_simulations(
    df: pd.DataFrame,
    unit_investment: float = 100.0,
    fee_rate: float = TRADING_FEE_RATE,
    baseline_col: str = "pred_baseline",
    custom_col: str = "pred_custom_rule",
    return_col: str = "future_return_4h",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run vectorized simulations generating sim1_* and sim2_* revenue columns."""
    out = df.copy()
    round_trip_fee = 2.0 * fee_rate
    fwd_ret = out[return_col].fillna(0.0).astype(float)

    sig1 = out[baseline_col].fillna(0).astype(int)
    sig2 = out[custom_col].fillna(0).astype(int)

    # Vectorized Simulation 1: Baseline threshold rule
    out["sim1_gross_rev"] = sig1 * unit_investment * fwd_ret
    out["sim1_net_rev"] = sig1 * unit_investment * (fwd_ret - round_trip_fee)

    # Vectorized Simulation 2: Custom high-probability + sentiment + volatility filter
    out["sim2_gross_rev"] = sig2 * unit_investment * fwd_ret
    out["sim2_net_rev"] = sig2 * unit_investment * (fwd_ret - round_trip_fee)

    out["sim1_cum_net_rev"] = out["sim1_net_rev"].cumsum()
    out["sim2_cum_net_rev"] = out["sim2_net_rev"].cumsum()

    def _summarize(sig_col: str, gross_col: str, net_col: str) -> dict[str, Any]:
        active = out[out[sig_col] == 1]
        n_trades = int(len(active))
        if n_trades == 0:
            return {
                "signals": 0,
                "win_rate": 0.0,
                "gross_pnl_usd": 0.0,
                "net_pnl_usd": 0.0,
                "avg_net_return_pct": 0.0,
                "profit_factor": 0.0,
            }
        gross_pnl = float(active[gross_col].sum())
        net_pnl = float(active[net_col].sum())
        wins = active[active[net_col] > 0.0][net_col].sum()
        losses = abs(active[active[net_col] < 0.0][net_col].sum())
        profit_factor = float(wins / losses) if losses > 0 else float("inf")
        return {
            "signals": n_trades,
            "win_rate": round(float((active[net_col] > 0.0).mean()), 5),
            "gross_pnl_usd": round(gross_pnl, 2),
            "net_pnl_usd": round(net_pnl, 2),
            "avg_net_return_pct": round(float(net_pnl / (n_trades * unit_investment) * 100.0), 4),
            "profit_factor": round(profit_factor, 4) if np.isfinite(profit_factor) else 999.0,
        }

    summary = {
        "unit_investment_usd": unit_investment,
        "round_trip_fee_rate": round_trip_fee,
        "sim1_baseline": _summarize(baseline_col, "sim1_gross_rev", "sim1_net_rev"),
        "sim2_custom_rule": _summarize(custom_col, "sim2_gross_rev", "sim2_net_rev"),
    }
    return out, summary


def _compute_confidence_scale(prob: float) -> float:
    """Scale position allocation smoothly by model probability in [0.35, 1.00]."""
    raw = (prob - 0.50) / 0.35
    return float(np.clip(raw, 0.35, 1.00))


def run_exact_simulation(
    df: pd.DataFrame,
    signal_col: str = "pred_custom_rule",
    prob_col: str = "pred_prob_ensemble",
    initial_capital: float = INITIAL_CAPITAL,
    max_concurrent_positions: int = MAX_CONCURRENT_POSITIONS,
    fee_rate: float = TRADING_FEE_RATE,
    stop_loss_pct: float = STOP_LOSS_PCT,
    take_profit_pct: float = TAKE_PROFIT_PCT,
    max_holding_bars: int = TARGET_HORIZON_BARS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Exact event-driven iterative portfolio simulation via DataFrame.itertuples().

    Rules enforced at each chronological 15m bar:
    1. Existing positions are evaluated first for intrabar Stop-Loss (2%),
       Take-Profit (4%), or Horizon Timeout (16 bars = 4h).
       Conservative tie-breaker: if a bar touches both SL and TP, SL executes first.
    2. New BUY signals enter at bar close if open positions < max_concurrent_positions
       and the symbol does not already have an active position.
    3. Dynamic position sizing reinvests current portfolio equity, scaled by prediction
       probability and capped by available cash.
    4. 0.1% exchange fee is charged on both entry and exit notional.
    """
    required_cols = [
        "timestamp",
        "datetime_utc",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        signal_col,
        prob_col,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Exact simulation missing required columns: {missing}")

    work = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if "sentiment_regime" not in work.columns:
        work["sentiment_regime"] = "unknown"
    if "volatility_regime" not in work.columns:
        work["volatility_regime"] = "unknown"

    sim_cols = [
        "timestamp",
        "datetime_utc",
        "symbol",
        "high",
        "low",
        "close",
        signal_col,
        prob_col,
        "sentiment_regime",
        "volatility_regime",
    ]
    # Rename dynamic signal/prob columns to valid namedtuple identifiers for itertuples
    iter_df = work[sim_cols].rename(
        columns={signal_col: "sim_signal", prob_col: "sim_prob"}
    )

    cash = float(initial_capital)
    open_positions: dict[str, OpenPosition] = {}
    last_known_price: dict[str, float] = {}
    closed_trades: list[dict[str, Any]] = []
    equity_records: list[dict[str, Any]] = []
    total_fees_paid = 0.0

    current_ts: int | None = None
    current_dt_utc: str = ""
    pending_entries: list[Any] = []

    def _mark_to_market_equity() -> float:
        mtm = cash
        for sym, pos in open_positions.items():
            px = last_known_price.get(sym, pos.entry_price)
            mtm += pos.quantity * px
        return mtm

    def _close_position(
        pos: OpenPosition,
        exit_price: float,
        exit_ts: int,
        exit_dt: str,
        exit_reason: str,
    ) -> None:
        nonlocal cash, total_fees_paid
        gross_proceeds = pos.quantity * exit_price
        exit_fee = gross_proceeds * fee_rate
        net_proceeds = gross_proceeds - exit_fee
        cash += net_proceeds
        total_fees_paid += exit_fee

        gross_pnl = gross_proceeds - pos.entry_notional
        net_pnl = net_proceeds - (pos.entry_notional + pos.entry_fee)
        return_pct = net_pnl / (pos.entry_notional + pos.entry_fee)

        closed_trades.append(
            {
                "symbol": pos.symbol,
                "entry_timestamp": pos.entry_timestamp,
                "entry_datetime_utc": pos.entry_datetime,
                "exit_timestamp": exit_ts,
                "exit_datetime_utc": exit_dt,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "quantity": pos.quantity,
                "entry_notional": pos.entry_notional,
                "entry_prob": pos.entry_prob,
                "bars_held": pos.bars_held,
                "exit_reason": exit_reason,
                "gross_pnl_usd": gross_pnl,
                "fees_usd": pos.entry_fee + exit_fee,
                "net_pnl_usd": net_pnl,
                "net_return_pct": return_pct,
                "sentiment_regime": pos.sentiment_regime,
                "volatility_regime": pos.volatility_regime,
            }
        )

    def _flush_timestamp_step(ts_val: int, dt_val: str) -> None:
        nonlocal cash, total_fees_paid, pending_entries
        # Rank simultaneous signals by highest predicted probability first
        if pending_entries and len(open_positions) < max_concurrent_positions:
            pending_entries.sort(key=lambda r: r.sim_prob, reverse=True)
            for row in pending_entries:
                if len(open_positions) >= max_concurrent_positions:
                    break
                if row.symbol in open_positions:
                    continue

                portfolio_equity = _mark_to_market_equity()
                base_slot = portfolio_equity / float(max_concurrent_positions)
                scale = _compute_confidence_scale(float(row.sim_prob))
                target_spend = base_slot * scale

                # Reserve enough cash to cover entry notional + entry fee
                max_affordable_notional = cash / (1.0 + fee_rate)
                entry_notional = min(target_spend, max_affordable_notional)
                if entry_notional < 10.0:
                    continue

                entry_price = float(row.close)
                if entry_price <= 0.0:
                    continue

                entry_fee = entry_notional * fee_rate
                quantity = entry_notional / entry_price
                cash -= entry_notional + entry_fee
                total_fees_paid += entry_fee

                open_positions[row.symbol] = OpenPosition(
                    symbol=row.symbol,
                    entry_timestamp=ts_val,
                    entry_datetime=dt_val,
                    entry_price=entry_price,
                    quantity=quantity,
                    entry_notional=entry_notional,
                    entry_fee=entry_fee,
                    stop_loss_price=entry_price * (1.0 - stop_loss_pct),
                    take_profit_price=entry_price * (1.0 + take_profit_pct),
                    entry_prob=float(row.sim_prob),
                    sentiment_regime=str(row.sentiment_regime),
                    volatility_regime=str(row.volatility_regime),
                    bars_held=0,
                )

        pending_entries = []
        equity_records.append(
            {
                "timestamp": ts_val,
                "datetime_utc": dt_val,
                "cash": cash,
                "equity": _mark_to_market_equity(),
                "open_positions": len(open_positions),
            }
        )

    # Core iterative event loop using itertuples
    for row in iter_df.itertuples(index=False):
        row_ts = int(row.timestamp)
        row_dt = str(row.datetime_utc)

        if current_ts is not None and row_ts != current_ts:
            _flush_timestamp_step(current_ts, current_dt_utc)

        current_ts = row_ts
        current_dt_utc = row_dt
        last_known_price[row.symbol] = float(row.close)

        # 1. Manage existing open position for this symbol
        if row.symbol in open_positions:
            pos = open_positions[row.symbol]
            pos.bars_held += 1

            hit_sl = float(row.low) <= pos.stop_loss_price
            hit_tp = float(row.high) >= pos.take_profit_price

            if hit_sl:
                _close_position(pos, pos.stop_loss_price, row_ts, row_dt, "STOP_LOSS")
                del open_positions[row.symbol]
            elif hit_tp:
                _close_position(pos, pos.take_profit_price, row_ts, row_dt, "TAKE_PROFIT")
                del open_positions[row.symbol]
            elif pos.bars_held >= max_holding_bars:
                _close_position(pos, float(row.close), row_ts, row_dt, "HORIZON_TIMEOUT")
                del open_positions[row.symbol]

        # 2. Collect candidate BUY entries at this timestamp
        if int(row.sim_signal) == 1 and row.symbol not in open_positions:
            pending_entries.append(row)

    # Flush final timestamp and liquidate any remaining open positions at final close
    if current_ts is not None:
        _flush_timestamp_step(current_ts, current_dt_utc)
        for sym in list(open_positions.keys()):
            pos = open_positions.pop(sym)
            final_px = last_known_price.get(sym, pos.entry_price)
            _close_position(pos, final_px, current_ts, current_dt_utc, "END_OF_SIMULATION")
        if equity_records:
            equity_records[-1]["cash"] = cash
            equity_records[-1]["equity"] = cash
            equity_records[-1]["open_positions"] = 0

    trades_df = pd.DataFrame(closed_trades)
    equity_df = pd.DataFrame(equity_records)
    if not equity_df.empty:
        equity_df["datetime"] = pd.to_datetime(equity_df["datetime_utc"], utc=True)

    metrics = compute_performance_metrics(
        equity_df=equity_df,
        trades_df=trades_df,
        initial_capital=initial_capital,
        total_fees_paid=total_fees_paid,
    )
    return trades_df, equity_df, metrics


def compute_performance_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame | None = None,
    initial_capital: float = INITIAL_CAPITAL,
    total_fees_paid: float = 0.0,
) -> dict[str, Any]:
    """Compute CAGR, Sharpe Ratio, Max Drawdown, 30d Rolling Returns, and trade stats."""
    if equity_df.empty:
        return {
            "initial_capital": initial_capital,
            "final_equity": initial_capital,
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "rolling_30d_return_mean_pct": 0.0,
            "rolling_30d_return_min_pct": 0.0,
            "rolling_30d_return_max_pct": 0.0,
            "rolling_30d_positive_pct": 0.0,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "total_fees_usd": 0.0,
            "exit_reasons": {},
        }

    eq = equity_df.sort_values("datetime").copy()
    eq = eq.set_index("datetime")
    final_equity = float(eq["equity"].iloc[-1])
    total_return = (final_equity / initial_capital) - 1.0

    # Annualized CAGR using exact elapsed calendar days (24/7 crypto market)
    elapsed_days = max((eq.index[-1] - eq.index[0]).total_seconds() / 86400.0, 1.0)
    years = elapsed_days / 365.25
    if final_equity > 0.0 and years > 0.0:
        cagr = (final_equity / initial_capital) ** (1.0 / years) - 1.0
    else:
        cagr = -1.0

    # Resample to daily closes for Sharpe and 30-day rolling return statistics
    daily_eq = eq["equity"].resample("1D").last().ffill().dropna()
    daily_ret = daily_eq.pct_change().dropna()

    if len(daily_ret) > 1 and float(daily_ret.std(ddof=1)) > 0.0:
        sharpe = float((daily_ret.mean() / daily_ret.std(ddof=1)) * np.sqrt(365.0))
    else:
        sharpe = 0.0

    # Max Drawdown on full intraday equity curve
    running_max = eq["equity"].cummax()
    drawdown = (eq["equity"] / running_max) - 1.0
    max_drawdown = float(drawdown.min())

    # 30-day rolling returns (falls back to 7-day or overall return if series < 30 days)
    rolling_window = 30 if len(daily_eq) > 30 else max(min(len(daily_eq) - 1, 7), 1)
    rolling_ret = daily_eq.pct_change(rolling_window).dropna()
    if len(rolling_ret) > 0:
        roll_mean = float(rolling_ret.mean() * 100.0)
        roll_min = float(rolling_ret.min() * 100.0)
        roll_max = float(rolling_ret.max() * 100.0)
        roll_pos = float((rolling_ret > 0.0).mean() * 100.0)
    else:
        roll_mean = roll_min = roll_max = float(total_return * 100.0)
        roll_pos = 100.0 if total_return > 0.0 else 0.0

    metrics: dict[str, Any] = {
        "initial_capital": round(float(initial_capital), 2),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(float(total_return * 100.0), 3),
        "cagr_pct": round(float(cagr * 100.0), 3),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(float(max_drawdown * 100.0), 3),
        "rolling_window_days": int(rolling_window),
        "rolling_30d_return_mean_pct": round(roll_mean, 3),
        "rolling_30d_return_min_pct": round(roll_min, 3),
        "rolling_30d_return_max_pct": round(roll_max, 3),
        "rolling_30d_positive_pct": round(roll_pos, 2),
        "elapsed_days": round(elapsed_days, 2),
        "total_fees_usd": round(float(total_fees_paid), 2),
    }

    if trades_df is not None and not trades_df.empty:
        wins = trades_df[trades_df["net_pnl_usd"] > 0.0]["net_pnl_usd"].sum()
        losses = abs(trades_df[trades_df["net_pnl_usd"] < 0.0]["net_pnl_usd"].sum())
        pf = float(wins / losses) if losses > 0 else 999.0
        metrics.update(
            {
                "total_trades": int(len(trades_df)),
                "win_rate_pct": round(float((trades_df["net_pnl_usd"] > 0.0).mean() * 100.0), 2),
                "profit_factor": round(pf, 4),
                "avg_trade_pnl_usd": round(float(trades_df["net_pnl_usd"].mean()), 2),
                "exit_reasons": {
                    str(k): int(v) for k, v in trades_df["exit_reason"].value_counts().items()
                },
            }
        )
    else:
        metrics.update(
            {
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "avg_trade_pnl_usd": 0.0,
                "exit_reasons": {},
            }
        )

    return metrics


def compute_btc_benchmark(
    df: pd.DataFrame,
    benchmark_symbol: str = "BTC/USDT",
    initial_capital: float = INITIAL_CAPITAL,
    fee_rate: float = TRADING_FEE_RATE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute BTC/USDT Buy-and-Hold benchmark equity curve and metrics."""
    btc = df[df["symbol"] == benchmark_symbol].sort_values("timestamp").copy()
    if btc.empty:
        # Fallback to first available symbol if BTC/USDT is not present in a custom subset
        first_sym = str(df["symbol"].iloc[0])
        btc = df[df["symbol"] == first_sym].sort_values("timestamp").copy()
        benchmark_symbol = first_sym

    entry_price = float(btc["close"].iloc[0])
    qty = (initial_capital * (1.0 - fee_rate)) / entry_price
    btc["equity"] = qty * btc["close"].astype(float)
    btc["datetime"] = pd.to_datetime(btc["datetime_utc"], utc=True)

    bench_eq = btc[["timestamp", "datetime_utc", "datetime", "equity"]].copy()
    metrics = compute_performance_metrics(
        equity_df=bench_eq,
        trades_df=None,
        initial_capital=initial_capital,
        total_fees_paid=initial_capital * fee_rate,
    )
    metrics["benchmark_symbol"] = benchmark_symbol
    return bench_eq, metrics


def run_diagnostics(
    df_sim: pd.DataFrame,
    trades_df: pd.DataFrame,
    signal_col: str = "pred_custom_rule",
    baseline_col: str = "pred_baseline",
    net_rev_col: str = "sim2_net_rev",
) -> dict[str, Any]:
    """Slice precision and PnL by ticker, month, sentiment, and volatility; debug FPs."""
    work = df_sim.copy()
    dt = pd.to_datetime(work["datetime_utc"], utc=True)
    work["year_month"] = dt.dt.strftime("%Y-%m")

    def _slice_table(group_col: str) -> list[dict[str, Any]]:
        if group_col not in work.columns:
            return []
        rows: list[dict[str, Any]] = []
        for grp_val, grp in work.groupby(group_col, sort=True):
            preds = grp[grp[signal_col] == 1]
            n_sig = int(len(preds))
            precision = (
                float(preds[TARGET_COL].mean())
                if n_sig > 0 and TARGET_COL in preds.columns
                else 0.0
            )
            vec_net = float(preds[net_rev_col].sum()) if n_sig > 0 and net_rev_col in preds.columns else 0.0

            exact_pnl = 0.0
            exact_trades = 0
            exact_win_rate = 0.0
            if not trades_df.empty:
                if group_col == "year_month":
                    t_slice = trades_df[
                        trades_df["entry_datetime_utc"].str.slice(0, 7) == str(grp_val)
                    ]
                elif group_col in trades_df.columns:
                    t_slice = trades_df[trades_df[group_col] == grp_val]
                else:
                    t_slice = pd.DataFrame()
                if not t_slice.empty:
                    exact_trades = int(len(t_slice))
                    exact_pnl = float(t_slice["net_pnl_usd"].sum())
                    exact_win_rate = float((t_slice["net_pnl_usd"] > 0.0).mean())

            rows.append(
                {
                    group_col: str(grp_val),
                    "signals": n_sig,
                    "precision": round(precision, 4),
                    "vector_net_pnl_usd": round(vec_net, 2),
                    "exact_trades": exact_trades,
                    "exact_win_rate": round(exact_win_rate, 4),
                    "exact_net_pnl_usd": round(exact_pnl, 2),
                }
            )
        return rows

    # False Positive debugging comparing TP vs FP feature signatures
    fp_debug: dict[str, Any] = {}
    if TARGET_COL in work.columns:
        base_signals = work[work[baseline_col] == 1]
        custom_signals = work[work[signal_col] == 1]

        base_fp = int((base_signals[TARGET_COL] == 0).sum())
        custom_fp = int((custom_signals[TARGET_COL] == 0).sum())
        custom_tp = int((custom_signals[TARGET_COL] == 1).sum())

        fp_reduction_pct = (
            ((base_fp - custom_fp) / base_fp * 100.0) if base_fp > 0 else 0.0
        )

        # Analyze feature means on False Positives vs True Positives for baseline signals
        diagnostic_features = [
            c
            for c in ["rsi_14", "atr_pct_14", "vol_24h", "ema_50_200_ratio", "vwap_ratio", "fg_value"]
            if c in work.columns
        ]
        tp_rows = base_signals[base_signals[TARGET_COL] == 1]
        fp_rows = base_signals[base_signals[TARGET_COL] == 0]
        feature_deltas: dict[str, dict[str, float]] = {}
        for feat in diagnostic_features:
            tp_mean = float(tp_rows[feat].mean()) if len(tp_rows) > 0 else 0.0
            fp_mean = float(fp_rows[feat].mean()) if len(fp_rows) > 0 else 0.0
            feature_deltas[feat] = {
                "tp_mean": round(tp_mean, 5),
                "fp_mean": round(fp_mean, 5),
                "fp_minus_tp": round(fp_mean - tp_mean, 5),
            }

        fp_debug = {
            "baseline_signals": int(len(base_signals)),
            "baseline_false_positives": base_fp,
            "baseline_precision": round(float(base_signals[TARGET_COL].mean()), 4)
            if len(base_signals) > 0
            else 0.0,
            "custom_signals": int(len(custom_signals)),
            "custom_true_positives": custom_tp,
            "custom_false_positives": custom_fp,
            "custom_precision": round(float(custom_signals[TARGET_COL].mean()), 4)
            if len(custom_signals) > 0
            else 0.0,
            "false_positive_reduction_pct": round(fp_reduction_pct, 2),
            "fp_vs_tp_feature_profile": feature_deltas,
        }

    return {
        "by_ticker": _slice_table("symbol"),
        "by_month": _slice_table("year_month"),
        "by_sentiment_regime": _slice_table("sentiment_regime"),
        "by_volatility_regime": _slice_table("volatility_regime"),
        "false_positive_debugging": fp_debug,
    }


def format_simulation_report(
    vec_summary: dict[str, Any],
    exact_metrics: dict[str, Any],
    bench_metrics: dict[str, Any],
) -> str:
    """Render a clean ASCII comparison table for console logging and notebooks."""
    s1 = vec_summary["sim1_baseline"]
    s2 = vec_summary["sim2_custom_rule"]
    lines = [
        "==============================================================================",
        "AEGIS ALPHA - SIMULATION & BENCHMARK PERFORMANCE SUMMARY",
        "==============================================================================",
        "VECTORIZED SIMULATIONS ($100 unit allocation, 0.2% round-trip fees)",
        f"  Sim 1 (Baseline Rule) : signals={s1['signals']:,} | win_rate={s1['win_rate']*100:.2f}% | "
        f"gross=${s1['gross_pnl_usd']:,.2f} | net=${s1['net_pnl_usd']:,.2f} | PF={s1['profit_factor']:.2f}",
        f"  Sim 2 (Custom Rule)   : signals={s2['signals']:,} | win_rate={s2['win_rate']*100:.2f}% | "
        f"gross=${s2['gross_pnl_usd']:,.2f} | net=${s2['net_pnl_usd']:,.2f} | PF={s2['profit_factor']:.2f}",
        "------------------------------------------------------------------------------",
        f"{'Metric':<30} | {'Exact Strategy':>18} | {'BTC Buy-and-Hold':>18}",
        "------------------------------------------------------------------------------",
        f"{'Initial Capital (USD)':<30} | {exact_metrics['initial_capital']:>18,.2f} | {bench_metrics['initial_capital']:>18,.2f}",
        f"{'Final Equity (USD)':<30} | {exact_metrics['final_equity']:>18,.2f} | {bench_metrics['final_equity']:>18,.2f}",
        f"{'Total Return (%)':<30} | {exact_metrics['total_return_pct']:>17.2f}% | {bench_metrics['total_return_pct']:>17.2f}%",
        f"{'CAGR (%)':<30} | {exact_metrics['cagr_pct']:>17.2f}% | {bench_metrics['cagr_pct']:>17.2f}%",
        f"{'Sharpe Ratio (Annualized)':<30} | {exact_metrics['sharpe_ratio']:>18.3f} | {bench_metrics['sharpe_ratio']:>18.3f}",
        f"{'Max Drawdown (%)':<30} | {exact_metrics['max_drawdown_pct']:>17.2f}% | {bench_metrics['max_drawdown_pct']:>17.2f}%",
        f"{'Rolling 30d Mean Return (%)':<30} | {exact_metrics['rolling_30d_return_mean_pct']:>17.2f}% | {bench_metrics['rolling_30d_return_mean_pct']:>17.2f}%",
        f"{'Total Closed Trades':<30} | {exact_metrics['total_trades']:>18,} | {'1 (Hold)':>18}",
        f"{'Win Rate (%)':<30} | {exact_metrics['win_rate_pct']:>17.2f}% | {'N/A':>18}",
        f"{'Total Fees Paid (USD)':<30} | {exact_metrics['total_fees_usd']:>18,.2f} | {bench_metrics['total_fees_usd']:>18,.2f}",
        "==============================================================================",
    ]
    return "\n".join(lines)
