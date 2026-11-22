"""End-to-end CLI orchestrator for the Aegis Alpha algorithmic trading pipeline.

Supports two execution regimes:
- --mode online  : Incrementally fetches live OHLCV (ccxt/Binance) and Crypto Fear &
                   Greed Index into SQLite -> engineers 40 numeric + 57 dummy features
                   -> loads or trains multi-model ensemble -> runs Vector + Exact
                   simulations + diagnostics -> executes dry-run/testnet broker orders
                   -> dispatches Telegram/console notifications.
- --mode offline : Loads pre-saved local SQLite DB or Parquet/CSV snapshot (with
                   optional HTTP/Google Drive download via --offline-url) for fast,
                   deterministic peer-review reproduction without waiting for full
                   API backfills.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.broker import BinanceBroker
from src.config import (
    DATA_DIR,
    DB_PATH,
    MODELS_DIR,
    START_DATE_ISO,
    SYMBOLS,
    configure_logging,
    ensure_directories,
)
from src.data_fetcher import format_database_summary, run_ingestion
from src.features import (
    build_features_from_sqlite,
    load_dataframe,
    save_dataframe,
)
from src.notifier import TelegramNotifier
from src.simulator import (
    compute_btc_benchmark,
    format_simulation_report,
    run_diagnostics,
    run_exact_simulation,
    run_vectorized_simulations,
)
from src.train import (
    chronological_split,
    predict_with_saved_models,
    train_all_models,
)

logger = logging.getLogger("aegis_alpha.main")

DEFAULT_OFFLINE_SNAPSHOT = DATA_DIR / "offline_snapshot.parquet"


def _download_offline_artifact(url: str, destination: Path) -> Path:
    """Download a pre-saved offline dataset (e.g., from Google Drive direct link)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading offline dataset from %s -> %s", url, destination)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(destination, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    logger.info("Offline dataset downloaded (%d bytes).", destination.stat().st_size)
    return destination


def load_or_build_features(
    mode: str,
    db_path: Path,
    offline_path: Path,
    offline_url: str | None = None,
    symbols: list[str] | None = None,
    start_date: str = START_DATE_ISO,
    export_offline_snapshot: bool = False,
) -> pd.DataFrame:
    """Resolve feature DataFrame according to online or offline execution mode."""
    ensure_directories()

    if mode == "online":
        logger.info("Running ONLINE mode: incremental API sync into %s.", db_path)
        ingestion_summary = run_ingestion(
            db_path=db_path,
            symbols=symbols or list(SYMBOLS),
            start_iso=start_date,
        )
        print(format_database_summary(ingestion_summary))
        df_features = build_features_from_sqlite(db_path=db_path, symbols=symbols)
        if export_offline_snapshot:
            saved_path = save_dataframe(df_features, offline_path)
            logger.info("Exported offline feature snapshot to %s.", saved_path)
        return df_features

    # OFFLINE mode
    logger.info("Running OFFLINE mode for deterministic peer-review reproduction.")
    if offline_url and not offline_path.exists():
        _download_offline_artifact(offline_url, offline_path)

    if offline_path.exists():
        logger.info("Loading pre-computed offline snapshot from %s.", offline_path)
        return load_dataframe(offline_path)

    csv_fallback = offline_path.with_suffix(".csv")
    if csv_fallback.exists():
        logger.info("Loading pre-computed offline CSV snapshot from %s.", csv_fallback)
        return load_dataframe(csv_fallback)

    if db_path.exists():
        logger.info(
            "Offline snapshot %s not found; building features locally from SQLite DB %s.",
            offline_path,
            db_path,
        )
        df_features = build_features_from_sqlite(db_path=db_path, symbols=symbols)
        if export_offline_snapshot:
            save_dataframe(df_features, offline_path)
        return df_features

    raise FileNotFoundError(
        f"Offline mode could not find {offline_path} or {db_path}. "
        "Provide --offline-path / --offline-url or run '--mode online' once to populate SQLite."
    )


def run_pipeline(
    mode: str = "offline",
    db_path: Path = DB_PATH,
    offline_path: Path = DEFAULT_OFFLINE_SNAPSHOT,
    offline_url: str | None = None,
    models_dir: Path = MODELS_DIR,
    retrain: bool = False,
    skip_lstm: bool = False,
    tuning_method: str = "optuna",
    tuning_trials: int = 12,
    lstm_epochs: int = 6,
    execute_orders: bool = False,
    notify: bool = True,
    export_offline_snapshot: bool = False,
    symbols: list[str] | None = None,
    start_date: str = START_DATE_ISO,
) -> dict[str, Any]:
    """Execute the full Aegis Alpha workflow from ingestion through notifications."""
    df_features = load_or_build_features(
        mode=mode,
        db_path=db_path,
        offline_path=offline_path,
        offline_url=offline_url,
        symbols=symbols,
        start_date=start_date,
        export_offline_snapshot=export_offline_snapshot,
    )

    xgb_artifact = Path(models_dir) / "xgboost_tuned.joblib"
    if retrain or not xgb_artifact.exists():
        logger.info("Training multi-model suite (DT, RF, Tuned XGBoost, LSTM=%s).", not skip_lstm)
        train_result = train_all_models(
            df=df_features,
            models_dir=models_dir,
            tuning_method=tuning_method,
            tuning_trials=tuning_trials,
            train_lstm=not skip_lstm,
            lstm_epochs=lstm_epochs,
        )
        eval_df = train_result["test_predictions"]
        model_metrics = train_result["metrics"]
    else:
        logger.info("Loading persisted model artifacts from %s for inference.", models_dir)
        split = chronological_split(df_features)
        eval_df = predict_with_saved_models(split.test, models_dir=models_dir)
        meta_path = Path(models_dir) / "model_metadata.json"
        model_metrics = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else {}
        )

    # Phase 4: Vectorized + Exact Iterative Simulations + Benchmark + Diagnostics
    df_vec, vec_summary = run_vectorized_simulations(eval_df)
    trades_df, equity_df, exact_metrics = run_exact_simulation(df_vec)
    _, bench_metrics = compute_btc_benchmark(df_vec)
    diagnostics = run_diagnostics(df_vec, trades_df)

    report_text = format_simulation_report(vec_summary, exact_metrics, bench_metrics)
    print(report_text)

    # Persist simulation and diagnostics report to models/
    sim_report_path = Path(models_dir) / "simulation_report.json"
    sim_payload = {
        "mode": mode,
        "vectorized_summary": vec_summary,
        "exact_metrics": exact_metrics,
        "benchmark_metrics": bench_metrics,
        "diagnostics": diagnostics,
    }
    sim_report_path.write_text(json.dumps(sim_payload, indent=2), encoding="utf-8")
    logger.info("Saved simulation and diagnostic report to %s.", sim_report_path)

    # Phase 5: Broker Execution (Dry-Run / Testnet) and Telegram / Console Notifications
    broker = BinanceBroker(dry_run=not execute_orders)
    order_receipts = broker.execute_latest_signals(df_vec)

    if notify:
        notifier = TelegramNotifier()
        notifier.notify_portfolio_summary(
            exact_metrics=exact_metrics,
            bench_metrics=bench_metrics,
            run_mode=mode,
        )
        if order_receipts:
            notifier.notify_trades(order_receipts)

    return {
        "mode": mode,
        "feature_rows": int(len(df_features)),
        "eval_rows": int(len(df_vec)),
        "model_metrics": model_metrics,
        "vectorized_summary": vec_summary,
        "exact_metrics": exact_metrics,
        "benchmark_metrics": bench_metrics,
        "diagnostics": diagnostics,
        "orders": [r.to_dict() for r in order_receipts],
        "trades_df": trades_df,
        "equity_df": equity_df,
        "predictions_df": df_vec,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aegis Alpha end-to-end algorithmic trading pipeline."
    )
    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        default="offline",
        help="Execution regime: 'online' (live API sync) or 'offline' (local DB/Parquet).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help=f"SQLite database path (default: {DB_PATH}).",
    )
    parser.add_argument(
        "--offline-path",
        type=Path,
        default=DEFAULT_OFFLINE_SNAPSHOT,
        help=f"Offline Parquet/CSV snapshot path (default: {DEFAULT_OFFLINE_SNAPSHOT}).",
    )
    parser.add_argument(
        "--offline-url",
        type=str,
        default=None,
        help="Optional direct download URL (e.g. Google Drive) for the offline snapshot.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"Directory for saved model artifacts (default: {MODELS_DIR}).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Force retraining of all models even if artifacts already exist.",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip PyTorch LSTM training (useful on low-RAM CPU environments).",
    )
    parser.add_argument(
        "--tuning-method",
        choices=["optuna", "random_search"],
        default="optuna",
        help="Hyperparameter tuning strategy for XGBClassifier.",
    )
    parser.add_argument(
        "--tuning-trials",
        type=int,
        default=12,
        help="Number of Optuna trials or RandomizedSearchCV iterations (default: 12).",
    )
    parser.add_argument(
        "--lstm-epochs",
        type=int,
        default=6,
        help="Number of training epochs for the PyTorch LSTM classifier (default: 6).",
    )
    parser.add_argument(
        "--execute-orders",
        action="store_true",
        help="Allow Binance Spot Testnet/Live order submission when API keys are set "
        "(otherwise always runs in DRY-RUN mode).",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Disable Telegram/console alert dispatch.",
    )
    parser.add_argument(
        "--export-offline-snapshot",
        action="store_true",
        help="Export engineered feature DataFrame to --offline-path for peer review.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Optional subset of trading pairs (default: all 5 configured pairs).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=START_DATE_ISO,
        help=f"Backfill start ISO date for --mode online (default: {START_DATE_ISO}).",
    )
    return parser


def main() -> None:
    """CLI entrypoint for Aegis Alpha."""
    configure_logging()
    args = _build_parser().parse_args()
    run_pipeline(
        mode=args.mode,
        db_path=args.db_path,
        offline_path=args.offline_path,
        offline_url=args.offline_url,
        models_dir=args.models_dir,
        retrain=args.train,
        skip_lstm=args.skip_lstm,
        tuning_method=args.tuning_method,
        tuning_trials=args.tuning_trials,
        lstm_epochs=args.lstm_epochs,
        execute_orders=args.execute_orders,
        notify=not args.no_notify,
        export_offline_snapshot=args.export_offline_snapshot,
        symbols=args.symbols,
        start_date=args.start_date,
    )


if __name__ == "__main__":
    main()
