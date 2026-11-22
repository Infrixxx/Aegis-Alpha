"""Centralized configuration for the Aegis Alpha pipeline.

Loads secrets and runtime overrides from environment variables via python-dotenv
without hardcoding credentials.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Project directories
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"

# Load environment variables from project root .env if present
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


def _get_bool_env(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable safely."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Credentials (loaded exclusively from environment variables)
TELEGRAM_BOT_TOKEN: Final[str] = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID: Final[str] = os.getenv("TELEGRAM_CHAT_ID", "").strip()
BINANCE_API_KEY: Final[str] = os.getenv("BINANCE_API_KEY", "").strip()
BINANCE_API_SECRET: Final[str] = os.getenv("BINANCE_API_SECRET", "").strip()
BINANCE_TESTNET: Final[bool] = _get_bool_env("BINANCE_TESTNET", default=True)

# Storage paths
_db_env_path = os.getenv("DATABASE_PATH", "data/market_data.db").strip()
DB_PATH: Final[Path] = (
    Path(_db_env_path)
    if Path(_db_env_path).is_absolute()
    else PROJECT_ROOT / _db_env_path
)
LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO").strip().upper()

# Phase 1: Market data and alternative data ingestion constants
SYMBOLS: Final[list[str]] = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "ADA/USDT",
]
TIMEFRAME: Final[str] = "15m"
TIMEFRAME_SECONDS: Final[int] = 15 * 60
TIMEFRAME_MS: Final[int] = TIMEFRAME_SECONDS * 1000
START_DATE_ISO: Final[str] = "2020-01-01T00:00:00Z"
OHLCV_BATCH_LIMIT: Final[int] = 1000

FEAR_GREED_API_URL: Final[str] = "https://api.alternative.me/fng/?limit=0&format=json"
FEAR_GREED_WEB_URL: Final[str] = "https://alternative.me/crypto/fear-and-greed-index/"

# Network and SQLite reliability parameters
SQLITE_BUSY_TIMEOUT_MS: Final[int] = 30000
REQUEST_TIMEOUT_SEC: Final[int] = 30
MAX_RETRIES: Final[int] = 5
RETRY_BACKOFF_SEC: Final[float] = 2.0

# Shared downstream pipeline contracts (Phases 2-5)
TRAIN_END_ISO: Final[str] = "2024-12-31T23:59:59Z"
VAL_START_ISO: Final[str] = "2025-01-01T00:00:00Z"
VAL_END_ISO: Final[str] = "2025-12-31T23:59:59Z"
TEST_START_ISO: Final[str] = "2026-01-01T00:00:00Z"

INITIAL_CAPITAL: Final[float] = 10000.0
MAX_CONCURRENT_POSITIONS: Final[int] = 3
TRADING_FEE_RATE: Final[float] = 0.001
STOP_LOSS_PCT: Final[float] = 0.02
TAKE_PROFIT_PCT: Final[float] = 0.04


def ensure_directories() -> None:
    """Create required runtime directories if they do not exist."""
    for directory in (DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR, DB_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)


def configure_logging(level: str | None = None) -> logging.Logger:
    """Configure standardized ASCII logging for the Aegis Alpha pipeline."""
    resolved_level = (level or LOG_LEVEL).upper()
    numeric_level = getattr(logging, resolved_level, logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("aegis_alpha")
