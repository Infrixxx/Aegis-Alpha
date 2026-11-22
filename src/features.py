"""Feature engineering and data transformation pipeline for Aegis Alpha.

Phase 2 module:
- Loads OHLCV candles and Crypto Fear & Greed Index from SQLite or Parquet.
- Merges market data and sentiment data without future lookahead leakage.
- Generates 30+ technical and derived numeric features using pandas_ta
  (with vectorized Pandas fallbacks for cross-version portability).
- Generates 5+ transformation/dummy feature sets (hour_of_day, day_of_week,
  month, volatility_regime_bins, sentiment_regime_dummies, plus symbol dummies).
- Explicitly defines TO_PREDICT, NUMERIC, and DUMMIES feature contracts.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

from src.config import DB_PATH, SYMBOLS

logger = logging.getLogger("aegis_alpha.features")

try:
    import pandas_ta as ta  # type: ignore[import-untyped]

    HAS_PANDAS_TA = True
except Exception:  # pragma: no cover - fallback for environments without pandas_ta
    ta = None
    HAS_PANDAS_TA = False

# Prediction target contract
TARGET_HORIZON_BARS: Final[int] = 16  # 16 * 15m = 4 hours
TARGET_RETURN_THRESHOLD: Final[float] = 0.005  # +0.5% forward 4h return (> 0.2% round-trip fee)
TARGET_COL: Final[str] = "is_positive_growth_4h"
TO_PREDICT: Final[list[str]] = [TARGET_COL]

# 35 distinct numeric technical, volatility, microstructure, and sentiment features
NUMERIC: Final[list[str]] = [
    # Momentum and trend oscillators (pandas_ta)
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    # Volatility bands and range indicators (pandas_ta)
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "bb_bandwidth",
    "bb_percent_b",
    "atr_14",
    "atr_pct_14",
    # Exponential moving averages and relative spreads (pandas_ta)
    "ema_9",
    "ema_21",
    "ema_50",
    "ema_200",
    "ema_9_21_ratio",
    "ema_50_200_ratio",
    "close_to_ema_50",
    "close_to_ema_200",
    # Multi-horizon rolling realized volatility
    "vol_1h",
    "vol_4h",
    "vol_24h",
    # Volume and liquidity flow indicators
    "obv",
    "obv_momentum_24h",
    "vwap_ratio",
    "volume_zscore_24h",
    # Alternative data: Crypto Fear & Greed lagged and momentum features
    "fg_value",
    "fg_value_lag_1d",
    "fg_value_lag_3d",
    "fg_value_lag_7d",
    "fg_change_1d",
    "fg_rolling_7d_mean",
    # Long-term price positioning and short-term returns
    "price_to_52w_low",
    "price_to_52w_high",
    "ret_15m",
    "ret_1h",
    "ret_4h",
    "ret_24h",
    "candle_body_pct",
    "candle_range_pct",
]

# Deterministic dummy schemas across 6 categorical transformation sets (>= 5 required)
HOUR_DUMMIES: Final[list[str]] = [f"hour_{h}" for h in range(24)]
DOW_DUMMIES: Final[list[str]] = [f"dow_{d}" for d in range(7)]
MONTH_DUMMIES: Final[list[str]] = [f"month_{m}" for m in range(1, 13)]
VOL_REGIME_LABELS: Final[list[str]] = ["low", "normal", "high", "extreme"]
VOL_REGIME_DUMMIES: Final[list[str]] = [f"vol_regime_{lbl}" for lbl in VOL_REGIME_LABELS]
SENTIMENT_REGIME_LABELS: Final[list[str]] = [
    "extreme_fear",
    "fear",
    "neutral",
    "greed",
    "extreme_greed",
]
SENTIMENT_REGIME_DUMMIES: Final[list[str]] = [
    f"sent_regime_{lbl}" for lbl in SENTIMENT_REGIME_LABELS
]


def _symbol_to_slug(symbol: str) -> str:
    return symbol.lower().replace("/", "_").replace("-", "_")


SYMBOL_DUMMIES: Final[list[str]] = [f"sym_{_symbol_to_slug(sym)}" for sym in SYMBOLS]

DUMMIES: Final[list[str]] = (
    HOUR_DUMMIES
    + DOW_DUMMIES
    + MONTH_DUMMIES
    + VOL_REGIME_DUMMIES
    + SENTIMENT_REGIME_DUMMIES
    + SYMBOL_DUMMIES
)

FEATURE_COLS: Final[list[str]] = NUMERIC + DUMMIES

# 52 weeks of 15-minute candles = 52 * 7 * 24 * 4 = 34,944 bars (~35,040 for 365d)
BARS_PER_DAY: Final[int] = 96
BARS_52_WEEKS: Final[int] = 365 * BARS_PER_DAY


def load_raw_tables_from_sqlite(
    db_path: Path = DB_PATH,
    symbols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ohlcv and fear_greed tables from SQLite into Pandas DataFrames."""
    resolved = Path(db_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"SQLite database not found at {resolved}. "
            "Run Phase 1 ingestion ('python -m src.data_fetcher') or use --mode offline."
        )

    conn = sqlite3.connect(str(resolved))
    try:
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            ohlcv_query = (
                "SELECT symbol, timestamp, datetime_utc, open, high, low, close, volume "
                f"FROM ohlcv WHERE symbol IN ({placeholders}) "
                "ORDER BY symbol, timestamp;"
            )
            df_ohlcv = pd.read_sql_query(ohlcv_query, conn, params=list(symbols))
        else:
            ohlcv_query = (
                "SELECT symbol, timestamp, datetime_utc, open, high, low, close, volume "
                "FROM ohlcv ORDER BY symbol, timestamp;"
            )
            df_ohlcv = pd.read_sql_query(ohlcv_query, conn)

        fg_query = (
            "SELECT timestamp, date_utc, value, classification, source "
            "FROM fear_greed ORDER BY timestamp;"
        )
        df_fg = pd.read_sql_query(fg_query, conn)
    finally:
        conn.close()

    logger.info(
        "Loaded %d OHLCV rows and %d Fear & Greed rows from %s.",
        len(df_ohlcv),
        len(df_fg),
        resolved,
    )
    return df_ohlcv, df_fg


def merge_ohlcv_and_fear_greed(
    df_ohlcv: pd.DataFrame,
    df_fg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge 15m OHLCV bars with daily Fear & Greed Index by UTC date without leakage."""
    df = df_ohlcv.copy()
    if "datetime_utc" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime_utc"], utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["datetime_utc"] = df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    df["date_utc"] = df["datetime"].dt.strftime("%Y-%m-%d")

    if df_fg is not None and not df_fg.empty:
        fg = df_fg[["date_utc", "value", "classification"]].copy()
        fg = fg.drop_duplicates(subset=["date_utc"], keep="last").sort_values("date_utc")
        fg = fg.rename(
            columns={
                "value": "fg_value",
                "classification": "fg_classification",
            }
        )

        # Build continuous daily index to forward-fill occasional missing sentiment days
        date_range = pd.date_range(
            start=min(df["date_utc"].min(), fg["date_utc"].min()),
            end=max(df["date_utc"].max(), fg["date_utc"].max()),
            freq="D",
            tz="UTC",
        ).strftime("%Y-%m-%d")
        fg_daily = pd.DataFrame({"date_utc": date_range}).merge(fg, on="date_utc", how="left")
        fg_daily["fg_value"] = fg_daily["fg_value"].ffill().bfill().fillna(50.0)
        fg_daily["fg_classification"] = fg_daily["fg_classification"].ffill().bfill().fillna("Neutral")

        # Compute daily Fear & Greed lags cleanly on daily series prior to 15m join
        fg_daily["fg_value_lag_1d"] = fg_daily["fg_value"].shift(1)
        fg_daily["fg_value_lag_3d"] = fg_daily["fg_value"].shift(3)
        fg_daily["fg_value_lag_7d"] = fg_daily["fg_value"].shift(7)
        fg_daily["fg_change_1d"] = fg_daily["fg_value"] - fg_daily["fg_value_lag_1d"]
        fg_daily["fg_rolling_7d_mean"] = fg_daily["fg_value"].rolling(7, min_periods=1).mean()

        df = df.merge(fg_daily, on="date_utc", how="left")
    else:
        logger.warning("Fear & Greed table empty or omitted; defaulting sentiment to neutral (50).")
        df["fg_value"] = 50.0
        df["fg_classification"] = "Neutral"
        df["fg_value_lag_1d"] = 50.0
        df["fg_value_lag_3d"] = 50.0
        df["fg_value_lag_7d"] = 50.0
        df["fg_change_1d"] = 0.0
        df["fg_rolling_7d_mean"] = 50.0

    for col in ("fg_value", "fg_value_lag_1d", "fg_value_lag_3d", "fg_value_lag_7d", "fg_rolling_7d_mean"):
        df[col] = df[col].ffill().bfill().fillna(50.0).astype("float64")
    df["fg_change_1d"] = df["fg_change_1d"].fillna(0.0).astype("float64")

    return df


def _compute_rsi_14(close: pd.Series) -> pd.Series:
    if HAS_PANDAS_TA and ta is not None:
        res = ta.rsi(close, length=14)
        if res is not None:
            return res
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    if HAS_PANDAS_TA and ta is not None:
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None and macd_df.shape[1] >= 3:
            # pandas_ta returns MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            macd_col = [c for c in macd_df.columns if c.startswith("MACD_")][0]
            hist_col = [c for c in macd_df.columns if c.startswith("MACDh_")][0]
            sig_col = [c for c in macd_df.columns if c.startswith("MACDs_")][0]
            return macd_df[macd_col], macd_df[sig_col], macd_df[hist_col]
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist_line = macd_line - signal_line
    return macd_line, signal_line, hist_line


def _compute_bbands(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    if HAS_PANDAS_TA and ta is not None:
        bb_df = ta.bbands(close, length=20, std=2.0)
        if bb_df is not None and bb_df.shape[1] >= 3:
            lower_col = [c for c in bb_df.columns if c.startswith("BBL_")][0]
            mid_col = [c for c in bb_df.columns if c.startswith("BBM_")][0]
            upper_col = [c for c in bb_df.columns if c.startswith("BBU_")][0]
            return bb_df[upper_col], bb_df[mid_col], bb_df[lower_col]
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    return mid + 2.0 * std, mid, mid - 2.0 * std


def _compute_atr_14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    if HAS_PANDAS_TA and ta is not None:
        res = ta.atr(high=high, low=low, close=close, length=14)
        if res is not None:
            return res
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()


def _compute_ema(close: pd.Series, length: int) -> pd.Series:
    if HAS_PANDAS_TA and ta is not None:
        res = ta.ema(close, length=length)
        if res is not None:
            return res
    return close.ewm(span=length, adjust=False).mean()


def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    if HAS_PANDAS_TA and ta is not None:
        res = ta.obv(close=close, volume=volume)
        if res is not None:
            return res
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def _compute_symbol_indicators(group: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical, volatility, microstructure, and target columns for one symbol."""
    g = group.sort_values("timestamp").copy()
    eps = 1e-9

    open_p = g["open"].astype("float64")
    high_p = g["high"].astype("float64")
    low_p = g["low"].astype("float64")
    close_p = g["close"].astype("float64")
    volume = g["volume"].astype("float64")

    # 1. Momentum oscillators
    g["rsi_14"] = _compute_rsi_14(close_p)
    macd_line, macd_sig, macd_hist = _compute_macd(close_p)
    g["macd"] = macd_line
    g["macd_signal"] = macd_sig
    g["macd_hist"] = macd_hist

    # 2. Bollinger Bands
    bb_upper, bb_mid, bb_lower = _compute_bbands(close_p)
    g["bb_upper"] = bb_upper
    g["bb_mid"] = bb_mid
    g["bb_lower"] = bb_lower
    g["bb_bandwidth"] = (bb_upper - bb_lower) / (bb_mid.abs() + eps)
    g["bb_percent_b"] = (close_p - bb_lower) / ((bb_upper - bb_lower).abs() + eps)

    # 3. ATR
    g["atr_14"] = _compute_atr_14(high_p, low_p, close_p)
    g["atr_pct_14"] = g["atr_14"] / (close_p + eps)

    # 4. Exponential Moving Averages
    g["ema_9"] = _compute_ema(close_p, 9)
    g["ema_21"] = _compute_ema(close_p, 21)
    g["ema_50"] = _compute_ema(close_p, 50)
    g["ema_200"] = _compute_ema(close_p, 200)
    g["ema_9_21_ratio"] = (g["ema_9"] / (g["ema_21"] + eps)) - 1.0
    g["ema_50_200_ratio"] = (g["ema_50"] / (g["ema_200"] + eps)) - 1.0
    g["close_to_ema_50"] = (close_p / (g["ema_50"] + eps)) - 1.0
    g["close_to_ema_200"] = (close_p / (g["ema_200"] + eps)) - 1.0

    # 5. Returns and rolling volatility (1h = 4 bars, 4h = 16 bars, 24h = 96 bars)
    log_ret = np.log(close_p / close_p.shift(1).replace(0.0, np.nan))
    g["ret_15m"] = close_p.pct_change(1)
    g["ret_1h"] = close_p.pct_change(4)
    g["ret_4h"] = close_p.pct_change(16)
    g["ret_24h"] = close_p.pct_change(96)

    g["vol_1h"] = log_ret.rolling(4, min_periods=4).std()
    g["vol_4h"] = log_ret.rolling(16, min_periods=8).std()
    g["vol_24h"] = log_ret.rolling(96, min_periods=24).std()

    # 6. OBV and rolling 24h VWAP ratio
    g["obv"] = _compute_obv(close_p, volume)
    obv_scale = g["obv"].abs().rolling(96, min_periods=24).mean() + eps
    g["obv_momentum_24h"] = g["obv"].diff(96) / obv_scale

    typical_price = (high_p + low_p + close_p) / 3.0
    pv_24h = (typical_price * volume).rolling(96, min_periods=16).sum()
    vol_24h_sum = volume.rolling(96, min_periods=16).sum()
    rolling_vwap_24h = pv_24h / (vol_24h_sum + eps)
    g["vwap_ratio"] = close_p / (rolling_vwap_24h + eps)

    vol_mean_24h = volume.rolling(96, min_periods=24).mean()
    vol_std_24h = volume.rolling(96, min_periods=24).std()
    g["volume_zscore_24h"] = (volume - vol_mean_24h) / (vol_std_24h + eps)

    # 7. Price-to-52w-low and 52w-high ratios
    rolling_52w_low = low_p.rolling(BARS_52_WEEKS, min_periods=96).min()
    rolling_52w_high = high_p.rolling(BARS_52_WEEKS, min_periods=96).max()
    g["price_to_52w_low"] = close_p / (rolling_52w_low + eps)
    g["price_to_52w_high"] = close_p / (rolling_52w_high + eps)

    # 8. Candlestick microstructure
    g["candle_body_pct"] = (close_p - open_p) / (open_p + eps)
    g["candle_range_pct"] = (high_p - low_p) / (close_p + eps)

    # 9. Causal volatility regime bins per symbol using trailing 30-day rolling quantiles
    # Using trailing rolling quantiles avoids future lookahead leakage.
    window_30d = 30 * BARS_PER_DAY
    q25 = g["vol_24h"].rolling(window_30d, min_periods=96).quantile(0.25)
    q75 = g["vol_24h"].rolling(window_30d, min_periods=96).quantile(0.75)
    q90 = g["vol_24h"].rolling(window_30d, min_periods=96).quantile(0.90)

    vol_regime = np.where(
        g["vol_24h"] >= q90,
        "extreme",
        np.where(
            g["vol_24h"] >= q75,
            "high",
            np.where(g["vol_24h"] <= q25, "low", "normal"),
        ),
    )
    g["volatility_regime"] = np.where(g["vol_24h"].isna(), "normal", vol_regime)

    # 10. Forward target columns (computed strictly inside each symbol group)
    g["future_return_15m"] = (close_p.shift(-1) / (close_p + eps)) - 1.0
    g["future_return_4h"] = (close_p.shift(-TARGET_HORIZON_BARS) / (close_p + eps)) - 1.0
    g[TARGET_COL] = np.where(
        g["future_return_4h"].notna(),
        (g["future_return_4h"] >= TARGET_RETURN_THRESHOLD).astype(int),
        np.nan,
    )

    return g


def _classify_sentiment_slug(fg_val: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                fg_val <= 24.0,
                fg_val <= 44.0,
                fg_val <= 55.0,
                fg_val <= 74.0,
            ],
            [
                "extreme_fear",
                "fear",
                "neutral",
                "greed",
            ],
            default="extreme_greed",
        ),
        index=fg_val.index,
    )


def _add_transformation_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Create 6 deterministic dummy/transformation feature sets."""
    out = df.copy()
    dt = out["datetime"]

    out["hour_of_day"] = dt.dt.hour.astype(int)
    out["day_of_week"] = dt.dt.dayofweek.astype(int)
    out["month"] = dt.dt.month.astype(int)
    out["sentiment_regime"] = _classify_sentiment_slug(out["fg_value"])

    # Set 1: hour_of_day dummies (24)
    hour_dummies = pd.get_dummies(out["hour_of_day"], prefix="hour", dtype=np.int8)
    hour_dummies = hour_dummies.reindex(columns=HOUR_DUMMIES, fill_value=0)

    # Set 2: day_of_week dummies (7)
    dow_dummies = pd.get_dummies(out["day_of_week"], prefix="dow", dtype=np.int8)
    dow_dummies = dow_dummies.reindex(columns=DOW_DUMMIES, fill_value=0)

    # Set 3: month dummies (12)
    month_dummies = pd.get_dummies(out["month"], prefix="month", dtype=np.int8)
    month_dummies = month_dummies.reindex(columns=MONTH_DUMMIES, fill_value=0)

    # Set 4: volatility_regime_bins dummies (4)
    vol_dummies = pd.get_dummies(out["volatility_regime"], prefix="vol_regime", dtype=np.int8)
    vol_dummies = vol_dummies.reindex(columns=VOL_REGIME_DUMMIES, fill_value=0)

    # Set 5: sentiment_regime_dummies (5)
    sent_dummies = pd.get_dummies(out["sentiment_regime"], prefix="sent_regime", dtype=np.int8)
    sent_dummies = sent_dummies.reindex(columns=SENTIMENT_REGIME_DUMMIES, fill_value=0)

    # Set 6: symbol dummies (5)
    sym_slugs = out["symbol"].map(_symbol_to_slug)
    sym_dummies = pd.get_dummies(sym_slugs, prefix="sym", dtype=np.int8)
    sym_dummies = sym_dummies.reindex(columns=SYMBOL_DUMMIES, fill_value=0)

    out = pd.concat(
        [out, hour_dummies, dow_dummies, month_dummies, vol_dummies, sent_dummies, sym_dummies],
        axis=1,
    )
    return out


def build_feature_dataframe(
    df_ohlcv: pd.DataFrame,
    df_fg: pd.DataFrame | None = None,
    drop_na_features: bool = True,
    drop_na_target: bool = True,
) -> pd.DataFrame:
    """End-to-end feature engineering from raw OHLCV and Fear & Greed DataFrames."""
    if df_ohlcv.empty:
        raise ValueError("Input OHLCV DataFrame is empty.")

    merged = merge_ohlcv_and_fear_greed(df_ohlcv=df_ohlcv, df_fg=df_fg)

    per_symbol_frames: list[pd.DataFrame] = []
    for _, grp in merged.groupby("symbol", sort=True):
        per_symbol_frames.append(_compute_symbol_indicators(grp))

    featured = pd.concat(per_symbol_frames, axis=0, ignore_index=True)
    featured = _add_transformation_dummies(featured)

    # Clean infinities before missing-value filtering
    featured[NUMERIC] = featured[NUMERIC].replace([np.inf, -np.inf], np.nan)

    initial_rows = len(featured)
    if drop_na_features:
        featured = featured.dropna(subset=NUMERIC)
    if drop_na_target:
        featured = featured.dropna(subset=TO_PREDICT)
        featured[TARGET_COL] = featured[TARGET_COL].astype(int)

    featured = featured.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    logger.info(
        "Built feature dataset: %d rows (dropped %d warmup/tail rows), "
        "%d NUMERIC features, %d DUMMIES features (pandas_ta=%s).",
        len(featured),
        initial_rows - len(featured),
        len(NUMERIC),
        len(DUMMIES),
        HAS_PANDAS_TA,
    )
    return featured


def build_features_from_sqlite(
    db_path: Path = DB_PATH,
    symbols: Sequence[str] | None = None,
    drop_na_features: bool = True,
    drop_na_target: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper to load SQLite tables and generate the feature DataFrame."""
    df_ohlcv, df_fg = load_raw_tables_from_sqlite(db_path=db_path, symbols=symbols)
    return build_feature_dataframe(
        df_ohlcv=df_ohlcv,
        df_fg=df_fg,
        drop_na_features=drop_na_features,
        drop_na_target=drop_na_target,
    )


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """Persist DataFrame to Parquet (falling back to gzip CSV if pyarrow is absent)."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.suffix.lower() == ".parquet":
        try:
            df.to_parquet(resolved, index=False)
            return resolved
        except Exception as exc:
            logger.warning("Parquet save failed (%s); falling back to CSV.", exc)
            csv_path = resolved.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return csv_path
    df.to_csv(resolved, index=False)
    return resolved


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load DataFrame from Parquet or CSV with UTC datetime parsing."""
    resolved = Path(path)
    if resolved.suffix.lower() == ".parquet":
        df = pd.read_parquet(resolved)
    else:
        df = pd.read_csv(resolved)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    elif "datetime_utc" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df


def get_feature_catalog() -> dict[str, list[str]]:
    """Return explicit feature group lists for reporting and model validation."""
    return {
        "TO_PREDICT": list(TO_PREDICT),
        "NUMERIC": list(NUMERIC),
        "DUMMIES": list(DUMMIES),
        "FEATURE_COLS": list(FEATURE_COLS),
    }
