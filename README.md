# Aegis Alpha: End-to-End Algorithmic Crypto Trading Pipeline

Aegis Alpha is a production-grade, modular quantitative research and algorithmic trading system built for Python 3.12 (Pop!_OS / Ubuntu Linux) and engineered to satisfy every technical criterion of the DataTalks.Club Stock Markets Analytics Zoomcamp grading rubric.

The pipeline ingests more than 1,050,000 15-minute OHLCV candles across five high-liquidity cryptocurrency spot pairs alongside historical Crypto Fear & Greed sentiment data into a WAL-enabled SQLite database, generates 40 continuous technical/sentiment features plus 57 categorical dummy indicators, trains four classical and deep-learning classifiers with strict chronological splits, evaluates strategies using both vectorized and exact event-driven portfolio simulations against a BTC Buy-and-Hold benchmark, and automates execution via Binance Spot Testnet, Telegram notifications, Cron, and Docker.

---

## Table of Contents

1. [Problem Description](#1-problem-description)
2. [System Architecture](#2-system-architecture)
3. [Data Sources and SQLite Storage Schema](#3-data-sources-and-sqlite-storage-schema)
4. [Feature Engineering and 40+ Feature Dictionary](#4-feature-engineering-and-40-feature-dictionary)
5. [Exploratory Data Analysis (EDA) Methodology and Findings](#5-exploratory-data-analysis-eda-methodology-and-findings)
6. [Multi-Model Training, Tuning, and Custom Decision Rules](#6-multi-model-training-tuning-and-custom-decision-rules)
7. [Trading Simulations: Vectorized vs. Exact Iterative vs. Benchmark](#7-trading-simulations-vectorized-vs-exact-iterative-vs-benchmark)
8. [Error Analysis and False Positive Debugging](#8-error-analysis-and-false-positive-debugging)
9. [Two-Regime Reproduction Guide (Offline vs. Online)](#9-two-regime-reproduction-guide-offline-vs-online)
10. [Automation: Cron Schedule, Docker, Broker, and Telegram](#10-automation-cron-schedule-docker-broker-and-telegram)
11. [Repository Structure and Security Controls](#11-repository-structure-and-security-controls)

---

## 1. Problem Description

Cryptocurrency spot markets operate 24/7 with frequent volatility clustering, rapid sentiment reversals, and non-trivial exchange transaction fees (0.10% maker/taker per side on Binance, or 0.20% round-trip). Naive technical crossover strategies suffer from excessive turnover and severe whipsaws during liquidation cascades.

**Quantitative Objective:** Predict short-horizon upward momentum events where an asset delivers at least a `+0.50%` return over the next 4 hours (16 consecutive 15-minute candles), providing a `2.5x` buffer over Binance round-trip fees:

- **Prediction Target (`TO_PREDICT`):** `is_positive_growth_4h = 1` if `(close[t + 16] / close[t]) - 1 >= 0.005`, else `0`.
- **Asset Universe (5 Pairs):** `BTC/USDT`, `ETH/USDT`, `SOL/USDT`, `BNB/USDT`, `ADA/USDT`.
- **Frequency and Horizon:** 15-minute (`15m`) OHLCV bars from `2020-01-01T00:00:00Z` to present (`>1,050,000` rows total).
- **Risk Mandate:** Maximize risk-adjusted returns (Sharpe Ratio and CAGR) while strictly limiting drawdowns via contrarian sentiment filtering (`fg_value < 45`), extreme-volatility exclusion, a 3-position concurrency cap, probability-scaled sizing, and hard `2%` Stop-Loss / `4%` Take-Profit exits.

---

## 2. System Architecture

```text
+-------------------------------------------------------------------------+
| PHASE 1: INCREMENTAL INGESTION (src/data_fetcher.py)                    |
|  - Binance 15m OHLCV via ccxt (enableRateLimit=True, limit=1000 loop)   |
|  - Crypto Fear & Greed API + BeautifulSoup / Playwright fallbacks       |
|  - SQLite WAL storage (data/market_data.db) with composite PKs          |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| PHASE 2: FEATURE ENGINEERING & EDA (src/features.py, 01_eda.ipynb)      |
|  - Leakage-free UTC daily join + lagged sentiment features              |
|  - 40 NUMERIC technical/volatility/volume/sentiment features (pandas_ta)|
|  - 6 DUMMIES transformation sets (57 binary indicators)                 |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| PHASE 3: CHRONOLOGICAL MODELING (src/train.py -> models/)               |
|  - Split: Train (2020-2024) | Validation (2025) | Test (2026+)          |
|  - Baselines: DecisionTreeClassifier, RandomForestClassifier            |
|  - Advanced : Tuned XGBClassifier (Optuna/RandSearch), PyTorch LSTM     |
|  - Decision : Ensemble prob > 0.65 + FG < 45 + non-extreme vol + RSI<70 |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| PHASE 4: SIMULATIONS & DIAGNOSTICS (src/simulator.py)                   |
|  - Vectorized: sim1_gross_rev/sim1_net_rev vs sim2_gross_rev/sim2_net   |
|  - Exact (itertuples): $10,000, max 3 pos, prob sizing, 0.1% fee, SL/TP |
|  - Benchmark: BTC/USDT Buy-and-Hold (CAGR, Sharpe, MaxDD, Rolling 30d)  |
|  - Slicing  : Precision & PnL by Ticker, Month, Sentiment, Volatility   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| PHASE 5: AUTOMATION & EXECUTION (main.py, broker.py, notifier.py)       |
|  - --mode online (live incremental) / --mode offline (instant review)   |
|  - Binance Spot Testnet / Dry-Run broker + Telegram / Console alerts    |
|  - Pop!_OS Linux Cron schedule + python:3.12-slim Dockerfile            |
+-------------------------------------------------------------------------+
```

---

## 3. Data Sources and SQLite Storage Schema

| Source | Endpoint / Interface | Granularity | Volume | SQLite Table | Primary Key |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Binance Spot | `ccxt.binance({'enableRateLimit': True})` | `15m` OHLCV | >1,050,000 rows (5 pairs x ~210k bars since 2020-01-01) | `ohlcv` | `(symbol, timestamp)` (`WITHOUT ROWID`) |
| Alternative.me | `https://api.alternative.me/fng/?limit=0&format=json` (BS4 + Playwright fallback) | Daily (`1D`) | >3,200 daily observations (since 2018-02-01) | `fear_greed` | `(timestamp)` + `UNIQUE(date_utc)` |

### Database Concurrency and Incremental Sync Guarantees
- **WAL Mode:** Every connection executes `PRAGMA journal_mode=WAL;`, `PRAGMA synchronous=NORMAL;`, and `PRAGMA busy_timeout=30000;`, enabling safe concurrent reads (EDA/inference) during scheduled writes.
- **Incremental Resumption:** Before paginating Binance (`limit=1000`, `since = last_timestamp + 1`), `src/data_fetcher.py` executes `SELECT MAX(timestamp) FROM ohlcv WHERE symbol = ?`. Subsequent runs pull only newly closed candles, avoiding partial/unclosed bar contamination.
- **Resilient Retries:** Transient `RateLimitExceeded`, `DDoSProtection`, `NetworkError`, and `RequestTimeout` exceptions are handled with exponential backoff.

---

## 4. Feature Engineering and 40+ Feature Dictionary

All feature lists are explicitly exported in `src/features.py` as `TO_PREDICT`, `NUMERIC`, `DUMMIES`, and `FEATURE_COLS`. Indicators are computed strictly per symbol in chronological order using `pandas_ta` (with deterministic vectorized Pandas fallbacks).

### Target Contract (`TO_PREDICT`)
| Column | Type | Definition |
| :--- | :--- | :--- |
| `is_positive_growth_4h` | `int` (0/1) | `1` when `(close.shift(-16) / close) - 1.0 >= 0.005`, else `0`. |

### Continuous Features (`NUMERIC` - 40 Features)

| Category | Feature Name | Definition / Formula |
| :--- | :--- | :--- |
| Momentum (`pandas_ta`) | `rsi_14` | 14-period Relative Strength Index |
| Momentum (`pandas_ta`) | `macd` | MACD line (EMA 12 minus EMA 26) |
| Momentum (`pandas_ta`) | `macd_signal` | 9-period EMA signal line of MACD |
| Momentum (`pandas_ta`) | `macd_hist` | MACD histogram (`macd - macd_signal`) |
| Volatility Bands (`pandas_ta`) | `bb_upper` | 20-period Bollinger Band Upper (`+2.0` std) |
| Volatility Bands (`pandas_ta`) | `bb_mid` | 20-period Bollinger Band Mid (SMA 20) |
| Volatility Bands (`pandas_ta`) | `bb_lower` | 20-period Bollinger Band Lower (`-2.0` std) |
| Volatility Bands | `bb_bandwidth` | Normalized band width: `(bb_upper - bb_lower) / bb_mid` |
| Volatility Bands | `bb_percent_b` | Relative band position: `(close - bb_lower) / (bb_upper - bb_lower)` |
| Volatility (`pandas_ta`) | `atr_14` | 14-period Average True Range |
| Volatility | `atr_pct_14` | Price-normalized ATR: `atr_14 / close` |
| Trend EMAs (`pandas_ta`) | `ema_9`, `ema_21`, `ema_50`, `ema_200` | 9, 21, 50, and 200-bar Exponential Moving Averages |
| Trend Spreads | `ema_9_21_ratio`, `ema_50_200_ratio` | Short-term (`ema_9 / ema_21 - 1`) and macro (`ema_50 / ema_200 - 1`) trend spreads |
| Mean Reversion | `close_to_ema_50`, `close_to_ema_200` | Distance from medium and macro trend anchors |
| Realized Volatility | `vol_1h`, `vol_4h`, `vol_24h` | Rolling log-return standard deviation over 4, 16, and 96 bars |
| Volume Flow (`pandas_ta`) | `obv` | On-Balance Volume cumulative flow |
| Volume Flow | `obv_momentum_24h` | 24h OBV change scaled by 24h mean absolute OBV |
| Liquidity Anchor | `vwap_ratio` | `close / rolling_24h_vwap` (typical price weighted by 96-bar volume) |
| Volume Anomaly | `volume_zscore_24h` | Standardized 15m volume vs. trailing 24h mean and std |
| Sentiment (Alt Data) | `fg_value` | Daily Crypto Fear & Greed Index (0-100) joined on UTC date |
| Sentiment Lags | `fg_value_lag_1d`, `fg_value_lag_3d`, `fg_value_lag_7d` | 1-day, 3-day, and 7-day lagged Fear & Greed Index |
| Sentiment Dynamics | `fg_change_1d`, `fg_rolling_7d_mean` | 1-day sentiment delta and 7-day rolling sentiment mean |
| Macro Price Position | `price_to_52w_low`, `price_to_52w_high` | Ratio of `close` to trailing 52-week (35,040-bar) low and high |
| Multi-Horizon Returns | `ret_15m`, `ret_1h`, `ret_4h`, `ret_24h` | Trailing 1, 4, 16, and 96-bar percentage returns |
| Candle Microstructure | `candle_body_pct`, `candle_range_pct` | `(close - open) / open` and `(high - low) / close` |

### Transformation and Dummy Feature Sets (`DUMMIES` - 6 Sets, 57 Indicators)

| Set | Columns | Construction |
| :--- | :--- | :--- |
| 1. `hour_of_day` | `hour_0` .. `hour_23` (24 cols) | UTC hour one-hot encoding capturing Asian/EU/US session liquidity shifts |
| 2. `day_of_week` | `dow_0` .. `dow_6` (7 cols) | Weekday vs. weekend regime encoding |
| 3. `month` | `month_1` .. `month_12` (12 cols) | Calendar seasonality indicators |
| 4. `volatility_regime_bins` | `vol_regime_low`, `vol_regime_normal`, `vol_regime_high`, `vol_regime_extreme` (4 cols) | Causal trailing 30-day rolling quantile bins (`<=25%`, `25-75%`, `75-90%`, `>=90%`) preventing future leakage |
| 5. `sentiment_regime_dummies` | `sent_regime_extreme_fear`, `sent_regime_fear`, `sent_regime_neutral`, `sent_regime_greed`, `sent_regime_extreme_greed` (5 cols) | Standard Alternative.me Fear & Greed classification bins |
| 6. `symbol` | `sym_btc_usdt` .. `sym_ada_usdt` (5 cols) | Asset-level fixed effects for cross-sectional modeling |

---

## 5. Exploratory Data Analysis (EDA) Methodology and Findings

EDA is implemented in [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) and covers four core checks:

1. **Missing-Value Audit:** Raw indicator computation produces expected leading warmup NaNs (primarily 200-bar EMA and 96-bar volatility windows) plus 16 trailing target rows per symbol. After warmup/tail filtering, remaining NaNs and duplicate `(symbol, timestamp)` keys are verified at `0`.
2. **Target Class Balance:** Across 15m crypto intervals, `is_positive_growth_4h` (`>= +0.5%` in 4h) represents roughly 30%-40% of bars depending on asset volatility (`SOL/USDT` and `ADA/USDT` exhibit higher positive-threshold frequency and wider return variance than `BTC/USDT` and `BNB/USDT`).
3. **Sentiment and Volatility Regime Asymmetry:** Pivoting forward 4h returns across `sentiment_regime x volatility_regime` reveals that `Extreme` volatility bins carry disproportionately severe 5th-percentile tail losses, while `Fear` / `Extreme Fear` regimes in `normal` or `high` (non-extreme) volatility offer superior risk-adjusted bounce asymmetry.
4. **Correlation Drivers:** Normalized volatility (`atr_pct_14`, `vol_24h`, `bb_bandwidth`), mean-reversion oscillators (`rsi_14`, `bb_percent_b`, `vwap_ratio`), and lagged sentiment features (`fg_value`, `fg_rolling_7d_mean`) consistently rank among the strongest linear and rank-correlated predictors of forward 4h upside moves.

---

## 6. Multi-Model Training, Tuning, and Custom Decision Rules

Implemented in [src/train.py](src/train.py) with strict chronological splitting to guarantee zero temporal leakage:

- **Train Split:** `2020-01-01` through `2024-12-31`
- **Validation Split:** `2025-01-01` through `2025-12-31` (hyperparameter tuning, early stopping, LSTM checkpointing)
- **Test Split (Out-of-Sample):** `2026-01-01` through present

### Model Architecture Matrix

| Model | Role | Feature Inputs | Key Regularization / Tuning | Saved Artifact |
| :--- | :--- | :--- | :--- | :--- |
| `DecisionTreeClassifier` | Lecture Baseline 1 | `NUMERIC + DUMMIES` (97) | `max_depth=10`, `min_samples_leaf=50`, `class_weight='balanced'` | `models/decision_tree.joblib` |
| `RandomForestClassifier` | Lecture Baseline 2 | `NUMERIC + DUMMIES` (97) | `n_estimators=150`, `max_depth=12`, `min_samples_leaf=50` | `models/random_forest.joblib` |
| `XGBClassifier` (Tuned) | Advanced Model 1 | `NUMERIC + DUMMIES` (97) | Optuna TPE search (or `RandomizedSearchCV` + `TimeSeriesSplit`) over depth, LR, subsample, colsample, L1/L2, `min_child_weight`, early stopping | `models/xgboost_tuned.joblib` |
| PyTorch `CryptoLSTMClassifier` | Advanced Model 2 | 16-bar (`4h`) sequences of Train-scaled `NUMERIC` (40) | 2-layer LSTM (`hidden_dim=64`), `LayerNorm`, `Dropout(0.25)`, `BCEWithLogitsLoss(pos_weight)`, `AdamW`, gradient clipping, early stopping | `models/lstm_classifier.pt` + `models/lstm_scaler.joblib` |
| **Ensemble Blend** | Production Probability | `0.65 * P(XGB) + 0.35 * P(LSTM)` | Combines tree-based tabular splits with temporal sequence representations | `models/model_metadata.json` |

### Custom High-Probability Decision Rule (`pred_custom_rule`)

To suppress noisy mid-confidence predictions and avoid fee drag, a `BUY` signal triggers **only** when all four gates pass simultaneously:

1. `pred_prob_ensemble > 0.65` (high conviction probability)
2. `fg_value < 45` (contrarian Fear / Extreme Fear sentiment filter)
3. `volatility_regime != "extreme"` (excludes liquidation cascades and gap whipsaws)
4. `rsi_14 < 70` (avoids chasing overbought momentum exhaustion)

Run metrics for every model (`roc_auc`, `accuracy`, `precision`, `recall`, `f1`, `precision_at_065`, and rule-level `precision`/`signal_count`) are automatically written to `models/model_metadata.json` on each training run.

---

## 7. Trading Simulations: Vectorized vs. Exact Iterative vs. Benchmark

Implemented in [src/simulator.py](src/simulator.py), every evaluation run produces both fast vectorized screening metrics and an exact path-dependent portfolio simulation benchmarked directly against BTC/USDT Buy-and-Hold over the identical test window.

### Simulation Methodologies

| Engine | Columns / Mechanics | Position Sizing & Risk Rules |
| :--- | :--- | :--- |
| **Vector Sim 1 (Baseline)** | `sim1_gross_rev`, `sim1_net_rev` on `pred_baseline` (`prob >= 0.55`) | Fixed `$100` unit notional per signal; `0.20%` round-trip exchange fee |
| **Vector Sim 2 (Custom Rule)** | `sim2_gross_rev`, `sim2_net_rev` on `pred_custom_rule` | Fixed `$100` unit notional per signal; `0.20%` round-trip fee + sentiment & volatility filters |
| **Exact Iterative (`itertuples`)** | Chronological bar-by-bar portfolio loop on `pred_custom_rule` | `$10,000` initial capital; max `3` concurrent positions (max 1 per coin); dynamic slot sizing scaled by `clip((prob - 0.50) / 0.35, 0.35, 1.0)` with full capital reinvestment; `0.10%` fee per side; intrabar `2.0%` Stop-Loss (checked before TP conservatively) / `4.0%` Take-Profit / `16-bar` (`4h`) horizon timeout |
| **Benchmark** | BTC/USDT Buy-and-Hold | `$10,000` invested in `BTC/USDT` on bar 0 after `0.10%` entry fee |

### Tracked Comparison Metrics (`models/simulation_report.json`)

| Metric | Vector Sim 1 (Baseline) | Vector Sim 2 (Custom Rule) | Exact Iterative Portfolio | BTC Buy-and-Hold |
| :--- | :--- | :--- | :--- | :--- |
| Initial Capital | `$100` / signal | `$100` / signal | `$10,000.00` | `$10,000.00` |
| Transaction Fees | `0.20%` round-trip | `0.20%` round-trip | `0.10%` entry + `0.10%` exit | `0.10%` entry |
| Risk Controls | None | Sentiment + Vol filter | Max 3 pos, Prob sizing, 2% SL / 4% TP | Passive hold |
| Primary Metrics Reported | Signals, Win Rate, Gross/Net PnL, Profit Factor | Signals, Win Rate, Gross/Net PnL, Profit Factor | Final Equity, Total Return %, CAGR %, Sharpe Ratio, Max Drawdown %, 30d Rolling Returns, Win Rate, Profit Factor, Exit Reasons | Final Equity, Total Return %, CAGR %, Sharpe Ratio, Max Drawdown %, 30d Rolling Returns |

Exact run values are printed to stdout by `format_simulation_report()` and persisted to `models/simulation_report.json` after running `main.py` or `notebooks/02_workflow_runner.ipynb`.

---

## 8. Error Analysis and False Positive Debugging

`run_diagnostics()` in [src/simulator.py](src/simulator.py) performs automated multi-dimensional error slicing and writes complete tables into `models/simulation_report.json`:

1. **By Ticker (`by_ticker`):** Compares signal count, classification precision, vectorized net PnL, exact trade win rate, and exact net PnL across `BTC/USDT`, `ETH/USDT`, `SOL/USDT`, `BNB/USDT`, and `ADA/USDT` to expose whether alpha is concentrated in high-beta altcoins or broad-based.
2. **By Month (`by_month`):** Tracks temporal stability and regime drift across every month of the out-of-sample test period.
3. **By Sentiment Regime (`by_sentiment_regime`) & Volatility Regime (`by_volatility_regime`):** Verifies that returns remain positive inside `fear` / `extreme_fear` while confirming the exclusion of `extreme` volatility regimes.
4. **False Positive Feature Profile (`false_positive_debugging`):**
   - Computes exact False Positive reduction percentage from `pred_baseline` to `pred_custom_rule`.
   - Compares mean feature values between **False Positives (FP)** and **True Positives (TP)** across `rsi_14`, `atr_pct_14`, `vol_24h`, `ema_50_200_ratio`, `vwap_ratio`, and `fg_value`.
   - **Key Root Causes Identified:** Baseline False Positives systematically exhibit elevated `atr_pct_14` / `vol_24h` (panic spikes that trigger 2% stop-losses before rebounding), elevated `fg_value` (crowded greed entries prone to mean reversion), and stretched `rsi_14` (late breakout chases). Filtering on `prob > 0.65`, `fg_value < 45`, `volatility_regime != "extreme"`, and `rsi_14 < 70` directly eliminates these structural failure modes.

---

## 9. Two-Regime Reproduction Guide (Offline vs. Online)

### Prerequisites (Python 3.12 on Pop!_OS / Ubuntu Linux)

```bash
git clone <your-repo-url> Aegis-Alpha
cd Aegis-Alpha
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### Regime A: Offline Mode (Instant Peer-Review Reproduction)

Use `--mode offline` to reproduce feature engineering, model training/inference, Vector + Exact simulations, diagnostics, and console notifications immediately from a local SQLite database (`data/market_data.db`) or pre-saved Parquet snapshot (`data/offline_snapshot.parquet`) without waiting for a full historical Binance API download:

```bash
# Using a local Parquet snapshot or pre-populated data/market_data.db
python main.py --mode offline --train

# Or automatically download a hosted offline snapshot (e.g. Google Drive direct link)
python main.py --mode offline \
  --offline-url "https://drive.google.com/uc?export=download&id=<FILE_ID>" \
  --offline-path data/offline_snapshot.parquet \
  --train

# Fast CPU smoke test skipping PyTorch LSTM if GPU/RAM is constrained
python main.py --mode offline --train --skip-lstm --tuning-trials 8
```

### Regime B: Online Mode (Full Live Incremental Pipeline)

Use `--mode online` to incrementally sync all 5 trading pairs (`15m`, `2020-01-01` to present) from Binance and the full Crypto Fear & Greed Index into `data/market_data.db`, optionally exporting a fresh Parquet snapshot for offline sharing:

```bash
# Step 1 (optional standalone ingestion check)
python -m src.data_fetcher
python -m src.data_fetcher --summary-only

# Step 2: Full online end-to-end run + model training + offline snapshot export
python main.py --mode online --train --export-offline-snapshot

# Step 3: Subsequent 15-minute live inference cycles (reuses saved models, Dry-Run orders)
python main.py --mode online

# Step 4: Enable Binance Spot Testnet execution (requires BINANCE_API_KEY/SECRET in .env)
python main.py --mode online --execute-orders
```

### Interactive Notebooks

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_workflow_runner.ipynb
```

---

## 10. Automation: Cron Schedule, Docker, Broker, and Telegram

### Pop!_OS / Linux Cron Schedule (`crontab -e`)

Run incremental ingestion, live inference, dry-run/testnet execution, and Telegram notifications 1 minute after every 15-minute candle close, plus weekly Sunday model retraining at 02:00 UTC:

```cron
# Ensure project environment and logs directory exist
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Live 15-minute inference and incremental SQLite sync (at :01, :16, :31, :46 past each hour)
1,16,31,46 * * * * cd /home/$USER/Aegis-Alpha && /home/$USER/Aegis-Alpha/.venv/bin/python main.py --mode online >> /home/$USER/Aegis-Alpha/logs/cron_online.log 2>&1

# Weekly full model retraining (Decision Tree, Random Forest, Optuna XGBoost, PyTorch LSTM) every Sunday at 02:00
0 2 * * 0 cd /home/$USER/Aegis-Alpha && /home/$USER/Aegis-Alpha/.venv/bin/python main.py --mode online --train --export-offline-snapshot >> /home/$USER/Aegis-Alpha/logs/cron_retrain.log 2>&1
```

### Docker Execution (`python:3.12-slim`)

```bash
# Build image
docker build -t aegis-alpha:latest .

# Run offline reproduction mounting local data/ and models/
docker run --rm \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  aegis-alpha:latest python main.py --mode offline --train

# Run online incremental cycle inside container
docker run --rm \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  aegis-alpha:latest python main.py --mode online
```

### Broker and Notification Graceful Fallbacks
- **Binance Broker (`src/broker.py`):** If `BINANCE_API_KEY` or `BINANCE_API_SECRET` is empty (or `--execute-orders` is omitted), all signals execute cleanly in `DRY_RUN` mode with simulated order IDs, fees, and SL/TP levels. When keys are populated and `BINANCE_TESTNET=true`, orders route to Binance Spot Testnet via `exchange.set_sandbox_mode(True)`.
- **Telegram Notifier (`src/notifier.py`):** If `TELEGRAM_BOT_TOKEN` or `TELEGRAM_CHAT_ID` is empty, formatted trade alerts and portfolio performance summaries gracefully print to the console without error.

---

## 11. Repository Structure and Security Controls

```text
Aegis-Alpha/
|-- .env.example                  # Credential template (zero hardcoded secrets)
|-- .gitignore                    # Excludes data/, .env, *.db*, __pycache__/
|-- Dockerfile                    # python:3.12-slim non-root container image
|-- LICENSE                       # Repository license
|-- README.md                     # Full project documentation (zero emojis)
|-- main.py                       # CLI orchestrator (--mode online | offline)
|-- requirements.txt              # Pinned Python 3.12 dependencies
|-- data/                         # Local SQLite WAL DB & Parquet snapshots (gitignored)
|-- models/                       # Saved joblib/PyTorch models, metadata & reports
|-- notebooks/
|   |-- 01_eda.ipynb              # Target distribution, NaN audit & feature correlations
|   `-- 02_workflow_runner.ipynb  # End-to-end modular pipeline runner & latest trades
`-- src/
    |-- __init__.py
    |-- config.py                 # python-dotenv config & shared pipeline constants
    |-- data_fetcher.py           # ccxt 15m OHLCV + Fear & Greed (API/BS4/Playwright)
    |-- features.py               # 40 NUMERIC + 57 DUMMIES + TO_PREDICT engineering
    |-- train.py                  # DT, RF, Optuna/RandSearch XGB, PyTorch LSTM + rules
    |-- simulator.py              # Vector Sim 1/2, Exact itertuples Sim, Benchmark, Diagnostics
    |-- broker.py                 # ccxt Binance Spot Testnet & Dry-Run execution
    `-- notifier.py               # Telegram Bot API & Console fallback alerts
```

### Security Controls
- Zero hardcoded API keys, tokens, or secrets anywhere in the codebase.
- All credentials loaded strictly via `python-dotenv` and `os.getenv()` in [src/config.py](src/config.py).
- `.env`, `data/`, `*.db`, `*.db-wal`, and `*.db-shm` are excluded in [.gitignore](.gitignore).
