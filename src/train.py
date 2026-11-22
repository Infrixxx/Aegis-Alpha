"""Multi-model training, hyperparameter tuning, and custom decision rules.

Phase 3 module for Aegis Alpha:
- Chronological time-series split (Train: 2020-2024, Validation: 2025, Test: 2026).
- Baseline models: DecisionTreeClassifier, RandomForestClassifier.
- Advanced models: Hyperparameter-tuned XGBClassifier (Optuna or RandomizedSearchCV)
  and PyTorch LSTM sequence classifier operating on per-symbol sliding windows.
- Custom high-probability decision rule (pred_custom_rule) combining model probability,
  Crypto Fear & Greed filter, volatility regime filter, and RSI confirmation.
- Persists trained models, scalers, metadata, and test predictions to models/.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    MODELS_DIR,
    TEST_START_ISO,
    TRAIN_END_ISO,
    VAL_END_ISO,
    VAL_START_ISO,
    ensure_directories,
)
from src.features import (
    DUMMIES,
    FEATURE_COLS,
    NUMERIC,
    TARGET_COL,
    TO_PREDICT,
    save_dataframe,
)

logger = logging.getLogger("aegis_alpha.train")

RANDOM_STATE = 42
LSTM_SEQ_LEN = 16  # 16 * 15m = 4 hours of sequential market context


@dataclass
class ChronologicalSplit:
    """Container holding strictly ordered Train, Validation, and Test DataFrames."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    split_mode: str


def chronological_split(
    df: pd.DataFrame,
    train_end_iso: str = TRAIN_END_ISO,
    val_start_iso: str = VAL_START_ISO,
    val_end_iso: str = VAL_END_ISO,
    test_start_iso: str = TEST_START_ISO,
) -> ChronologicalSplit:
    """Split dataset chronologically into Train (2020-2024), Val (2025), and Test (2026).

    Falls back to a strict chronological 70/15/15 timestamp split only when
    running on a short custom subset that does not span 2020-2026.
    """
    ordered = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    dt = pd.to_datetime(ordered["datetime"], utc=True)

    train_end = pd.Timestamp(train_end_iso)
    val_start = pd.Timestamp(val_start_iso)
    val_end = pd.Timestamp(val_end_iso)
    test_start = pd.Timestamp(test_start_iso)

    train_df = ordered[dt <= train_end].copy()
    val_df = ordered[(dt >= val_start) & (dt <= val_end)].copy()
    test_df = ordered[dt >= test_start].copy()

    if len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0:
        logger.info(
            "Applied canonical chronological split: train=%d (<=%s), val=%d (%s..%s), test=%d (>=%s).",
            len(train_df),
            train_end_iso,
            len(val_df),
            val_start_iso,
            val_end_iso,
            len(test_df),
            test_start_iso,
        )
        return ChronologicalSplit(
            train=train_df,
            val=val_df,
            test=test_df,
            split_mode="canonical_2020_2024_train_2025_val_2026_test",
        )

    logger.warning(
        "Dataset span does not cover all canonical 2020-2026 windows; "
        "applying chronological 70/15/15 timestamp fallback split."
    )
    unique_ts = np.sort(ordered["timestamp"].unique())
    if len(unique_ts) < 10:
        raise ValueError("Dataset has too few unique timestamps for chronological splitting.")

    idx_train = int(len(unique_ts) * 0.70)
    idx_val = int(len(unique_ts) * 0.85)
    ts_train_end = unique_ts[max(idx_train - 1, 0)]
    ts_val_end = unique_ts[max(idx_val - 1, idx_train)]

    train_df = ordered[ordered["timestamp"] <= ts_train_end].copy()
    val_df = ordered[
        (ordered["timestamp"] > ts_train_end) & (ordered["timestamp"] <= ts_val_end)
    ].copy()
    test_df = ordered[ordered["timestamp"] > ts_val_end].copy()

    return ChronologicalSplit(
        train=train_df,
        val=val_df,
        test=test_df,
        split_mode="fallback_chronological_70_15_15",
    )


def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.50,
    high_prob_threshold: float = 0.65,
) -> dict[str, float]:
    """Compute standard binary classification metrics plus high-confidence precision."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    y_hat = (y_p >= threshold).astype(int)
    y_high = (y_p > high_prob_threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.5
    except ValueError:
        auc = 0.5

    return {
        "roc_auc": round(auc, 5),
        "accuracy": round(float(accuracy_score(y_t, y_hat)), 5),
        "precision": round(float(precision_score(y_t, y_hat, zero_division=0)), 5),
        "recall": round(float(recall_score(y_t, y_hat, zero_division=0)), 5),
        "f1": round(float(f1_score(y_t, y_hat, zero_division=0)), 5),
        "precision_at_065": round(float(precision_score(y_t, y_high, zero_division=0)), 5),
        "signals_at_065": int(y_high.sum()),
    }


def evaluate_binary_signal(
    y_true: np.ndarray | pd.Series,
    signal: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Evaluate a discrete 0/1 trading rule signal against ground truth."""
    y_t = np.asarray(y_true, dtype=int)
    sig = np.asarray(signal, dtype=int)
    return {
        "signal_count": int(sig.sum()),
        "signal_rate": round(float(sig.mean()), 5),
        "precision": round(float(precision_score(y_t, sig, zero_division=0)), 5),
        "recall": round(float(recall_score(y_t, sig, zero_division=0)), 5),
        "f1": round(float(f1_score(y_t, sig, zero_division=0)), 5),
    }


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> DecisionTreeClassifier:
    """Train Baseline Model 1: Depth-regularized DecisionTreeClassifier."""
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    logger.info("Trained Baseline DecisionTreeClassifier (max_depth=10).")
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 150,
) -> RandomForestClassifier:
    """Train Baseline Model 2: Bagged RandomForestClassifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=50,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    logger.info("Trained Baseline RandomForestClassifier (n_estimators=%d).", n_estimators)
    return model


def train_tuned_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    tuning_method: str = "optuna",
    n_trials: int = 15,
) -> tuple[Any, dict[str, Any]]:
    """Train Advanced Model 1: Hyperparameter-tuned XGBClassifier.

    Supports Optuna Bayesian search (evaluating directly on the chronological
    validation set) or RandomizedSearchCV with TimeSeriesSplit.
    """
    from xgboost import XGBClassifier

    pos_count = max(int((y_train == 1).sum()), 1)
    neg_count = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = float(neg_count / pos_count)

    best_params: dict[str, Any] = {
        "n_estimators": 350,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 20,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    }

    use_optuna = tuning_method == "optuna"
    if use_optuna:
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("optuna not installed; falling back to RandomizedSearchCV.")
            use_optuna = False

    if use_optuna and n_trials > 0:
        import optuna

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 60),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            }
            clf = XGBClassifier(
                **params,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                early_stopping_rounds=25,
            )
            clf.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            val_prob = clf.predict_proba(X_val)[:, 1]
            return float(roc_auc_score(y_val, val_prob))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params.update(study.best_params)
        logger.info(
            "Optuna XGBoost tuning finished (%d trials): best val ROC-AUC=%.5f | params=%s",
            n_trials,
            study.best_value,
            best_params,
        )
    elif n_trials > 0:
        param_dist = {
            "n_estimators": [150, 250, 350, 450],
            "max_depth": [4, 5, 6, 7],
            "learning_rate": [0.02, 0.05, 0.08, 0.12],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [10, 25, 50],
            "reg_alpha": [0.0, 0.2, 1.0],
            "reg_lambda": [1.0, 2.5, 5.0],
        }
        base_xgb = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_dist,
            n_iter=min(n_trials, 20),
            scoring="roc_auc",
            cv=tscv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_params.update(search.best_params_)
        logger.info(
            "RandomizedSearchCV XGBoost tuning finished: best CV ROC-AUC=%.5f | params=%s",
            search.best_score_,
            best_params,
        )

    final_model = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        early_stopping_rounds=30,
    )
    final_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return final_model, best_params


def _build_lstm_class() -> Any:
    """Lazily define PyTorch CryptoLSTMClassifier to keep module import lightweight."""
    import torch
    import torch.nn as nn

    class CryptoLSTMClassifier(nn.Module):
        """Multi-layer LSTM sequence classifier for 15m cryptocurrency feature windows."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.25,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.dropout_rate = dropout

            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            last_step = self.norm(out[:, -1, :])
            return self.head(last_step).squeeze(-1)

    return CryptoLSTMClassifier


def _prepare_symbol_sequences(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: list[str],
    seq_len: int = LSTM_SEQ_LEN,
    include_target: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Build sliding windows strictly inside each symbol without cross-asset bleed.

    Returns (X_seq, y_seq, original_row_positions).
    """
    scaled_all = scaler.transform(df[feature_cols].to_numpy(dtype=np.float32))
    scaled_all = np.clip(scaled_all, -10.0, 10.0)

    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    pos_chunks: list[np.ndarray] = []

    positions = np.arange(len(df))
    symbols = df["symbol"].to_numpy()
    timestamps = df["timestamp"].to_numpy()
    targets = df[TARGET_COL].to_numpy(dtype=np.float32) if include_target and TARGET_COL in df.columns else None

    for sym in np.unique(symbols):
        mask = np.where(symbols == sym)[0]
        order = mask[np.argsort(timestamps[mask])]
        if len(order) < seq_len:
            continue

        sym_x = scaled_all[order]
        # Vectorized sliding window view: shape (n_windows, seq_len, n_features)
        windows = np.lib.stride_tricks.sliding_window_view(
            sym_x, window_shape=seq_len, axis=0
        )
        # sliding_window_view places window axis at the end: (N, F, T) -> transpose to (N, T, F)
        windows = np.transpose(windows, (0, 2, 1))
        end_positions = positions[order[seq_len - 1 :]]

        x_chunks.append(windows)
        pos_chunks.append(end_positions)
        if targets is not None:
            y_chunks.append(targets[order[seq_len - 1 :]])

    if not x_chunks:
        empty_x = np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.float32) if include_target else None
        return empty_x, empty_y, np.zeros((0,), dtype=np.int64)

    X_seq = np.concatenate(x_chunks, axis=0)
    pos_all = np.concatenate(pos_chunks, axis=0)
    y_seq = np.concatenate(y_chunks, axis=0) if targets is not None else None
    return X_seq, y_seq, pos_all


def predict_lstm_probabilities(
    model: Any,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_cols: list[str] = NUMERIC,
    seq_len: int = LSTM_SEQ_LEN,
    batch_size: int = 1024,
    fallback_prob: float = 0.5,
) -> np.ndarray:
    """Run batched LSTM inference and align predictions back to DataFrame row order."""
    import torch

    probs = np.full(len(df), fallback_prob, dtype=np.float64)
    X_seq, _, row_positions = _prepare_symbol_sequences(
        df=df,
        scaler=scaler,
        feature_cols=feature_cols,
        seq_len=seq_len,
        include_target=False,
    )
    if len(X_seq) == 0:
        return probs

    device = next(model.parameters()).device
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X_seq), batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(X_seq[start : start + batch_size], dtype=np.float32)
            ).to(device)
            logits = model(batch)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(batch_probs)

    probs[row_positions] = np.concatenate(preds, axis=0)
    return probs


def train_pytorch_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str] = NUMERIC,
    seq_len: int = LSTM_SEQ_LEN,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.25,
    epochs: int = 8,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    max_train_windows: int = 150000,
) -> tuple[Any, StandardScaler, dict[str, Any]]:
    """Train Advanced Model 2: PyTorch LSTM sequence classifier."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fit scaler strictly on training split to prevent data leakage
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].to_numpy(dtype=np.float32))

    X_train_seq, y_train_seq, _ = _prepare_symbol_sequences(
        df=train_df, scaler=scaler, feature_cols=feature_cols, seq_len=seq_len, include_target=True
    )
    X_val_seq, y_val_seq, _ = _prepare_symbol_sequences(
        df=val_df, scaler=scaler, feature_cols=feature_cols, seq_len=seq_len, include_target=True
    )

    if len(X_train_seq) == 0 or len(X_val_seq) == 0:
        raise ValueError("Insufficient sequence windows to train PyTorch LSTM.")

    # Optional recency cap on huge >800k-row training splits to keep CPU training tractable
    if len(X_train_seq) > max_train_windows:
        logger.info(
            "Subsampling most recent %d LSTM training windows out of %d for training speed.",
            max_train_windows,
            len(X_train_seq),
        )
        X_train_seq = X_train_seq[-max_train_windows:]
        y_train_seq = y_train_seq[-max_train_windows:]

    train_dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X_train_seq, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y_train_seq, dtype=np.float32)),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    CryptoLSTMClassifier = _build_lstm_class()
    model = CryptoLSTMClassifier(
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    pos_rate = float(np.clip(y_train_seq.mean(), 1e-4, 1.0 - 1e-4))
    pos_weight = torch.tensor([(1.0 - pos_rate) / pos_rate], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_auc = -1.0
    best_state_dict: dict[str, Any] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        val_probs = predict_lstm_probabilities(
            model=model,
            scaler=scaler,
            df=val_df,
            feature_cols=feature_cols,
            seq_len=seq_len,
        )
        val_auc = float(roc_auc_score(val_df[TARGET_COL].to_numpy(), val_probs)) if len(
            np.unique(val_df[TARGET_COL])
        ) > 1 else 0.5
        avg_loss = running_loss / max(n_batches, 1)
        scheduler.step(val_auc)
        history.append({"epoch": epoch, "train_loss": round(avg_loss, 5), "val_auc": round(val_auc, 5)})
        logger.info(
            "LSTM Epoch %d/%d | train_loss=%.5f | val_roc_auc=%.5f | device=%s",
            epoch,
            epochs,
            avg_loss,
            val_auc,
            device,
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.to(torch.device("cpu"))

    lstm_config = {
        "input_dim": len(feature_cols),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "seq_len": seq_len,
        "feature_cols": list(feature_cols),
        "best_val_auc": round(best_val_auc, 5),
        "history": history,
    }
    return model, scaler, lstm_config


def apply_custom_decision_rules(
    df: pd.DataFrame,
    prob_col: str = "pred_prob_ensemble",
    baseline_prob_col: str = "pred_prob_xgb",
    baseline_threshold: float = 0.55,
    custom_prob_threshold: float = 0.65,
    max_fear_greed: float = 45.0,
    max_rsi: float = 70.0,
) -> pd.DataFrame:
    """Attach baseline and custom high-probability trading decision rules.

    Custom rule (pred_custom_rule):
    - Model probability > 0.65
    - Crypto Fear & Greed Index < 45 (contrarian Fear / Extreme Fear zone)
    - Volatility regime is not 'extreme'
    - RSI-14 < 70 (avoid overbought chase entries)
    """
    out = df.copy()
    base_col = baseline_prob_col if baseline_prob_col in out.columns else prob_col
    out["pred_baseline"] = (out[base_col] >= baseline_threshold).astype(int)

    prob_cond = out[prob_col] > custom_prob_threshold
    fg_cond = out["fg_value"] < max_fear_greed
    vol_cond = out["volatility_regime"] != "extreme"
    rsi_cond = out["rsi_14"] < max_rsi

    out["pred_custom_rule"] = (prob_cond & fg_cond & vol_cond & rsi_cond).astype(int)
    return out


def save_lstm_checkpoint(model: Any, lstm_config: dict[str, Any], path: Path) -> None:
    """Save PyTorch LSTM weights and architecture config."""
    import torch

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": lstm_config,
        },
        path,
    )


def load_lstm_checkpoint(path: Path) -> tuple[Any, dict[str, Any]]:
    """Restore PyTorch LSTM classifier from checkpoint."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    CryptoLSTMClassifier = _build_lstm_class()
    model = CryptoLSTMClassifier(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg


def train_all_models(
    df: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
    tuning_method: str = "optuna",
    tuning_trials: int = 12,
    train_lstm: bool = True,
    lstm_epochs: int = 6,
    rf_estimators: int = 150,
) -> dict[str, Any]:
    """Train all 4 models, evaluate on Validation and Test, apply rules, and save artifacts."""
    ensure_directories()
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = chronological_split(df)
    X_train, y_train = split.train[FEATURE_COLS], split.train[TARGET_COL]
    X_val, y_val = split.val[FEATURE_COLS], split.val[TARGET_COL]
    X_test, y_test = split.test[FEATURE_COLS], split.test[TARGET_COL]

    # 1. Lecture Baseline 1: DecisionTreeClassifier
    dt_model = train_decision_tree(X_train, y_train)
    dt_val_prob = dt_model.predict_proba(X_val)[:, 1]
    dt_test_prob = dt_model.predict_proba(X_test)[:, 1]
    joblib.dump(dt_model, out_dir / "decision_tree.joblib")

    # 2. Lecture Baseline 2: RandomForestClassifier
    rf_model = train_random_forest(X_train, y_train, n_estimators=rf_estimators)
    rf_val_prob = rf_model.predict_proba(X_val)[:, 1]
    rf_test_prob = rf_model.predict_proba(X_test)[:, 1]
    joblib.dump(rf_model, out_dir / "random_forest.joblib")

    # 3. Advanced Model 1: Tuned XGBClassifier
    xgb_model, xgb_best_params = train_tuned_xgboost(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_method=tuning_method,
        n_trials=tuning_trials,
    )
    xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]
    joblib.dump(xgb_model, out_dir / "xgboost_tuned.joblib")

    # 4. Advanced Model 2: PyTorch LSTM Sequence Classifier
    lstm_trained = False
    lstm_cfg: dict[str, Any] = {}
    lstm_val_prob: np.ndarray | None = None
    lstm_test_prob: np.ndarray | None = None

    if train_lstm:
        try:
            lstm_model, lstm_scaler, lstm_cfg = train_pytorch_lstm(
                train_df=split.train,
                val_df=split.val,
                feature_cols=NUMERIC,
                seq_len=LSTM_SEQ_LEN,
                epochs=lstm_epochs,
            )
            lstm_val_prob = predict_lstm_probabilities(
                model=lstm_model,
                scaler=lstm_scaler,
                df=split.val,
                feature_cols=NUMERIC,
                seq_len=LSTM_SEQ_LEN,
            )
            lstm_test_prob = predict_lstm_probabilities(
                model=lstm_model,
                scaler=lstm_scaler,
                df=split.test,
                feature_cols=NUMERIC,
                seq_len=LSTM_SEQ_LEN,
            )
            save_lstm_checkpoint(lstm_model, lstm_cfg, out_dir / "lstm_classifier.pt")
            joblib.dump(lstm_scaler, out_dir / "lstm_scaler.joblib")
            lstm_trained = True
        except Exception as exc:
            logger.warning("PyTorch LSTM training skipped/failed (%s); using XGBoost ensemble.", exc)

    # Assemble Validation and Test prediction DataFrames
    val_preds = split.val.copy()
    test_preds = split.test.copy()

    val_preds["pred_prob_dt"] = dt_val_prob
    val_preds["pred_prob_rf"] = rf_val_prob
    val_preds["pred_prob_xgb"] = xgb_val_prob

    test_preds["pred_prob_dt"] = dt_test_prob
    test_preds["pred_prob_rf"] = rf_test_prob
    test_preds["pred_prob_xgb"] = xgb_test_prob

    if lstm_trained and lstm_val_prob is not None and lstm_test_prob is not None:
        val_preds["pred_prob_lstm"] = lstm_val_prob
        test_preds["pred_prob_lstm"] = lstm_test_prob
        val_preds["pred_prob_ensemble"] = 0.65 * xgb_val_prob + 0.35 * lstm_val_prob
        test_preds["pred_prob_ensemble"] = 0.65 * xgb_test_prob + 0.35 * lstm_test_prob
    else:
        val_preds["pred_prob_lstm"] = np.nan
        test_preds["pred_prob_lstm"] = np.nan
        val_preds["pred_prob_ensemble"] = xgb_val_prob
        test_preds["pred_prob_ensemble"] = xgb_test_prob

    val_preds = apply_custom_decision_rules(val_preds)
    test_preds = apply_custom_decision_rules(test_preds)

    metrics: dict[str, Any] = {
        "split_mode": split.split_mode,
        "row_counts": {
            "train": int(len(split.train)),
            "val": int(len(split.val)),
            "test": int(len(split.test)),
        },
        "validation": {
            "decision_tree": evaluate_predictions(y_val, dt_val_prob),
            "random_forest": evaluate_predictions(y_val, rf_val_prob),
            "xgboost_tuned": evaluate_predictions(y_val, xgb_val_prob),
            "ensemble": evaluate_predictions(y_val, val_preds["pred_prob_ensemble"]),
            "rule_baseline": evaluate_binary_signal(y_val, val_preds["pred_baseline"]),
            "rule_custom": evaluate_binary_signal(y_val, val_preds["pred_custom_rule"]),
        },
        "test": {
            "decision_tree": evaluate_predictions(y_test, dt_test_prob),
            "random_forest": evaluate_predictions(y_test, rf_test_prob),
            "xgboost_tuned": evaluate_predictions(y_test, xgb_test_prob),
            "ensemble": evaluate_predictions(y_test, test_preds["pred_prob_ensemble"]),
            "rule_baseline": evaluate_binary_signal(y_test, test_preds["pred_baseline"]),
            "rule_custom": evaluate_binary_signal(y_test, test_preds["pred_custom_rule"]),
        },
        "xgboost_best_params": xgb_best_params,
        "lstm_trained": lstm_trained,
        "lstm_config": lstm_cfg,
        "feature_counts": {
            "TO_PREDICT": len(TO_PREDICT),
            "NUMERIC": len(NUMERIC),
            "DUMMIES": len(DUMMIES),
            "TOTAL": len(FEATURE_COLS),
        },
    }

    if lstm_trained and lstm_val_prob is not None and lstm_test_prob is not None:
        metrics["validation"]["lstm"] = evaluate_predictions(y_val, lstm_val_prob)
        metrics["test"]["lstm"] = evaluate_predictions(y_test, lstm_test_prob)

    metadata_path = out_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Persist test predictions for downstream simulation and notebooks
    keep_cols = [
        c
        for c in [
            "symbol",
            "timestamp",
            "datetime_utc",
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi_14",
            "atr_pct_14",
            "vol_24h",
            "ema_50_200_ratio",
            "vwap_ratio",
            "fg_value",
            "sentiment_regime",
            "volatility_regime",
            "future_return_15m",
            "future_return_4h",
            TARGET_COL,
            "pred_prob_dt",
            "pred_prob_rf",
            "pred_prob_xgb",
            "pred_prob_lstm",
            "pred_prob_ensemble",
            "pred_baseline",
            "pred_custom_rule",
        ]
        if c in test_preds.columns
    ]
    preds_path = save_dataframe(test_preds[keep_cols], out_dir / "predictions_test.parquet")
    logger.info("Saved model metadata to %s and test predictions to %s.", metadata_path, preds_path)

    return {
        "metrics": metrics,
        "val_predictions": val_preds,
        "test_predictions": test_preds,
        "predictions_path": str(preds_path),
    }


def predict_with_saved_models(
    df: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
) -> pd.DataFrame:
    """Load persisted model artifacts and generate predictions + custom decision rules."""
    out_dir = Path(models_dir)
    xgb_path = out_dir / "xgboost_tuned.joblib"
    if not xgb_path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {xgb_path}. Run training first."
        )

    out = df.copy()
    X = out[FEATURE_COLS]

    xgb_model = joblib.load(xgb_path)
    out["pred_prob_xgb"] = xgb_model.predict_proba(X)[:, 1]

    dt_path = out_dir / "decision_tree.joblib"
    if dt_path.exists():
        dt_model = joblib.load(dt_path)
        out["pred_prob_dt"] = dt_model.predict_proba(X)[:, 1]

    rf_path = out_dir / "random_forest.joblib"
    if rf_path.exists():
        rf_model = joblib.load(rf_path)
        out["pred_prob_rf"] = rf_model.predict_proba(X)[:, 1]

    lstm_pt = out_dir / "lstm_classifier.pt"
    lstm_scaler_path = out_dir / "lstm_scaler.joblib"
    if lstm_pt.exists() and lstm_scaler_path.exists():
        try:
            lstm_model, lstm_cfg = load_lstm_checkpoint(lstm_pt)
            lstm_scaler = joblib.load(lstm_scaler_path)
            out["pred_prob_lstm"] = predict_lstm_probabilities(
                model=lstm_model,
                scaler=lstm_scaler,
                df=out,
                feature_cols=lstm_cfg.get("feature_cols", NUMERIC),
                seq_len=int(lstm_cfg.get("seq_len", LSTM_SEQ_LEN)),
            )
            out["pred_prob_ensemble"] = 0.65 * out["pred_prob_xgb"] + 0.35 * out["pred_prob_lstm"]
        except Exception as exc:
            logger.warning("Failed to run LSTM inference (%s); falling back to XGBoost.", exc)
            out["pred_prob_ensemble"] = out["pred_prob_xgb"]
    else:
        out["pred_prob_ensemble"] = out["pred_prob_xgb"]

    out = apply_custom_decision_rules(out)
    return out
