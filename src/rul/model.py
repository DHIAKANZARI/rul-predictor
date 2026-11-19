"""Model training, evaluation and maintenance decision rule."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import FEATURE_COLUMNS, add_rul, last_cycle_features, make_features


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric PHM08 score: predicting too late is penalised more than too early."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def train(train_df, seed: int = 0) -> HistGradientBoostingRegressor:
    labelled = add_rul(train_df)
    X = make_features(labelled)[FEATURE_COLUMNS]
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed)
    model.fit(X, labelled.loc[X.index, "rul"])
    return model


def predict_last(model, df) -> np.ndarray:
    X = last_cycle_features(df)[FEATURE_COLUMNS]
    return np.clip(model.predict(X), 0, None)


@dataclass
class Decision:
    status: str
    message: str


def maintenance_decision(rul: float, urgent: int = 20, plan: int = 50) -> Decision:
    """Turn a RUL estimate into an action for the maintenance team."""
    if rul <= urgent:
        return Decision("URGENT", f"~{rul:.0f} cycles left: ground the engine and inspect now")
    if rul <= plan:
        return Decision("PLAN", f"~{rul:.0f} cycles left: schedule maintenance at the next slot")
    return Decision("OK", f"~{rul:.0f} cycles left: no action needed")


def save(model, path: str | Path, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, path)


def load(path: str | Path):
    b = joblib.load(path)
    return b["model"], b["meta"]
