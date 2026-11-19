"""Load the NASA C-MAPSS turbofan degradation dataset and build features.

Each engine starts healthy and degrades until failure. Every row is one
operating cycle with 3 operational settings and 21 sensor readings.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/"
COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
# Sensors that carry degradation signal in FD001 (the others are constant or noise).
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
RUL_CAP = 125  # early in life the RUL is not observable; a plateau is standard practice
WINDOW = 10


def _download(name: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    if not path.exists():
        urllib.request.urlretrieve(BASE_URL + name, path)
    return path


def read_cycles(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS
    return df


def load_fd001(cache: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return (train, test, true RUL at the last cycle of each test engine)."""
    cache = Path(cache)
    train = read_cycles(_download("train_FD001.txt", cache))
    test = read_cycles(_download("test_FD001.txt", cache))
    rul = pd.read_csv(_download("RUL_FD001.txt", cache), header=None).iloc[:, 0].to_numpy()
    return train, test, rul


def add_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Label each training cycle with its remaining cycles before failure (capped)."""
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = np.minimum(last - out["cycle"], cap)
    return out


def make_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Rolling mean / std / slope of each useful sensor, per engine."""
    df = df.sort_values(["unit", "cycle"]).copy()
    g = df.groupby("unit")
    feats = {"cycle": df["cycle"]}
    for s in SENSORS:
        roll = g[s].rolling(window, min_periods=1)
        feats[f"{s}_mean"] = roll.mean().reset_index(level=0, drop=True)
        feats[f"{s}_std"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        feats[f"{s}_delta"] = df[s] - g[s].transform("first")
    out = pd.DataFrame(feats, index=df.index)
    out.insert(0, "unit", df["unit"])
    return out


def last_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features at the most recent cycle of each engine (what we predict on)."""
    f = make_features(df)
    return f.groupby("unit").tail(1).set_index("unit")


FEATURE_COLUMNS = ["cycle"] + [f"{s}_{k}" for s in SENSORS for k in ("mean", "std", "delta")]
