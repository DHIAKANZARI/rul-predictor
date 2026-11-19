"""REST API: send an engine's recent sensor history, get RUL + maintenance action.

    uvicorn rul.api:app --reload
"""
from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import COLUMNS
from .model import load, maintenance_decision, predict_last

app = FastAPI(title="RUL Predictor", version="1.0.0")


class Cycle(BaseModel):
    cycle: int
    op1: float = 0.0
    op2: float = 0.0
    op3: float = 100.0
    sensors: dict[str, float] = Field(..., description="s1..s21 readings")


class EngineHistory(BaseModel):
    engine_id: str
    cycles: list[Cycle] = Field(..., min_length=1)


@lru_cache(maxsize=1)
def model():
    path = os.environ.get("RUL_MODEL", "models/rul_model.joblib")
    if not os.path.exists(path):
        raise HTTPException(503, "model not trained yet: run `python -m rul.cli train`")
    return load(path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_info():
    return model()[1]


@app.post("/predict")
def predict(h: EngineHistory):
    rows = []
    for c in h.cycles:
        missing = [f"s{i}" for i in range(1, 22) if f"s{i}" not in c.sensors]
        if missing:
            raise HTTPException(422, f"cycle {c.cycle}: missing sensors {missing[:5]}")
        rows.append({"unit": 1, "cycle": c.cycle, "op1": c.op1, "op2": c.op2, "op3": c.op3, **c.sensors})
    df = pd.DataFrame(rows)[COLUMNS]
    rul = float(predict_last(model()[0], df)[0])
    d = maintenance_decision(rul)
    return {"engine_id": h.engine_id, "rul_cycles": round(rul, 1), "status": d.status, "action": d.message}
