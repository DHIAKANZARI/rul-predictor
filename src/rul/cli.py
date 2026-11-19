"""python -m rul.cli train | evaluate"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import load_fd001
from .model import maintenance_decision, nasa_score, predict_last, rmse, save, train

MODEL = "models/rul_model.joblib"


def cmd_train(args):
    tr, te, true_rul = load_fd001(args.data)
    model = train(tr)
    pred = predict_last(model, te)
    capped = np.minimum(true_rul, 125)
    metrics = {
        "dataset": "NASA C-MAPSS FD001",
        "test_engines": int(len(true_rul)),
        "rmse": round(rmse(capped, pred), 2),
        "nasa_score": round(nasa_score(capped, pred), 1),
        "within_20_cycles": round(float(np.mean(np.abs(pred - capped) <= 20)), 3),
    }
    # How well does the decision rule catch engines that really need action?
    urgent_true = capped <= 20
    urgent_pred = np.array([maintenance_decision(p).status == "URGENT" for p in pred])
    metrics["urgent_recall"] = round(float((urgent_pred & urgent_true).sum() / max(urgent_true.sum(), 1)), 3)
    metrics["urgent_precision"] = round(float((urgent_pred & urgent_true).sum() / max(urgent_pred.sum(), 1)), 3)
    save(model, args.model, metrics)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2))

    order = np.argsort(capped)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(capped[order], label="True RUL", lw=2)
    ax.plot(pred[order], label="Predicted RUL", lw=1.5, alpha=.85)
    ax.axhline(20, ls="--", c="crimson", lw=1, label="Urgent threshold")
    ax.set_xlabel("Test engines (sorted by true RUL)"); ax.set_ylabel("Cycles left")
    ax.set_title(f"FD001 test set — RMSE {metrics['rmse']}")
    ax.legend(); fig.tight_layout(); fig.savefig("docs/predictions.png", dpi=120)
    print(json.dumps(metrics, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser("rul")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", default="data"); t.add_argument("--model", default=MODEL)
    t.set_defaults(func=cmd_train)
    a = p.parse_args(argv); a.func(a)


if __name__ == "__main__":
    main()
