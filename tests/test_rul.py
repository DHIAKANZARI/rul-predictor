import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from rul.data import COLUMNS, FEATURE_COLUMNS, add_rul, make_features
from rul.model import maintenance_decision, nasa_score, predict_last, rmse, save, train


def fake_fleet(n_units=12, seed=0):
    """Synthetic engines: sensors drift linearly towards failure."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        life = rng.integers(120, 220)
        for c in range(1, life + 1):
            wear = c / life
            r = {"unit": u, "cycle": c, "op1": 0.0, "op2": 0.0, "op3": 100.0}
            for i in range(1, 22):
                r[f"s{i}"] = 500 + i + 10 * wear * (1 if i % 2 else -1) + rng.normal(0, 0.3)
            rows.append(r)
    return pd.DataFrame(rows)[COLUMNS]


def test_rul_labels_capped_and_end_at_zero():
    df = add_rul(fake_fleet(3))
    assert df["rul"].max() <= 125
    assert (df.groupby("unit")["rul"].min() == 0).all()


def test_features_columns():
    f = make_features(fake_fleet(2))
    assert set(FEATURE_COLUMNS) <= set(f.columns)
    assert not f[FEATURE_COLUMNS].isna().any().any()


def test_scores():
    assert rmse([10, 20], [10, 20]) == 0
    late, early = nasa_score([50], [60]), nasa_score([50], [40])
    assert late > early > 0  # being late costs more than being early


def test_decision_rule():
    assert maintenance_decision(5).status == "URGENT"
    assert maintenance_decision(35).status == "PLAN"
    assert maintenance_decision(120).status == "OK"


def test_model_learns_degradation():
    fleet = fake_fleet(14)
    model = train(fleet[fleet.unit <= 10])
    test = fleet[fleet.unit > 10]
    cut = test[test.cycle <= test.groupby("unit").cycle.transform("max") - 30]
    pred = predict_last(model, cut)
    assert np.all(np.abs(pred - 30) < 25)


def test_api(tmp_path, monkeypatch):
    from rul import api
    fleet = fake_fleet(8)
    path = tmp_path / "m.joblib"
    save(train(fleet), path, {"rmse": 0})
    monkeypatch.setenv("RUL_MODEL", str(path)); api.model.cache_clear()
    c = TestClient(api.app)
    eng = fleet[fleet.unit == 1].head(60)
    body = {"engine_id": "E1", "cycles": [{"cycle": int(r.cycle), "sensors": {f"s{i}": float(r[f's{i}']) for i in range(1, 22)}} for _, r in eng.iterrows()]}
    r = c.post("/predict", json=body)
    assert r.status_code == 200 and r.json()["status"] in {"OK", "PLAN", "URGENT"}
    body["cycles"][0]["sensors"].pop("s5")
    assert c.post("/predict", json=body).status_code == 422
