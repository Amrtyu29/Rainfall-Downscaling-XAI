"""
Train and compare downscaling models:
    - Linear Regression (baseline, ~ traditional statistical downscaling)
    - Random Forest
    - XGBoost
    - LightGBM

Trains on a random subsample of the (large) training set, evaluates on the
FULL test set. Saves models, a metrics table, and test-set predictions.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

import config as C
import data_prep as D

TRAIN_SAMPLE = 500_000   # rows used to fit the tree models (keeps RF tractable)


def metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    pcc = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Bias": bias, "PCC": pcc}


def build_models():
    return {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=20, n_jobs=-1,
            random_state=C.RANDOM_STATE),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=C.RANDOM_STATE),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=600, max_depth=-1, num_leaves=63,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=C.RANDOM_STATE, verbose=-1),
    }


def main():
    print("Loading cube + building table ...")
    df = D.add_subregion(D.cube_to_table(D.load_cube()))
    tr, va, te = D.temporal_split(df)

    tr_s = tr.sample(n=min(TRAIN_SAMPLE, len(tr)), random_state=C.RANDOM_STATE)
    print(f"Training on {len(tr_s):,} sampled rows; testing on {len(te):,} rows.")

    Xtr, ytr = tr_s[C.FEATURES].values, tr_s[C.TARGET].values
    Xte, yte = te[C.FEATURES].values, te[C.TARGET].values

    rows = []
    preds = {}
    for name, model in build_models().items():
        print(f"  -> fitting {name} ...")
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)
        m = metrics(yte, yp)
        m["Model"] = name
        rows.append(m)
        preds[name] = yp
        with open(C.MODELS_DIR / f"{name.lower()}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"     {name:12s} RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  "
              f"R2={m['R2']:.3f}  Bias={m['Bias']:+.2f}  PCC={m['PCC']:.3f}")

    # metrics table
    mt = pd.DataFrame(rows)[["Model", "RMSE", "MAE", "R2", "Bias", "PCC"]]
    mt = mt.sort_values("R2", ascending=False).reset_index(drop=True)
    mt.to_csv(C.OUT_PRED / "model_metrics.csv", index=False)
    print("\n=== Test-set metrics (mm/month) ===")
    print(mt.to_string(index=False))

    # persist test predictions + keys for downstream validation / SHAP
    out = te[["time", "lat", "lon", "month", "season", "subregion", C.TARGET]].copy()
    for name, yp in preds.items():
        out[f"pred_{name}"] = yp
    out.to_parquet(C.OUT_PRED / "test_predictions.parquet", index=False)
    with open(C.MODELS_DIR / "feature_names.pkl", "wb") as f:
        pickle.dump(C.FEATURES, f)
    print(f"\nSaved models, metrics, and {len(out):,} test predictions.")


if __name__ == "__main__":
    main()
