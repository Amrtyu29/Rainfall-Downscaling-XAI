"""
Extreme-rainfall extension, steps 2-3: classify and EXPLAIN flood-level days.

  target   : IMD "heavy rain" days — rainfall >= 64.5 mm/day (official IMD
             category), per 0.25-degree cell, monsoon season (JJAS) 2000-2023
  features : 7 daily NCEP predictors (regridded to 0.25) + location + month
  model    : XGBoost classifier, class-weighted (heavy days are ~1% of rows)
  split    : train 2000-2015, test 2016-2023 (strictly out-of-sample years)
  XAI      : SHAP — what drives flood-level rainfall, nationally, per cell,
             and versus ordinary rain days.

Outputs in outputs/extremes/.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import xgboost as xgb
import shap
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import config as C
import mapstyle as M
from driver_map import PHYS, LABELS, COLORS

DATA = C.ROOT / "data_daily" / "ncep_daily_jjas_2000_2023.nc"
IMD = C.ROOT / "data" / "raw" / "imd" / "imd_rainfall_0.25deg_1980_2023.nc"
OUT = C.ROOT / "outputs" / "extremes"
OUT.mkdir(parents=True, exist_ok=True)

HEAVY = 64.5              # mm/day — IMD "heavy rainfall" threshold
TRAIN_YEARS = range(2000, 2016)
TEST_YEARS = range(2016, 2024)
NEG_FRACTION = 0.15       # negatives kept for training (positives all kept)
FEATURES = PHYS + ["lat", "lon", "month_sin", "month_cos"]
SHAP_SAMPLE = 300_000


def build_year(pred_ds, rain, year):
    """One year's (day, cell) table over land cells."""
    p = pred_ds.sel(time=pred_ds.time.dt.year == year)
    r = rain.sel(time=rain.time.dt.year == year)
    p = p.sel(time=p.time.dt.month.isin([6, 7, 8, 9]))
    r = r.sel(time=r.time.dt.month.isin([6, 7, 8, 9]))
    # align calendars (NCEP and IMD both daily)
    common = np.intersect1d(p.time.values, r.time.values)
    p, r = p.sel(time=common), r.sel(time=common)

    df = p.to_dataframe().reset_index()
    df["rain"] = r.to_dataframe().reset_index()["rainfall"].values
    df = df.dropna(subset=["rain"] + PHYS)
    df["heavy"] = (df["rain"] >= HEAVY).astype(int)
    df["month"] = pd.to_datetime(df["time"]).dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    return df


def load_tables():
    rain = xr.open_dataset(IMD)["rainfall"]
    pred = xr.open_dataset(DATA)
    # regrid smooth large-scale predictors to the IMD 0.25-degree grid
    pred = pred.interp(lat=rain.lat, lon=rain.lon, method="linear")

    train_parts, test_parts = [], []
    for y in list(TRAIN_YEARS) + list(TEST_YEARS):
        df = build_year(pred, rain, y)
        if y in TRAIN_YEARS:
            pos = df[df.heavy == 1]
            neg = df[df.heavy == 0].sample(frac=NEG_FRACTION,
                                           random_state=C.RANDOM_STATE)
            train_parts.append(pd.concat([pos, neg]))
        else:
            test_parts.append(df)
        print(f"  {y}: {len(df):,} rows, {df.heavy.mean()*100:.2f}% heavy", flush=True)
    return pd.concat(train_parts, ignore_index=True), \
           pd.concat(test_parts, ignore_index=True)


def main():
    print("Building daily tables ...", flush=True)
    train, test = load_tables()
    print(f"train {len(train):,} rows ({train.heavy.mean()*100:.1f}% heavy after sampling)")
    print(f"test  {len(test):,} rows ({test.heavy.mean()*100:.2f}% heavy, natural rate)")

    spw = (train.heavy == 0).sum() / (train.heavy == 1).sum()
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        n_jobs=-1, random_state=C.RANDOM_STATE, eval_metric="aucpr")
    print("Training classifier ...", flush=True)
    clf.fit(train[FEATURES].values, train["heavy"].values)

    print("Scoring full test years ...", flush=True)
    prob = clf.predict_proba(test[FEATURES].values)[:, 1]
    roc = roc_auc_score(test["heavy"], prob)
    pr = average_precision_score(test["heavy"], prob)
    ths = np.quantile(prob, 1 - test["heavy"].mean())      # rate-matched threshold
    f1 = f1_score(test["heavy"], prob >= ths)
    base = test["heavy"].mean()
    pd.DataFrame([{"ROC_AUC": roc, "PR_AUC": pr, "F1_rate_matched": f1,
                   "base_rate": base, "n_test": len(test)}]) \
      .to_csv(OUT / "classifier_metrics.csv", index=False)
    print(f"ROC-AUC={roc:.3f}  PR-AUC={pr:.3f} (base rate {base*100:.2f}%)  "
          f"F1={f1:.3f}")

    import pickle
    with open(C.MODELS_DIR / "extreme_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    # ---------------- SHAP: what drives flood-level rain? ----------------
    print("SHAP on test sample ...", flush=True)
    samp = test.sample(n=min(SHAP_SAMPLE, len(test)), random_state=C.RANDOM_STATE)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(samp[FEATURES])
    absv = pd.DataFrame(np.abs(sv), columns=FEATURES)[PHYS]

    # 1) beeswarm
    plt.figure()
    shap.summary_plot(sv, samp[FEATURES], show=False, max_display=len(FEATURES))
    plt.title("What drives heavy-rain days (>=64.5 mm/day)?", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "shap_extremes_beeswarm.png", dpi=150)
    plt.close()

    # 2) extreme vs ordinary rain days: mean |SHAP| comparison
    wet = (samp["rain"].values >= 1) & (samp["rain"].values < HEAVY)
    hvy = samp["heavy"].values == 1
    comp = pd.DataFrame({
        "ordinary rain day": absv[wet].mean(),
        "heavy rain day": absv[hvy].mean(),
    })
    comp.to_csv(OUT / "shap_extreme_vs_ordinary.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    comp.sort_values("heavy rain day").plot.barh(ax=ax, color=["#92c5de", "#b2182b"])
    ax.set_xlabel("mean |SHAP| (log-odds of a heavy-rain day)")
    ax.set_title("Drivers: heavy vs ordinary rain days", fontweight="bold")
    plt.tight_layout()
    M.save(fig, OUT / "shap_extreme_vs_ordinary.png")

    # 3) extreme-driver map of India
    cell = absv.copy()
    cell[["lat", "lon"]] = samp[["lat", "lon"]].values
    dom = cell.groupby(["lat", "lon"])[PHYS].mean().idxmax(axis=1) \
              .rename("driver").reset_index()
    codes = {p: i for i, p in enumerate(PHYS)}
    piv = dom.assign(code=dom["driver"].map(codes)) \
             .pivot(index="lat", columns="lon", values="code")
    fig = plt.figure(figsize=(9, 8.5))
    ax = M.india_axes(fig)
    mesh = ax.pcolormesh(piv.columns.values.astype(float),
                         piv.index.values.astype(float),
                         piv.values, cmap=ListedColormap(COLORS),
                         vmin=-0.5, vmax=len(PHYS) - 0.5,
                         shading="nearest", transform=M.ccrs.PlateCarree())
    M.clip_to_india(ax, mesh)
    M.draw_boundaries(ax)
    M.gridlines(ax)
    ax.set_title("Dominant driver of HEAVY rainfall days (>=64.5 mm/day)\n"
                 "JJAS 2016-2023, SHAP on daily classifier", pad=10)
    handles = [Patch(facecolor=COLORS[i], edgecolor="k", linewidth=0.3,
                     label=LABELS[p]) for i, p in enumerate(PHYS)]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.04))
    M.save(fig, OUT / "extreme_driver_map.png")

    print("Extreme-rainfall analysis complete ->", OUT)


if __name__ == "__main__":
    main()
