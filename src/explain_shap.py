"""
SHAP explainability for the best tree model (XGBoost by default).

Produces:
  - global SHAP summary (beeswarm) + bar importance
  - SHAP dependence plot for the top predictor
  - SEASONAL mean |SHAP| breakdown (JJAS/OND/JF/MAM)  -> the project's novelty
  - REGIONAL mean |SHAP| breakdown across Indian sub-regions
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

import config as C
import data_prep as D

MODEL_NAME = "xgboost"   # tree model to explain
SHAP_SAMPLE = 40_000     # rows sampled for SHAP (test period)


def load_model():
    with open(C.MODELS_DIR / f"{MODEL_NAME}.pkl", "rb") as f:
        return pickle.load(f)


def main():
    model = load_model()
    df = D.add_subregion(D.cube_to_table(D.load_cube()))
    _, _, te = D.temporal_split(df)
    samp = te.sample(n=min(SHAP_SAMPLE, len(te)), random_state=C.RANDOM_STATE).reset_index(drop=True)
    X = samp[C.FEATURES]

    print(f"Computing SHAP on {len(X):,} test rows ...")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    abs_sv = np.abs(sv)

    # 1) global beeswarm
    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=len(C.FEATURES))
    plt.tight_layout(); plt.savefig(C.OUT_SHAP / "shap_summary_beeswarm.png", dpi=150); plt.close()

    # 2) global bar
    mean_abs = abs_sv.mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    plt.figure(figsize=(7, 4))
    plt.barh([C.FEATURES[i] for i in order][::-1], mean_abs[order][::-1], color="#2c7fb8")
    plt.xlabel("mean |SHAP| (mm/month)"); plt.title("Global predictor importance (SHAP)")
    plt.tight_layout(); plt.savefig(C.OUT_SHAP / "shap_global_bar.png", dpi=150); plt.close()

    # 3) dependence for top predictor
    top = C.FEATURES[order[0]]
    plt.figure()
    shap.dependence_plot(top, sv, X, show=False)
    plt.tight_layout(); plt.savefig(C.OUT_SHAP / f"shap_dependence_{top}.png", dpi=150); plt.close()

    # 4) seasonal mean|SHAP| table + heatmap
    seasonal = _grouped_importance(abs_sv, samp["season"], C.FEATURES, list(C.SEASONS.keys()))
    seasonal.to_csv(C.OUT_SHAP / "shap_by_season.csv")
    _heatmap(seasonal, "Mean |SHAP| by season", C.OUT_SHAP / "shap_seasonal_heatmap.png")

    # 5) regional mean|SHAP| table + heatmap
    regions = [r for r in C.SUBREGIONS if (samp["subregion"] == r).any()]
    regional = _grouped_importance(abs_sv, samp["subregion"], C.FEATURES, regions)
    regional.to_csv(C.OUT_SHAP / "shap_by_region.csv")
    _heatmap(regional, "Mean |SHAP| by sub-region", C.OUT_SHAP / "shap_regional_heatmap.png")

    print("Top predictors (global):",
          ", ".join(f"{C.FEATURES[i]}={mean_abs[i]:.2f}" for i in order[:5]))
    print("Saved SHAP figures + seasonal/regional CSVs to", C.OUT_SHAP)


def _grouped_importance(abs_sv, groups, feats, order):
    out = {}
    g = pd.Series(groups).reset_index(drop=True)
    for key in order:
        idx = (g == key).values
        if idx.sum() == 0:
            continue
        out[key] = abs_sv[idx].mean(axis=0)
    return pd.DataFrame(out, index=feats)


def _heatmap(tab, title, path):
    fig, ax = plt.subplots(figsize=(1.4 * tab.shape[1] + 3, 0.5 * tab.shape[0] + 2))
    im = ax.imshow(tab.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(tab.shape[1])); ax.set_xticklabels(tab.columns, rotation=30, ha="right")
    ax.set_yticks(range(tab.shape[0])); ax.set_yticklabels(tab.index)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            ax.text(j, i, f"{tab.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, label="mean |SHAP|"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


if __name__ == "__main__":
    main()
