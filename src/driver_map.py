"""
The project's novelty figure: a "rainfall driver map of India".

For every 0.25-degree grid cell we compute the mean |SHAP| of each PHYSICAL
predictor (u850, v850, q850, z500, rh850, mslp, t2m — location/season encodings
excluded) and colour the cell by the predictor the model relies on most.

Outputs
  outputs/shap/driver_map_annual.png            (+ JJAS / DJF panels)
  outputs/shap/driver_map_cellwise.csv          per-cell dominant driver + shares
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import shap

import config as C
import data_prep as D
import mapstyle as M

MODEL_NAME = "xgboost"
SHAP_SAMPLE = 400_000                       # big sample for dense cell coverage
PHYS = C.PREDICTORS                          # physical drivers only
LABELS = {
    "q850": "Specific humidity (850 hPa)",
    "rh850": "Relative humidity (850 hPa)",
    "u850": "Zonal wind (850 hPa)",
    "v850": "Meridional wind (850 hPa)",
    "mslp": "Sea-level pressure",
    "t2m": "2 m temperature",
    "z500": "Geopotential height (500 hPa)",
}
COLORS = ["#1f78b4", "#a6cee3", "#33a02c", "#b2df8a",
          "#e31a1c", "#ff7f00", "#6a3d9a"]


def compute_shap():
    with open(C.MODELS_DIR / f"{MODEL_NAME}.pkl", "rb") as f:
        model = pickle.load(f)
    df = D.cube_to_table(D.load_cube())
    _, _, te = D.temporal_split(df)
    samp = te.sample(n=min(SHAP_SAMPLE, len(te)),
                     random_state=C.RANDOM_STATE).reset_index(drop=True)
    print(f"Computing SHAP for {len(samp):,} test rows ...")
    sv = shap.TreeExplainer(model).shap_values(samp[C.FEATURES])
    abs_sv = pd.DataFrame(np.abs(sv), columns=C.FEATURES)[PHYS]
    abs_sv[["lat", "lon"]] = samp[["lat", "lon"]]
    abs_sv["season"] = samp["season"].values
    return abs_sv


def dominant_by_cell(abs_sv, season=None):
    d = abs_sv if season is None else abs_sv[abs_sv["season"] == season]
    cell = d.groupby(["lat", "lon"])[PHYS].mean()
    dom = cell.idxmax(axis=1).rename("driver")
    share = (cell.max(axis=1) / cell.sum(axis=1)).rename("share")
    return pd.concat([cell, dom, share], axis=1).reset_index()


def plot_panel(ax, celldf, title):
    codes = {p: i for i, p in enumerate(PHYS)}
    piv = celldf.assign(code=celldf["driver"].map(codes)) \
                .pivot(index="lat", columns="lon", values="code")
    cmap = ListedColormap(COLORS)
    mesh = ax.pcolormesh(piv.columns.values.astype(float),
                         piv.index.values.astype(float),
                         piv.values, cmap=cmap, vmin=-0.5, vmax=len(PHYS) - 0.5,
                         shading="nearest", transform=M.ccrs.PlateCarree())
    M.clip_to_india(ax, mesh)
    M.draw_boundaries(ax)
    M.gridlines(ax)
    ax.set_title(title, pad=10)


def main():
    abs_sv = compute_shap()

    panels = [("Annual", None), ("Monsoon (JJAS)", "JJAS"), ("Winter (JF)", "JF")]
    fig = plt.figure(figsize=(21, 7.5))
    for i, (title, season) in enumerate(panels, start=1):
        cell = dominant_by_cell(abs_sv, season)
        if season is None:
            cell.to_csv(C.OUT_SHAP / "driver_map_cellwise.csv", index=False)
            counts = cell["driver"].value_counts(normalize=True) * 100
            print("\nDominant driver share of India (annual):")
            print(counts.round(1).to_string())
        ax = M.india_axes(fig, 130 + i)
        plot_panel(ax, cell, f"Dominant rainfall driver — {title}")

    handles = [Patch(facecolor=COLORS[i], edgecolor="k", linewidth=0.3,
                     label=LABELS[p]) for i, p in enumerate(PHYS)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Which large-scale variable drives the model's rainfall prediction?"
                 "  (per-cell dominant mean |SHAP|, XGBoost, test 2016–2023)",
                 fontsize=14, fontweight="bold", y=1.0)
    M.save(fig, C.OUT_SHAP / "driver_map_india.png")


if __name__ == "__main__":
    main()
