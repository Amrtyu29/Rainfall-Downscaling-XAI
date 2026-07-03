"""
Week 4 of the CMIP6 extension (proposal B.5, Step 5) — the novel result.

SHAP attribution of PROJECTED RAINFALL CHANGE: compare the model's SHAP
contributions between the corrected-historical baseline and the future
scenario, per predictor, for the monsoon season (JJAS).

  "Rainfall over region X increases by Y% — and SHAP shows the increase is
   driven almost entirely by rising low-level humidity."

Outputs (per GCM, for SSP5-8.5 2070-2100 vs baseline 1985-2014):
  outputs/cmip6/shap_change_bar_<gcm>.png       delta mean SHAP per predictor
  outputs/cmip6/shap_change_map_<gcm>.png       map: driver of change per cell
  outputs/cmip6/shap_change_<gcm>.csv           numbers behind both figures
"""
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import shap

import config as C
import data_prep as D
import mapstyle as M
from cmip6_project import land_mask
from driver_map import PHYS, LABELS, COLORS

RAW = C.ROOT / "data_cmip6"
OUT = C.ROOT / "outputs" / "cmip6"
GCMS = ["MPI-ESM1-2-HR", "EC-Earth3"]
SSP = "ssp585"
FUTURE = ("2070-01-01", "2100-12-31")
BASELINE = ("1985-01-01", "2014-12-31")
SAMPLE = 150_000
BEST = "xgboost"


def jjas_table(path, t0, t1, mask):
    ds = xr.open_dataset(path).sel(time=slice(t0, t1)).where(mask)
    df = ds.to_dataframe().reset_index().dropna(subset=C.PREDICTORS)
    df["month"] = pd.to_datetime(df["time"]).dt.month
    df = df[df["month"].isin(C.SEASONS["JJAS"])]
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    return df.reset_index(drop=True)


def shap_frame(explainer, df):
    samp = df.sample(n=min(SAMPLE, len(df)), random_state=C.RANDOM_STATE)
    sv = explainer.shap_values(samp[C.FEATURES])
    out = pd.DataFrame(sv, columns=C.FEATURES)[PHYS]
    out[["lat", "lon"]] = samp[["lat", "lon"]].values
    return out


def analyse(gcm):
    with open(C.MODELS_DIR / f"{BEST}.pkl", "rb") as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    mask = land_mask()

    print(f"[{gcm}] SHAP on baseline JJAS ...")
    base = shap_frame(explainer,
                      jjas_table(RAW / f"prepared_{gcm}_historical.nc", *BASELINE, mask))
    print(f"[{gcm}] SHAP on {SSP} {FUTURE[0][:4]}-{FUTURE[1][:4]} JJAS ...")
    fut = shap_frame(explainer,
                     jjas_table(RAW / f"prepared_{gcm}_{SSP}.nc", *FUTURE, mask))

    # ---- national attribution: change in mean signed SHAP per predictor ----
    delta = fut[PHYS].mean() - base[PHYS].mean()
    delta.sort_values(ascending=False).to_csv(OUT / f"shap_change_{gcm}.csv",
                                              header=["delta_mean_shap_mm"])
    print(f"[{gcm}] change attribution (mm/month):")
    print(delta.sort_values(ascending=False).round(2).to_string())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    d = delta.sort_values()
    ax.barh([LABELS[p] for p in d.index], d.values,
            color=["#b2182b" if v > 0 else "#2166ac" for v in d.values])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Δ mean SHAP contribution to JJAS rainfall (mm/month)")
    ax.set_title(f"Why does monsoon rainfall change?  {gcm}, {SSP.upper()} "
                 f"2070–2100 vs 1985–2014", fontsize=11, fontweight="bold")
    plt.tight_layout()
    M.save(fig, OUT / f"shap_change_bar_{gcm}.png")

    # ---- map: which predictor's contribution grows most, per cell ----
    b = base.groupby(["lat", "lon"])[PHYS].mean()
    f = fut.groupby(["lat", "lon"])[PHYS].mean()
    dcell = (f - b).dropna()
    driver = dcell.idxmax(axis=1).rename("driver").reset_index()

    codes = {p: i for i, p in enumerate(PHYS)}
    piv = driver.assign(code=driver["driver"].map(codes)) \
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
    ax.set_title(f"Driver of projected monsoon rainfall change\n"
                 f"{gcm}, {SSP.upper()} 2070–2100 vs 1985–2014", pad=10)
    handles = [Patch(facecolor=COLORS[i], edgecolor="k", linewidth=0.3,
                     label=LABELS[p]) for i, p in enumerate(PHYS)]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.04))
    M.save(fig, OUT / f"shap_change_map_{gcm}.png")


def main():
    for gcm in GCMS:
        analyse(gcm)


if __name__ == "__main__":
    main()
