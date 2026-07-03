"""
Week 3 of the CMIP6 extension (proposal B.5, Steps 3-4).

Applies the trained downscaling model to bias-corrected CMIP6 predictors and
maps projected rainfall change over India.

  baseline : corrected GCM historical, 1985-2014
  futures  : SSP2-4.5 and SSP5-8.5, near (2040-2070) and far (2070-2100)

Outputs
  outputs/cmip6/<model>_rainfall_projection.nc     per-cell seasonal means
  outputs/cmip6/change_maps_<model>.png            JJAS % change, 2x2 panel
  outputs/cmip6/change_summary_<model>.csv         per-subregion change table
"""
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import data_prep as D
import mapstyle as M

RAW = C.ROOT / "data_cmip6"
OUT = C.ROOT / "outputs" / "cmip6"
OUT.mkdir(parents=True, exist_ok=True)

BEST = "xgboost"
GCMS = ["MPI-ESM1-2-HR", "EC-Earth3"]
BASELINE = ("1985-01-01", "2014-12-31")
HORIZONS = {"2040-2070": ("2040-01-01", "2070-12-31"),
            "2070-2100": ("2070-01-01", "2100-12-31")}
SCENARIOS = ["ssp245", "ssp585"]


def land_mask():
    """Cells where IMD rainfall exists (land India)."""
    ds = xr.open_dataset(C.MODEL_INPUT)
    return ds[C.TARGET].isel(time=6).notnull()   # any monsoon month


def predict_cube(model, ds):
    """Run the trained model over every (time, cell) of a predictor cube."""
    df = ds.to_dataframe().reset_index().dropna(subset=C.PREDICTORS)
    df["month"] = pd.to_datetime(df["time"]).dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["pred"] = model.predict(df[C.FEATURES].values)
    df.loc[df["pred"] < 0, "pred"] = 0.0
    return df


def seasonal_mean(df, months):
    d = df[df["month"].isin(months)]
    return d.groupby(["lat", "lon"])["pred"].mean()


def project(gcm):
    with open(C.MODELS_DIR / f"{BEST}.pkl", "rb") as f:
        model = pickle.load(f)
    mask = land_mask()

    hist = xr.open_dataset(RAW / f"prepared_{gcm}_historical.nc") \
             .sel(time=slice(*BASELINE)).where(mask)
    print(f"[{gcm}] predicting baseline 1985-2014 ...")
    base_df = predict_cube(model, hist)
    base_jjas = seasonal_mean(base_df, C.SEASONS["JJAS"])

    results = {"baseline_jjas": base_jjas}
    rows = []
    for ssp in SCENARIOS:
        fut_all = xr.open_dataset(RAW / f"prepared_{gcm}_{ssp}.nc").where(mask)
        for hz, (t0, t1) in HORIZONS.items():
            print(f"[{gcm}] predicting {ssp} {hz} ...")
            fut_df = predict_cube(model, fut_all.sel(time=slice(t0, t1)))
            fut_jjas = seasonal_mean(fut_df, C.SEASONS["JJAS"])
            change = 100.0 * (fut_jjas - base_jjas) / base_jjas
            results[f"{ssp}_{hz}_jjas"] = fut_jjas
            results[f"{ssp}_{hz}_change"] = change

            # sub-regional summary: aggregate rainfall FIRST, then % change
            # (mean of per-cell percentages explodes where the baseline ~0 mm)
            merged = pd.concat([base_jjas.rename("base"),
                                fut_jjas.rename("fut")], axis=1).reset_index()
            merged = D.add_subregion(merged)
            for reg, g in merged.groupby("subregion"):
                pct = 100.0 * (g["fut"].mean() - g["base"].mean()) / g["base"].mean()
                rows.append({"scenario": ssp, "horizon": hz, "subregion": reg,
                             "JJAS_change_pct": round(pct, 1)})
    # save cube
    xr.Dataset({k: v.to_xarray() for k, v in results.items()}) \
      .to_netcdf(OUT / f"{gcm}_rainfall_projection.nc")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / f"change_summary_{gcm}.csv", index=False)
    print(f"[{gcm}] saved projection cube + summary")
    return results


def change_maps(gcm, results):
    fig = plt.figure(figsize=(15, 13))
    idx = 1
    for ssp in SCENARIOS:
        for hz in HORIZONS:
            ax = M.india_axes(fig, 220 + idx)
            piv = results[f"{ssp}_{hz}_change"].reset_index() \
                .pivot(index="lat", columns="lon",
                       values=results[f"{ssp}_{hz}_change"].name or "pred")
            mesh = M.field(ax, piv, cmap="BrBG", vmin=-40, vmax=40)
            M.finish_map(ax, mesh,
                         f"{ssp.upper()} {hz}",
                         "JJAS rainfall change (%)")
            idx += 1
    fig.suptitle(f"Projected monsoon (JJAS) rainfall change over India — {gcm}\n"
                 f"(explainable ML downscaling, baseline 1985–2014)",
                 fontsize=15, fontweight="bold")
    M.save(fig, OUT / f"change_maps_{gcm}.png")


def main():
    for gcm in GCMS:
        if not (RAW / f"prepared_{gcm}_historical.nc").exists():
            print(f"[skip] {gcm}: prepared files not found (run cmip6_prepare.py)")
            continue
        results = project(gcm)
        change_maps(gcm, results)


if __name__ == "__main__":
    main()
