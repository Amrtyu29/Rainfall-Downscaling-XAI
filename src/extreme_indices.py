"""
extreme_indices.py
==================
Compute the standard WMO ETCCDI / RClimDex extreme-precipitation indices
(the ones from the table the guide provided) from daily IMD rainfall
(0.25 deg, 1980-2023), per grid cell per year.

Major indices computed & mapped:
    PRCPTOT  annual total wet-day precipitation (mm)      [RR >= 1 mm]
    RX1day   annual max 1-day precipitation (mm)
    Rx5day   annual max consecutive 5-day precipitation (mm)
    R95p     very wet days: total from days > 95th pct (mm)
    R99p     extremely wet days: total from days > 99th pct (mm)
    R20      count of very heavy days (RR >= 20 mm)        (days)
    CDD      max consecutive dry days (RR < 1 mm)          (days)
    SDII     simple daily intensity index (mm/day)
Also computed for the archive netCDF: R10, CWD.

Percentile thresholds (R95p/R99p) use wet-day precip over the base
period 1980-2010 (the training period), per cell.

Outputs (outputs/extremes/indices/):
    etccdi_indices_annual.nc     per-year per-cell fields (all indices)
    etccdi_annual_series.csv     all-India annual mean series
    etccdi_trends.csv            per-index linear trend (Sen/OLS) + p-value
    <INDEX>_climatology.png      one map per major index (1980-2023 mean)
    etccdi_indices_panel.png     8-panel figure of all major indices
    etccdi_trends.png            all-India annual series + trend lines

Run:  python src/extreme_indices.py
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
import mapstyle as M  # noqa: E402

IMD = ROOT / "data" / "raw" / "imd" / "imd_rainfall_0.25deg_1980_2023.nc"
OUT = ROOT / "outputs" / "extremes" / "indices"
OUT.mkdir(parents=True, exist_ok=True)

START_YEAR = 1980        # analysis period start
END_YEAR = 2020          # analysis period end (inclusive)
BASE_END = 2010          # base period 1980-BASE_END for percentile thresholds
WET = 1.0                # mm/day, "wet day" threshold (ETCCDI standard)

warnings.filterwarnings("ok" if False else "ignore", category=RuntimeWarning)

# index -> (long name, units, colormap)
MAJOR = {
    "PRCPTOT": ("Annual wet-day total", "mm", "YlGnBu"),
    "RX1day":  ("Max 1-day precip", "mm", "YlGnBu"),
    "Rx5day":  ("Max 5-day precip", "mm", "YlGnBu"),
    "R95p":    ("Very wet days (>95th pct)", "mm", "YlGnBu"),
    "R99p":    ("Extremely wet days (>99th pct)", "mm", "YlGnBu"),
    "R20":     ("Very heavy days (>=20 mm)", "days", "viridis"),
    "CDD":     ("Consecutive dry days", "days", "YlOrRd"),
    "SDII":    ("Simple daily intensity", "mm/day", "YlGnBu"),
}


def longest_run(cond):
    """Longest run of True along axis 0. cond: (days, lat, lon) bool."""
    cur = np.zeros(cond.shape[1:], dtype=np.int16)
    best = np.zeros(cond.shape[1:], dtype=np.int16)
    for t in range(cond.shape[0]):
        c = cond[t]
        cur = np.where(c, cur + 1, 0)
        best = np.maximum(best, cur)
    return best.astype(np.float32)


def year_indices(x, thr95, thr99):
    """All indices for one year block x: (days, lat, lon), NaN over ocean."""
    wet = x >= WET
    dry = x < WET               # NaN < 1 -> False, so ocean never counts as dry
    xf = np.nan_to_num(x, nan=0.0)

    prcptot = np.where(wet, xf, 0.0).sum(axis=0)
    wetcount = wet.sum(axis=0).astype(np.float32)
    rx1 = xf.max(axis=0)
    n = xf.shape[0]
    five = (xf[0:n - 4] + xf[1:n - 3] + xf[2:n - 2] + xf[3:n - 1] + xf[4:n])
    rx5 = five.max(axis=0)
    r10 = (x >= 10).sum(axis=0).astype(np.float32)
    r20 = (x >= 20).sum(axis=0).astype(np.float32)
    r95 = np.where(x > thr95[None, :, :], xf, 0.0).sum(axis=0)
    r99 = np.where(x > thr99[None, :, :], xf, 0.0).sum(axis=0)
    cdd = longest_run(dry)
    cwd = longest_run(wet)
    with np.errstate(invalid="ignore", divide="ignore"):
        sdii = np.where(wetcount > 0, prcptot / wetcount, np.nan)
    return {"PRCPTOT": prcptot, "RX1day": rx1, "Rx5day": rx5, "R95p": r95,
            "R99p": r99, "R10": r10, "R20": r20, "CDD": cdd, "CWD": cwd,
            "SDII": sdii.astype(np.float32)}


def main():
    print("Loading daily IMD rainfall ...", flush=True)
    da = xr.open_dataset(IMD)["rainfall"]
    yr_all = da["time"].dt.year
    da = da.sel(time=(yr_all >= START_YEAR) & (yr_all <= END_YEAR))
    lats = da.lat.values
    lons = da.lon.values
    years = np.unique(da["time"].dt.year.values)
    print(f"  grid {len(lats)}x{len(lons)}  years {years[0]}-{years[-1]}", flush=True)

    # ---- base-period percentile thresholds (memory-safe, per lat row) ----
    print(f"Computing 95th/99th wet-day thresholds on 1980-{BASE_END} ...", flush=True)
    base = da.sel(time=da["time"].dt.year <= BASE_END).astype("float32").values
    base = np.where(base >= WET, base, np.nan).astype("float32")
    thr95 = np.full(base.shape[1:], np.nan, np.float32)
    thr99 = np.full(base.shape[1:], np.nan, np.float32)
    for i in range(base.shape[1]):
        slab = base[:, i, :]
        if np.isfinite(slab).any():
            q = np.nanpercentile(slab, [95, 99], axis=0)
            thr95[i], thr99[i] = q[0], q[1]
    del base
    print("  thresholds done", flush=True)

    # ---- per-year loop ----
    names = ["PRCPTOT", "RX1day", "Rx5day", "R95p", "R99p", "R10", "R20",
             "CDD", "CWD", "SDII"]
    acc = {k: [] for k in names}
    land = np.zeros((len(lats), len(lons)), dtype=bool)
    for y in years:
        x = da.sel(time=da["time"].dt.year == y).astype("float32").values
        land |= np.isfinite(x).any(axis=0)
        idx = year_indices(x, thr95, thr99)
        for k in names:
            acc[k].append(idx[k])
        print(f"  {y} done", flush=True)

    # stack -> (year, lat, lon), mask ocean
    fields = {}
    for k in names:
        arr = np.stack(acc[k]).astype(np.float32)
        arr[:, ~land] = np.nan
        fields[k] = arr

    # ---- save annual fields to netCDF ----
    ds = xr.Dataset(
        {k: (("year", "lat", "lon"), fields[k]) for k in names},
        coords={"year": years, "lat": lats, "lon": lons},
    )
    for k, (ln, un, _) in {**MAJOR, "R10": ("Heavy days (>=10mm)", "days", ""),
                           "CWD": ("Consecutive wet days", "days", "")}.items():
        if k in ds:
            ds[k].attrs.update(long_name=ln, units=un)
    ds.to_netcdf(OUT / "etccdi_indices_annual.nc")
    print("Saved etccdi_indices_annual.nc", flush=True)

    # ---- all-India annual series + trends ----
    rows = []
    series = {}
    for k in names:
        s = np.nanmean(fields[k].reshape(len(years), -1), axis=1)
        series[k] = s
        lr = stats.linregress(years, s)
        sen = stats.theilslopes(s, years)[0]
        rows.append({
            "index": k, "mean": np.nanmean(s),
            "slope_per_decade": lr.slope * 10,
            "sen_slope_per_decade": sen * 10,
            "pct_per_decade": (lr.slope * 10 / np.nanmean(s) * 100) if np.nanmean(s) else np.nan,
            "p_value": lr.pvalue,
            "significant_95": lr.pvalue < 0.05,
        })
    trends = pd.DataFrame(rows).set_index("index")
    trends.to_csv(OUT / "etccdi_trends.csv")
    pd.DataFrame(series, index=years).rename_axis("year").to_csv(OUT / "etccdi_annual_series.csv")
    print("\n=== All-India trends (per decade) ===", flush=True)
    print(trends[["mean", "slope_per_decade", "pct_per_decade", "p_value",
                  "significant_95"]].round(3).to_string(), flush=True)

    # ---- climatology maps (one per major index) ----
    print("\nRendering maps ...", flush=True)
    clim = {k: np.nanmean(fields[k], axis=0) for k in names}

    def piv(arr):
        return xr.DataArray(arr, coords={"lat": lats, "lon": lons},
                            dims=["lat", "lon"]).to_pandas()

    for k, (ln, un, cmap) in MAJOR.items():
        lo, hi = np.nanpercentile(clim[k], [2, 98])
        fig = plt.figure(figsize=(9.5, 8))
        ax = M.india_axes(fig)
        mesh = M.field(ax, piv(clim[k]), cmap=cmap, vmin=lo, vmax=hi)
        M.finish_map(ax, mesh, f"{k} — {ln} ({years[0]}-{years[-1]} mean)", un)
        fig.savefig(OUT / f"{k}_climatology.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # ---- combined 8-panel figure ----
    fig = plt.figure(figsize=(19, 9))
    for j, (k, (ln, un, cmap)) in enumerate(MAJOR.items()):
        lo, hi = np.nanpercentile(clim[k], [2, 98])
        ax = M.india_axes(fig, rect=240 + j + 1)
        mesh = M.field(ax, piv(clim[k]), cmap=cmap, vmin=lo, vmax=hi)
        M.finish_map(ax, mesh, f"{k} ({un})", un)
    fig.suptitle(f"ETCCDI extreme-precipitation indices over India — {years[0]}-{years[-1]} climatology",
                 fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "etccdi_indices_panel.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- trends figure (all-India series + OLS line) ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, k in zip(axes.ravel(), MAJOR):
        s = series[k]
        ax.plot(years, s, color="#1E40AF", lw=1.4, marker="o", ms=3)
        lr = stats.linregress(years, s)
        ax.plot(years, lr.intercept + lr.slope * years, "--", color="#B45309", lw=1.6)
        un = MAJOR[k][1]
        sig = "*" if lr.pvalue < 0.05 else ""
        ax.set_title(f"{k}  ({lr.slope*10:+.2f} {un}/decade{sig})", fontsize=11)
        ax.set_ylabel(un, fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle("All-India annual ETCCDI indices with linear trend  (* = significant, p<0.05)",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "etccdi_trends.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved panel + trend figures. Done.", flush=True)


if __name__ == "__main__":
    main()
