"""
End-to-end validation of the projection chain.

The claim: driving the trained downscaler with bias-corrected HISTORICAL GCM
fields reproduces the observed IMD monsoon climatology. This script computes
that comparison from saved artifacts so the number in the paper is
reproducible:

  downscaled baseline JJAS climatology (outputs/cmip6/<gcm>_rainfall_projection.nc,
  variable 'baseline_jjas', driven by prepared_<gcm>_historical.nc, 1985-2014)
      vs
  observed IMD JJAS climatology 1985-2014 (data/processed/model_input.nc)

Metrics per GCM: spatial pattern correlation (Pearson, over common land
cells) and all-India relative bias of the JJAS mean.

Run:  python src/chain_validation.py
Out:  outputs/cmip6/chain_validation.csv
"""
import numpy as np
import pandas as pd
import xarray as xr

import config as C

OUT = C.ROOT / "outputs" / "cmip6"
GCMS = ["MPI-ESM1-2-HR", "EC-Earth3"]
BASELINE = slice("1985-01-01", "2014-12-31")


def observed_jjas():
    ds = xr.open_dataset(C.MODEL_INPUT).sel(time=BASELINE)
    jjas = ds[C.TARGET].sel(time=ds["time"].dt.month.isin(C.SEASONS["JJAS"]))
    return jjas.mean("time")


def main():
    obs = observed_jjas()
    rows = []
    for gcm in GCMS:
        path = OUT / f"{gcm}_rainfall_projection.nc"
        if not path.exists():
            print(f"[skip] {path.name} not found")
            continue
        model = xr.open_dataset(path)["baseline_jjas"]
        o, m = xr.align(obs, model)
        ov, mv = o.values.ravel(), m.values.ravel()
        ok = np.isfinite(ov) & np.isfinite(mv)
        r = float(np.corrcoef(ov[ok], mv[ok])[0, 1])
        bias = float((mv[ok].mean() - ov[ok].mean()) / ov[ok].mean() * 100)
        rows.append({"gcm": gcm, "pattern_r": round(r, 3),
                     "all_india_bias_pct": round(bias, 1),
                     "n_cells": int(ok.sum())})
        print(f"{gcm:15s}  pattern r = {r:.3f}   bias = {bias:+.1f}%   "
              f"({ok.sum():,} cells)")
    pd.DataFrame(rows).to_csv(OUT / "chain_validation.csv", index=False)
    print("Saved outputs/cmip6/chain_validation.csv")


if __name__ == "__main__":
    main()
