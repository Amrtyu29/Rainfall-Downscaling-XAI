# 🌧️ Downscaling Rainfall over India using Explainable AI (XAI)

A clean, reproducible pipeline that predicts high-resolution IMD rainfall from
large-scale ERA5 atmospheric predictors and **explains** the predictions with
SHAP — broken down by season and Indian sub-region.

Method follows **Lyu & Yong (2025, JGR ML)** and **Hisam et al. (2025, STOTEN)**:
stack predictors → ML regression against gauge/gridded rainfall → SHAP attribution.

---

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1. Build table | `src/data_prep.py` | tidy (time,lat,lon) → rainfall table, temporal split |
| 2. Train + compare | `src/train_models.py` | Linear / RF / XGBoost / LightGBM, `model_metrics.csv`, predictions |
| 3. Explain | `src/explain_shap.py` | SHAP beeswarm, global bar, **seasonal & regional |SHAP| heatmaps** |
| 4. Validate | `src/validate.py` | per-season/region skill, **Taylor diagram**, spatial maps |
| 5. Driver map | `src/driver_map.py` | **"Rainfall driver map of India"** — per-cell dominant SHAP predictor, annual + JJAS + DJF |

All India maps use the state-boundary shapefile (`data/shapefiles/india_st.shp`),
are clipped to the national outline, and saved at 300 dpi (`src/mapstyle.py`).

## CMIP6 extension (explainable future projections)

| Step | Script | Output |
|------|--------|--------|
| 6. Download | `src/cmip6_download.py` | MPI-ESM1-2-HR + EC-Earth3 (r1i1p1f1), historical/SSP2-4.5/SSP5-8.5, 7 predictors, India box — public Google Cloud archive, resumable piece-level caching |
| 7. Prepare | `src/cmip6_prepare.py` | regrid to 0.25° + per-month per-cell quantile-mapping bias correction vs ERA5 (verified: corrected means match ERA5 exactly) |
| 8. Project | `src/cmip6_project.py` | future JJAS rainfall + change maps (2040–2070, 2070–2100), regional summaries |
| 9. Attribute | `src/cmip6_explain.py` | **SHAP attribution of projected change** — why the monsoon gets wetter (novel) |

`src/autopilot.sh` runs steps 6–8 unattended with retries (used with `caffeinate`).

**Chain validation:** GCM-driven downscaled JJAS climatology vs observed IMD:
pattern r = 0.965, all-India bias +3.6%.

**Headline projection:** both GCMs → wetter monsoon (all-India JJAS +17…+49% by
2040–2070 depending on scenario/model); SHAP attributes the increase to rising
low-level specific humidity (thermodynamic moistening), with temperature and
pressure changes weakly opposing. Far-future SSP5-8.5 magnitudes are upper-end
(stationarity/extrapolation caveat — see report).

## Interactive dashboard

```bash
source .venv/bin/activate
streamlit run dashboard/app.py       # opens at http://localhost:8501
```

Four tabs: Explore & Predict (publication-style Observed/Model/Difference maps for
any month), Drivers (XAI), Future Climate (CMIP6), Extremes. Maps are rendered with
the same `src/mapstyle.py` engine as the report figures.

## Reports

- `report/CMIP6_Extension_Proposal.docx` — professor-approved extension proposal
- `report/Final_Report_Rainfall_Downscaling_XAI.docx` — full final report (core + extension)
- `report/Defense_Presentation_Rainfall_XAI.pptx` — 13-slide defense deck

## Environment & data notes

- `pip install -r requirements.txt` into a Python 3.9+ venv reproduces the environment.
- `data/` and `.venv` are symlinks into `~/Desktop/summer project/` — do not delete or
  move that folder (or replace the symlinks with real copies first).
- Known caveat: the processed monthly cube carries rainfall on all grid-box cells;
  on strictly IMD-observed cells (29% of the box) test R² is 0.69 (LightGBM) vs 0.73
  on the full grid — conclusions unchanged.

Run everything:

```bash
source .venv/bin/activate
cd src && python run_all.py
```

## Data (verified, symlinked from the original folder)

- `data/processed/model_input.nc` — monthly cube, 1980–2023 (528 steps), 0.25°,
  predictors `u850,v850,q850,z500,rh850,mslp,t2m` + IMD `rainfall`.
- `data/raw/` — ERA5 (pressure + single levels) and IMD daily gridded rainfall.
- `data/shapefiles/` — India state/district boundaries.

## Headline result (test set 2016–2023)

| Model | R² | RMSE (mm/mo) | PCC |
|---|---|---|---|
| LightGBM | 0.73 | 52.7 | 0.85 |
| XGBoost | 0.71 | 54.1 | 0.84 |
| RandomForest | 0.70 | 55.3 | 0.84 |
| Linear (baseline) | 0.21 | 89.6 | 0.46 |

ML models far outperform the traditional linear baseline. SHAP identifies
low-level humidity (`q850`) as the dominant rainfall driver — consistent with
monsoon physics.

### Driver-map findings (novel contribution)

- Humidity controls Indian rainfall everywhere: specific humidity (`q850`,
  moisture supply) dominates 53.5% of grid cells, relative humidity (`rh850`,
  saturation) 46.4%.
- During the monsoon (JJAS) moisture supply (`q850`) rules most of the country;
  in winter (DJF) saturation (`rh850`) takes over the peninsula (northeast monsoon).
- Secondary drivers are regional and physically sensible: 2 m temperature in
  the Himalayan belt, meridional wind over the far south (northeast-monsoon flow),
  sea-level pressure near the Odisha coast (cyclone alley).

## Config

All knobs (predictors, splits, seasons, sub-region boxes) live in `src/config.py`.
