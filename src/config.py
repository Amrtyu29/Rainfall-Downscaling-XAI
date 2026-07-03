"""
Central configuration for the Rainfall Downscaling + XAI project.

Approach (after Lyu & Yong 2025 and Hisam et al. 2025):
  large-scale ERA5 atmospheric predictors  ->  IMD high-res rainfall
  explained with SHAP, broken down by season and Indian sub-region.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
SHAPEFILES = DATA / "shapefiles"

MODEL_INPUT = PROCESSED / "model_input.nc"          # monthly, 528 steps, 7 predictors + rainfall

MODELS_DIR = ROOT / "models"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_SHAP = ROOT / "outputs" / "shap"
OUT_PRED = ROOT / "outputs" / "predictions"
for _d in (MODELS_DIR, OUT_FIG, OUT_SHAP, OUT_PRED):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------
# Large-scale predictors present in model_input.nc
PREDICTORS = ["u850", "v850", "q850", "z500", "rh850", "mslp", "t2m"]
TARGET = "rainfall"

# Extra engineered location/season features added to the table
EXTRA_FEATURES = ["lat", "lon", "month_sin", "month_cos"]
FEATURES = PREDICTORS + EXTRA_FEATURES

# ----------------------------------------------------------------------
# Temporal split (from the synopsis)
# ----------------------------------------------------------------------
TRAIN_SLICE = ("1980-01-01", "2010-12-31")
VAL_SLICE = ("2011-01-01", "2015-12-31")
TEST_SLICE = ("2016-01-01", "2023-12-31")

# ----------------------------------------------------------------------
# Seasons (Indian monsoon convention)
# ----------------------------------------------------------------------
# IMD seasonal convention (non-overlapping)
SEASONS = {
    "JJAS": [6, 7, 8, 9],     # Southwest monsoon
    "OND": [10, 11, 12],      # Northeast monsoon / post-monsoon
    "JF": [1, 2],             # Winter (IMD convention: January-February)
    "MAM": [3, 4, 5],         # Pre-monsoon
}

# ----------------------------------------------------------------------
# Sub-regions for regional SHAP / skill analysis  (lat_min, lat_max, lon_min, lon_max)
# ----------------------------------------------------------------------
SUBREGIONS = {
    "Western Ghats":      (8.0, 21.0, 72.5, 77.0),
    "Indo-Gangetic":      (24.0, 30.0, 75.0, 88.0),
    "Northeast India":    (22.0, 29.0, 89.0, 97.0),
    "Arid Rajasthan":     (24.0, 30.0, 69.0, 75.0),
    "Peninsular India":   (8.0, 16.0, 76.0, 84.0),
}

RANDOM_STATE = 42
