"""Run the full pipeline end-to-end."""
import data_prep, train_models, explain_shap, validate, driver_map

if __name__ == "__main__":
    print("\n[1/4] Training models ...");        train_models.main()
    print("\n[2/4] SHAP explainability ...");     explain_shap.main()
    print("\n[3/4] Validation & figures ...");    validate.main()
    print("\n[4/4] Rainfall driver map ...");     driver_map.main()
    print("\nDone. See outputs/ for figures, predictions, and SHAP results.")
