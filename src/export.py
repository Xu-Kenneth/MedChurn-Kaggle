import json
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("data/output")


def export(metrics: dict, shap_importance: pd.DataFrame, scored: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    shap_importance.to_csv(OUTPUT_DIR / "shap_drivers.csv", index=False)
    scored.to_csv(OUTPUT_DIR / "scored_patients.csv", index=False)

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  metrics.json         — AUC-ROC, AUPRC, F1")
    print("  shap_drivers.csv     — feature importance by mean |SHAP|")
    print("  scored_patients.csv  — per-patient risk score and bucket")
