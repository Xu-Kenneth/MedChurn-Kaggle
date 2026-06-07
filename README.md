# MedChurn-Kaggle

Predicts patient appointment no-shows using XGBoost with SHAP explainability. No database, no secrets, no Kaggle account needed.

## Quick start

```bash
git clone https://github.com/kxm1088/MedChurn-Kaggle
cd MedChurn-Kaggle
pip install -r requirements.txt
python run.py
```

A 5,000-appointment sample dataset is bundled — just clone and run.

## Output

Results land in `data/output/`:

| File | Contents |
|---|---|
| `metrics.json` | AUC-ROC, AUPRC, F1, train/test size, no-show rate |
| `shap_drivers.csv` | Feature importance ranked by mean absolute SHAP value |
| `scored_patients.csv` | Per-patient no-show probability and risk bucket (Low / Medium / High) |

## Full dataset (optional)

For the complete 110K-appointment dataset, download `KaggleV2-May-2016.csv` from [Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments) and place it in `data/raw/`. The pipeline auto-detects and uses it automatically.

## How It Works

The pipeline mirrors a production architecture but strips away the infrastructure:

| Production | Demo |
|---|---|
| PostgreSQL / Epic Clarity | Bundled CSV sample |
| HMAC-SHA256 PHI tokenization | Not needed — dataset is already anonymized |
| Vault / `.env` secrets | Not needed |
| Optuna hyperparameter tuning | Fixed XGBoost params |
| MLflow experiment tracking | `metrics.json` |
| Parquet BI export | CSV output |

The ML approach is identical: temporal split (train on earlier appointments, test on later), `scale_pos_weight` for class imbalance, and SHAP mean absolute values for feature importance.

## Dataset

**Medical Appointment No Shows** — Joni Hoppen & Aquarela Advanced Analytics  
110,000+ outpatient appointments in Vitória, Brazil (2015–2016)  
License: [CC0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

**Label:** `no_show = 1` if the patient missed their appointment  
**Features used:** age, wait days, SMS received, hypertension, diabetes, alcoholism, scholarship, gender, handicap
