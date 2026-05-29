# MedChurn-Kaggle

Lightweight demo of Medchurn project using the public Kaggle No-Show Appointments dataset.

Predicts patient no-show probability using XGBoost with SHAP explainability. No database, no secrets, no setup overhead — just drop the CSV and run.

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Download the dataset**

Go to: https://www.kaggle.com/datasets/joniarroba/noshowappointments

Download `KaggleV2-May-2016.csv` and place it in `data/raw/`.

```
data/
└── raw/
    └── KaggleV2-May-2016.csv   ← put it here
```

**3. Run the pipeline**

```bash
python run.py
```

That's it. The full pipeline — feature engineering, model training, SHAP analysis, and scoring — runs end to end in one command.

## Output

Results land in `data/output/`:

| File | Contents |
|---|---|
| `metrics.json` | AUC-ROC, AUPRC, F1, train/test size, no-show rate |
| `shap_drivers.csv` | Feature importance ranked by mean absolute SHAP value |
| `scored_patients.csv` | Per-patient no-show probability and risk bucket (Low / Medium / High) |

## How It Works

The pipeline mirrors the full production architecture but strips away the infrastructure:

| Production | Demo |
|---|---|
| PostgreSQL / Epic Clarity | `KaggleV2-May-2016.csv` |
| HMAC-SHA256 PHI tokenization | Not needed — dataset is already anonymized |
| Vault / `.env` secrets | Not needed |
| Optuna hyperparameter tuning | Fixed XGBoost params |
| MLflow experiment tracking | `metrics.json` |
| Parquet BI export | CSV output |

The ML approach is identical: temporal split (train on earlier appointments, test on later), `scale_pos_weight` for class imbalance, and SHAP mean absolute values for feature importance.

## Dataset

**Medical Appointment No Shows** — Joni Hoppen & Aquarela Advanced Analytics  
110,000+ outpatient appointments in Vitória, Brazil (2015–2016)  
License: CC0 Public Domain  

**Label:** `no_show = 1` if the patient missed their appointment  
**Features used:** age, wait days, SMS received, hypertension, diabetes, alcoholism, scholarship, gender, handicap
