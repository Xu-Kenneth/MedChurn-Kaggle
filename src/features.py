import pandas as pd
from pathlib import Path

_FULL_PATH   = Path("data/raw/KaggleV2-May-2016.csv")
_SAMPLE_PATH = Path("data/raw/sample.csv")

FEATURE_COLS = [
    "age",
    "wait_days",
    "sms_received",
    "hypertension",
    "diabetes",
    "alcoholism",
    "scholarship",
    "gender_female",
    "handicap",
]


def load_and_engineer(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = _FULL_PATH if _FULL_PATH.exists() else _SAMPLE_PATH
    df = pd.read_csv(path)

    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

    df["wait_days"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days.clip(lower=0)
    df["gender_female"] = (df["Gender"] == "F").astype(int)
    df["no_show"] = (df["No-show"] == "Yes").astype(int)

    df = df.rename(columns={
        "Age": "age",
        "Scholarship": "scholarship",
        "Hipertension": "hypertension",
        "Diabetes": "diabetes",
        "Alcoholism": "alcoholism",
        "Handcap": "handicap",
        "SMS_received": "sms_received",
    })

    return df.sort_values("ScheduledDay").reset_index(drop=True)
