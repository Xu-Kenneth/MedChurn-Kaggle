from pathlib import Path

from src.features import load_and_engineer, _FULL_PATH, _SAMPLE_PATH
from src.train import train_and_evaluate
from src.export import export


def main():
    using = _FULL_PATH if _FULL_PATH.exists() else _SAMPLE_PATH
    print(f"Loading {'full dataset' if using == _FULL_PATH else 'sample (5k rows)'}...")
    df = load_and_engineer()
    print(f"  {len(df):,} appointments | no-show rate: {df['no_show'].mean():.1%}")

    print("\nTraining XGBoost model...")
    model, metrics, shap_importance, scored = train_and_evaluate(df)

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nTop SHAP drivers:")
    print(shap_importance.to_string(index=False))

    export(metrics, shap_importance, scored)


if __name__ == "__main__":
    main()
