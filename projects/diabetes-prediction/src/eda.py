"""Generate a small set of EDA charts for the project report."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import FEATURES, load_data

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "figures"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_data(ROOT / "data" / "diabetes.csv")

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5))
    sns.countplot(data=data, x="Outcome")
    plt.title("Class Distribution")
    plt.xlabel("Outcome")
    plt.ylabel("Records")
    plt.tight_layout()
    plt.savefig(OUTPUT / "class_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x="Outcome", y="Glucose")
    plt.title("Glucose Distribution by Outcome")
    plt.tight_layout()
    plt.savefig(OUTPUT / "glucose_by_outcome.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 7))
    sns.heatmap(data[FEATURES + ["Outcome"]].corr(), cmap="vlag", center=0)
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig(OUTPUT / "correlation_heatmap.png", dpi=160)
    plt.close()

    print(f"Saved EDA figures to {OUTPUT}")


if __name__ == "__main__":
    main()
