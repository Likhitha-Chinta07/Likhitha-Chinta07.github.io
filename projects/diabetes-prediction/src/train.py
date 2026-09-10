"""Train, compare and save diabetes classification models."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_utils import FEATURES, load_data, prepare_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "diabetes_decision_tree.joblib"
FIGURES_DIR = ROOT / "reports" / "figures"


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer([("numeric", numeric, FEATURES)], remainder="drop")


def build_models() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "Decision Tree": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
    }


def evaluate_models(models, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    rows = []
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "roc_auc": roc_auc_score(y_test, probabilities),
                "cv_auc_mean": cv_auc.mean(),
                "cv_auc_std": cv_auc.std(),
            }
        )

        print(f"\n{name}")
        print(classification_report(y_test, predictions, digits=3))
        print(f"Test ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")

        if name == "Decision Tree":
            ConfusionMatrixDisplay.from_predictions(y_test, predictions)
            plt.title("Diabetes Prediction — Decision Tree")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
            plt.close()
            joblib.dump(model, MODEL_PATH)

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def main() -> None:
    frame = load_data(ROOT / "data" / "diabetes.csv")
    X, y = prepare_features(frame)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    results = evaluate_models(build_models(), X_train, X_test, y_train, y_test)
    results.to_csv(ROOT / "reports" / "model_results.csv", index=False)

    print("\nModel comparison:")
    print(results.to_string(index=False))
    print(f"\nSaved primary model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
