# Breast Cancer Diagnosis Under Data Scarcity

A portfolio presentation of the [Breast Cancer Diagnosis Under Data Scarcity](https://github.com/Likhitha-Chinta07/breast-cancer-diagnosis-dl) project.

## Project focus
This project studies how a PyTorch feedforward neural network behaves as available training data is deliberately reduced from 455 patients to 22, while keeping the same stratified 114-patient test set fixed. It then compares the baseline experiment with SMOTE-based oversampling.

The goal is not to claim clinical performance. The goal is to understand data scarcity, malignant-case recall, false negatives, explainability and uncertainty in a controlled machine-learning experiment.

## Verified results
| Training data | Samples | Baseline malignant recall | SMOTE malignant recall | Baseline false negatives | SMOTE false negatives |
|---|---:|---:|---:|---:|---:|
| 100% | 455 | 97.38% | 97.62% | 1.1 | 1.0 |
| 50% | 227 | 96.19% | 96.90% | 1.6 | 1.3 |
| 25% | 113 | 94.76% | 95.71% | 2.2 | 1.8 |
| 10% | 45 | 94.05% | 94.29% | 2.5 | 2.4 |
| 5% | 22 | 92.62% | 93.57% | 3.1 | 2.7 |

Each condition was evaluated over 10 random seeds. The project documentation reports that SMOTE improved malignant recall at every tested data size, but the individual improvements did not reach standard statistical significance with the 10-seed experiment.

The notebook also records 7 of 114 test patients as uncertain under the 0.35–0.65 probability threshold, with 98.13% accuracy on confident-only cases and 96.49% across all cases.

## Dataset and pipeline
- Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn
- 569 records and 30 numeric features
- Stratified 80/20 split with a fixed 114-patient test set
- StandardScaler fitted only on training data
- Training fractions: 100%, 50%, 25%, 10%, 5%
- SMOTE applied only to scaled training data
- 10 random seeds per condition

## Model
The network is implemented in PyTorch:

30 inputs → 64 ReLU → Dropout(0.3) → 32 ReLU → Dropout(0.3) → 1 sigmoid output

## Explainability and uncertainty
SHAP is used for feature-level explanations. The project reports mean perimeter, mean area, mean radius and worst area among the strongest influencing features.

Predictions between 0.35 and 0.65 probability of benign are flagged as uncertain rather than forced into a confident category.

## Dashboard
The repository contains a Streamlit dashboard for selecting a test patient, viewing the model prediction/confidence and inspecting a SHAP waterfall explanation.

Repository note: the dashboard code loads results/final_model.pt, but that saved model artifact is not currently present in the repository's results/ directory. The portfolio therefore describes the dashboard as a prototype rather than implying that a fresh clone is immediately runnable.

## Technologies
Python · PyTorch · scikit-learn · imbalanced-learn / SMOTE · SHAP · Streamlit · pandas · matplotlib

## Limitations
This is a research/portfolio prototype, not a clinical diagnostic tool. Results are based on one benchmark dataset and one fixed test split. The uncertainty threshold was manually chosen, and the SMOTE experiment uses only 10 random seeds. External validation and stronger calibration would be needed before making any claims about performance outside this dataset.

## Source
[Original project repository](https://github.com/Likhitha-Chinta07/breast-cancer-diagnosis-dl)
