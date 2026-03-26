# Stroke Prediction EDA and Modeling

This repository contains a stroke prediction pipeline with EDA, modeling, and report generation.

## Files
- `Disease_Prediction_from_medical_data.py`: main script with data loading, preprocessing, EDA, model training (Logistic, SVM, RandomForest), ROC/PR plots, class-weight balancing, and SMOTE options.
- `eda_healthcare_stroke.py`: independent EDA script (also works as a utility entrypoint).
- `build_report.py`: generates HTML report from the saved EDA outputs and model metrics.
- `model_evaluation.py`: helper script to compute confusion matrices and classification metrics for best models.
- `healthcare-dataset-stroke-data.csv`: input dataset.

## Usage
1. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Run EDA only:
   ```bash
   D:/Neha/Code_Alpha/.venv/Scripts/python.exe Disease_Prediction_from_medical_data.py --eda --no-train
   ```
3. Run full training and evaluation:
   ```bash
   D:/Neha/Code_Alpha/.venv/Scripts/python.exe Disease_Prediction_from_medical_data.py
   ```
4. Run with class weight and SMOTE:
   ```bash
   D:/Neha/Code_Alpha/.venv/Scripts/python.exe Disease_Prediction_from_medical_data.py --balance --smote
   ```
5. Generate HTML report:
   ```bash
   D:/Neha/Code_Alpha/.venv/Scripts/python.exe build_report.py
   ```

## Outputs
- `eda_outputs/`: contains CSV summaries and PNG plots
- `eda_outputs/eda_report.html`: rendered EDA + model report

## Notes
- The data is imbalanced (`stroke` positive << negative). Use `--balance` and/or `--smote` to improve recall for minority class.
- The model evaluation metrics include accuracy, precision, recall, F1, ROC AUC, and PR curve.
