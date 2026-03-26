# Stroke Prediction using EDA and Machine Learning

This project is developed for **Healthcare & Medical Analytics domain**.
It performs Exploratory Data Analysis (EDA), preprocessing, machine learning modeling, evaluation, and report generation for stroke prediction using a public healthcare dataset.

The project includes Logistic Regression, SVM, and Random Forest models, with class balancing and SMOTE to handle imbalanced data.

GitHub Copilot was used during development for assisted code generation.

---

## Dataset

Dataset used:
healthcare-dataset-stroke-data.csv

Target variable:
stroke (0 = No, 1 = Yes)

---

## Files

Disease_Prediction_from_medical_data.py
Main script for preprocessing, training, evaluation, ROC/PR curves, feature importance, and metrics.

eda_healthcare_stroke.py
Standalone EDA script that generates plots and summary statistics.

build_report.py
Generates styled HTML report from EDA outputs and model results.

model_evaluation.py
Runs evaluation, confusion matrix, and classification report.

test_pipeline.py
Test script to verify EDA and training pipeline.

requirements.txt
Python dependencies.

healthcare-dataset-stroke-data.csv
Input dataset.

---

## Requirements

Install dependencies:

pip install -r requirements.txt

Libraries used:
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn

---

## Usage

## Usage

### Run EDA only

```
python Disease_Prediction_from_medical_data.py --eda --no-train
```

### Run full training

```
python Disease_Prediction_from_medical_data.py
```

### Run with class balancing

```
python Disease_Prediction_from_medical_data.py --balance
```

### Run with SMOTE

```
python Disease_Prediction_from_medical_data.py --balance --smote
```

### Generate report

```
python build_report.py
```

### Run evaluation

```
python model_evaluation.py
```

### Run tests

```
pytest
```


---

## Features

Exploratory Data Analysis
Missing value handling
Feature encoding
Logistic Regression
SVM
Random Forest
Class imbalance handling
SMOTE oversampling
ROC Curve
Precision-Recall Curve
Feature Importance
Metrics comparison
Confusion Matrix
HTML Report generation

---

## Outputs

Generated in folder:

eda_outputs/

Contains:

plots
csv summaries
roc / pr curves
comparison graphs
eda_report.html

---

## Notes

Dataset is highly imbalanced
Balanced models improve recall
SMOTE improves minority detection
Threshold tuning used for better prediction

---

## Tools Used

Python
scikit-learn
pandas
numpy
matplotlib
seaborn
imbalanced-learn
GitHub Copilot
VS Code

---

## Domain

Healthcare & Medical Analytics
Stroke Risk Prediction using Machine Learning
