# Stroke Prediction using EDA and Machine Learning

This project is developed in the **Artificial Intelligence & Machine Learning (AI/ML) domain**.
It performs Exploratory Data Analysis (EDA), data preprocessing, machine learning modeling, evaluation, and report generation for stroke prediction using a public healthcare dataset.

The system uses multiple machine learning algorithms including Logistic Regression, Support Vector Machine (SVM), and Random Forest, with class balancing and SMOTE to handle imbalanced data.

GitHub Copilot was used during development for assisted code generation.

---

## Dataset

Dataset used:
`healthcare-dataset-stroke-data.csv`

Target variable:

```
stroke (0 = No Stroke, 1 = Stroke)
```

---

## Project Files

* **Disease_Prediction_from_medical_data.py**
  Main script for preprocessing, training, evaluation, ROC/PR curves, feature importance, and metrics.

* **eda_healthcare_stroke.py**
  Standalone EDA script that generates plots and summary statistics.

* **build_report.py**
  Generates styled HTML report from EDA outputs and model results.

* **model_evaluation.py**
  Runs evaluation, confusion matrix, and classification report.

* **test_pipeline.py**
  Test script to verify EDA and training pipeline.

* **requirements.txt**
  Python dependencies required for the project.

* **healthcare-dataset-stroke-data.csv**
  Input dataset used for training and analysis.

---

## Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

Libraries used:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* imbalanced-learn

---

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

* Exploratory Data Analysis (EDA)
* Missing value handling
* Feature encoding
* Logistic Regression model
* Support Vector Machine (SVM)
* Random Forest classifier
* Class imbalance handling
* SMOTE oversampling
* ROC Curve analysis
* Precision–Recall Curve
* Feature importance analysis
* Metrics comparison
* Confusion Matrix
* HTML Report generation

---

## Outputs

Generated in folder:

```
eda_outputs/
```

Contains:

* Plots
* CSV summaries
* ROC / PR curves
* Model comparison graphs
* eda_report.html

---

## Notes

* Dataset is highly imbalanced
* Balanced models improve recall
* SMOTE improves minority class detection
* Threshold tuning used for better prediction performance

---

## Tools Used

* Python
* scikit-learn
* pandas
* numpy
* matplotlib
* seaborn
* imbalanced-learn
* GitHub Copilot
* VS Code

---

## Domain

* Artificial Intelligence & Machine Learning (AI/ML)
* Stroke Risk Prediction using Machine Learning
