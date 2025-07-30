# Disease Prediction from Medical Data

## 📌 Overview
This project predicts the likelihood of a patient suffering from a stroke based on medical history and demographic features.

## 🧠 Objective
To apply machine learning algorithms on structured medical data to predict the occurrence of stroke.

## 📊 Dataset
- **Total Records:** 5110  
- **Target Variable:** `stroke` (0 = No, 1 = Yes)  
- **Features:** Age, Hypertension, Heart Disease, Average Glucose Level, BMI, Smoking Status, etc.  
- **Source:** Healthcare Stroke Dataset

## ✅ Models Used
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest

## 📈 Evaluation Metrics
- Accuracy  
- Precision, Recall, F1-Score  
- ROC-AUC Score

## 🏆 Best Performing Model
- **All models** (Logistic Regression, SVM, Random Forest) achieved  
  - **Accuracy:** 93.93%  
  - **ROC-AUC Score (Logistic Regression):** 0.8512

> ⚠️ Note: Due to class imbalance (only 62 positive stroke cases), precision and recall for class 1 are low across all models. This could be improved using techniques like SMOTE or class weights in further development.

## 📁 Files
- `Disease_Prediction_from_medical_data.py` — Python script for full project  
- `healthcare-dataset-stroke-data.csv` — Input dataset  

## 👩‍💻 Author
Project by **Desai Neha**
