import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
try:
	from imblearn.over_sampling import SMOTE
except Exception:
	SMOTE = None
from sklearn.preprocessing import StandardScaler


def run_eda(df: pd.DataFrame, out_dir: str = "eda_outputs") -> dict:
	"""Run EDA on the raw dataframe and save CSVs/plots to out_dir. Returns a small summary dict."""
	os.makedirs(out_dir, exist_ok=True)

	# Basic outputs
	df.head().to_csv(os.path.join(out_dir, "head.csv"), index=False)
	summary = df.describe(include='all')
	summary.to_csv(os.path.join(out_dir, "summary_statistics.csv"))

	# Missing values
	missing = df.isnull().sum()
	missing[missing > 0].to_csv(os.path.join(out_dir, "missing_values.csv"))

	# Class balance
	if 'stroke' in df.columns:
		class_counts = df['stroke'].value_counts()
		class_counts.to_csv(os.path.join(out_dir, "class_balance.csv"))

	# Column types
	num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

	# Plots for numeric columns
	for c in num_cols:
		plt.figure(figsize=(6, 4))
		sns.histplot(df[c].dropna(), kde=True)
		plt.title(f"Histogram of {c}")
		plt.tight_layout()
		plt.savefig(os.path.join(out_dir, f"hist_{c}.png"))
		plt.close()

		plt.figure(figsize=(6, 4))
		sns.boxplot(x=df[c].dropna())
		plt.title(f"Boxplot of {c}")
		plt.tight_layout()
		plt.savefig(os.path.join(out_dir, f"box_{c}.png"))
		plt.close()

	# Countplots for categorical columns (reasonable cardinality)
	for c in cat_cols:
		if df[c].nunique() <= 20:
			plt.figure(figsize=(6, 4))
			sns.countplot(y=c, data=df, order=df[c].value_counts().index)
			plt.title(f"Countplot of {c}")
			plt.tight_layout()
			plt.savefig(os.path.join(out_dir, f"count_{c}.png"))
			plt.close()

	# Correlation heatmap for numeric cols
	if len(num_cols) >= 2:
		corr = df[num_cols].corr()
		plt.figure(figsize=(10, 8))
		sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
		plt.title('Correlation matrix')
		plt.tight_layout()
		plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"))
		plt.close()

	return {
		"rows": df.shape[0],
		"cols": df.shape[1],
		"missing_count": int(missing.sum()),
	}


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
	"""Preprocess dataframe and return X, y (features and target)."""
	df = df.copy()
	if 'id' in df.columns:
		df.drop('id', axis=1, inplace=True)
	if 'bmi' in df.columns:
		df['bmi'] = df['bmi'].fillna(df['bmi'].median())
	if 'smoking_status' in df.columns:
		df['smoking_status'] = df['smoking_status'].fillna('Unknown')

	if 'ever_married' in df.columns:
		df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
	if 'Residence_type' in df.columns:
		df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})

	df = pd.get_dummies(df, columns=[c for c in ['gender', 'work_type', 'smoking_status'] if c in df.columns], drop_first=True)

	X = df.drop('stroke', axis=1)
	y = df['stroke']
	return X, y


def train_and_eval(X: pd.DataFrame, y: pd.Series, use_class_weight: bool = False, use_smote: bool = False) -> dict:
	"""Scale, split, train three models and print evaluation metrics.

	If use_class_weight is True, models are created with class_weight='balanced'.
	"""
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

	# Optionally apply SMOTE to the training set
	if use_smote:
		if SMOTE is None:
			raise ImportError("SMOTE requested but imblearn is not installed.")
		sm = SMOTE(random_state=42)
		X_train, y_train = sm.fit_resample(X_train, y_train)
		print(f"Applied SMOTE. New training class distribution: {dict(pd.Series(y_train).value_counts())}")

	cw = 'balanced' if use_class_weight else None
	log_model = LogisticRegression(max_iter=1000, class_weight=cw)
	log_model.fit(X_train, y_train)
	log_pred = log_model.predict(X_test)
	print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
	print(classification_report(y_test, log_pred))

	svm_model = SVC(probability=True, class_weight=cw)
	svm_model.fit(X_train, y_train)
	svm_pred = svm_model.predict(X_test)
	print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
	print(classification_report(y_test, svm_pred))

	rf_model = RandomForestClassifier(class_weight=cw)
	rf_model.fit(X_train, y_train)
	rf_pred = rf_model.predict(X_test)
	print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
	print(classification_report(y_test, rf_pred))
	feature_importances = list(zip(X.columns, rf_model.feature_importances_))
	feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

	# ROC for logistic regression
	y_score = log_model.predict_proba(X_test)[:, 1]
	fpr, tpr, _ = roc_curve(y_test, y_score)
	roc_auc = roc_auc_score(y_test, y_score)
	print("ROC-AUC Score (Logistic Regression):", round(roc_auc, 4))

	# save ROC plot
	plt.figure(figsize=(8, 6))
	plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
	plt.plot([0, 1], [0, 1], 'k--', lw=1)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve — Logistic Regression')
	plt.legend(loc='lower right')
	plt.grid()
	plt.tight_layout()
	os.makedirs("eda_outputs", exist_ok=True)
	roc_path = os.path.join("eda_outputs", "roc_logistic.png")
	plt.savefig(roc_path)
	plt.close()
	print(f"Saved ROC plot to: {roc_path}")

	# Precision-Recall curve and average precision
	precision, recall, pr_thresholds = precision_recall_curve(y_test, y_score)
	avg_prec = average_precision_score(y_test, y_score)

	# Compute F1 for each threshold and find the best
	# pr_thresholds has length n-1 compared to precision/recall; align correctly
	import numpy as _np
	f1_scores = 2 * (precision[:-1] * recall[:-1]) / (_np.maximum(precision[:-1] + recall[:-1], 1e-12))
	if len(f1_scores) > 0:
		best_idx = int(_np.nanargmax(f1_scores))
		best_threshold = float(pr_thresholds[best_idx])
		best_f1 = float(f1_scores[best_idx])
	else:
		best_threshold = None
		best_f1 = None

	# Save PR plot, annotate best F1 threshold
	plt.figure(figsize=(8, 6))
	plt.plot(recall, precision, color='blue', lw=2, label='PR curve (AP = {:.2f})'.format(avg_prec))
	if best_threshold is not None:
		best_precision = precision[best_idx]
		best_recall = recall[best_idx]
		plt.scatter([best_recall], [best_precision], color='red', s=80, label=f'best F1={best_f1:.3f} @ thresh={best_threshold:.3f}')
		plt.annotate(f'th={best_threshold:.2f}', xy=(best_recall, best_precision), xytext=(best_recall + 0.05, best_precision - 0.1), arrowprops=dict(arrowstyle='->', color='red'))
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall Curve — Logistic Regression')
	plt.legend(loc='lower left')
	plt.grid()
	plt.tight_layout()
	pr_path = os.path.join("eda_outputs", "pr_logistic.png")
	plt.savefig(pr_path)
	plt.close()
	print(f"Saved PR plot to: {pr_path}")

	# Collect metrics to return
	from sklearn.metrics import classification_report as _cr
	lr_cr = _cr(y_test, log_pred, output_dict=True)
	svm_cr = _cr(y_test, svm_pred, output_dict=True)
	rf_cr = _cr(y_test, rf_pred, output_dict=True)

	metrics = {
		'logistic': {
			'accuracy': float(accuracy_score(y_test, log_pred)),
			'classification_report': lr_cr,
			'roc_auc': float(roc_auc),
			'roc_path': roc_path,
			'pr_path': pr_path,
			'best_threshold': best_threshold,
			'best_f1': best_f1,
			'average_precision': float(avg_prec),
		},
		'random_forest': {
			'accuracy': float(accuracy_score(y_test, rf_pred)),
			'classification_report': rf_cr,
			'feature_importances': feature_importances,
		},
		'svm': {
			'accuracy': float(accuracy_score(y_test, svm_pred)),
			'classification_report': svm_cr,
		}
	}

	return metrics


def main():
	parser = argparse.ArgumentParser(description="Disease prediction script with optional EDA")
	parser.add_argument('--data', default='healthcare-dataset-stroke-data.csv', help='Path to CSV data file')
	parser.add_argument('--eda', action='store_true', help='Run EDA and save outputs')
	parser.add_argument('--no-train', action='store_true', help='Do not run training/evaluation')
	parser.add_argument('--balance', action='store_true', help='Use class_weight="balanced" for models')
	parser.add_argument('--smote', action='store_true', help='Apply SMOTE to training data')

	args = parser.parse_args()

	print("Loading dataset:", args.data)
	df = pd.read_csv(args.data)
	print("Rows, Columns:", df.shape)

	if args.eda:
		eda_summary = run_eda(df)
		print("EDA summary:", eda_summary)

	if not args.no_train:
		X, y = preprocess(df)
	train_and_eval(X, y, use_class_weight=args.balance, use_smote=args.smote)
    

if __name__ == "__main__":
	main()
