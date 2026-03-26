import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "healthcare-dataset-stroke-data.csv"
OUT_DIR = "eda_outputs"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Rows, Columns:", df.shape)

# Basic info
info_buf = []
info_buf.append(str(df.info()))

# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0]
missing.to_csv(os.path.join(OUT_DIR, "missing_values.csv"))
print("Missing values saved to:", os.path.join(OUT_DIR, "missing_values.csv"))

# Summary statistics
summary = df.describe(include='all')
summary.to_csv(os.path.join(OUT_DIR, "summary_statistics.csv"))
print("Summary statistics saved to:", os.path.join(OUT_DIR, "summary_statistics.csv"))

# Class balance
if 'stroke' in df.columns:
    class_counts = df['stroke'].value_counts()
    class_counts.to_csv(os.path.join(OUT_DIR, "class_balance.csv"))
    print("Class balance saved to:", os.path.join(OUT_DIR, "class_balance.csv"))

# Save first few rows
df.head().to_csv(os.path.join(OUT_DIR, "head.csv"), index=False)

# Numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# Histograms for numeric columns
for c in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[c].dropna(), kde=True)
    plt.title(f"Histogram of {c}")
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"hist_{c}.png")
    plt.savefig(fname)
    plt.close()

# Boxplots for numeric columns
for c in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[c].dropna())
    plt.title(f"Boxplot of {c}")
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"box_{c}.png")
    plt.savefig(fname)
    plt.close()

# Countplots for categorical columns (limit to those with <= 20 unique values)
for c in cat_cols:
    if df[c].nunique() <= 20:
        plt.figure(figsize=(6,4))
        sns.countplot(y=c, data=df, order=df[c].value_counts().index)
        plt.title(f"Countplot of {c}")
        plt.tight_layout()
        fname = os.path.join(OUT_DIR, f"count_{c}.png")
        plt.savefig(fname)
        plt.close()

# Correlation heatmap (numeric)
if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"))
    plt.close()

print("EDA outputs written to:", OUT_DIR)
print("Done")
