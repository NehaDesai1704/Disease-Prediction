import os
import pandas as pd
from Disease_Prediction_from_medical_data import run_eda, preprocess, train_and_eval

OUT_DIR = 'eda_outputs'
REPORT_PATH = os.path.join(OUT_DIR, 'eda_report.html')

os.makedirs(OUT_DIR, exist_ok=True)

print('Loading dataset and running EDA...')
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
run_eda(df, out_dir=OUT_DIR)

print('Running baseline training (no class weight)')
X, y = preprocess(df)
baseline_metrics = train_and_eval(X, y, use_class_weight=False)

print('Running balanced training (class_weight=balanced)')
balanced_metrics = train_and_eval(X, y, use_class_weight=True)

print('Running SMOTE + balanced training')
smote_metrics = train_and_eval(X, y, use_class_weight=True, use_smote=True)

# Generate combined ROC/PR comparison plots
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

def model_run(X, y, class_weight=False, smote=False):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42)
    if smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    cw = 'balanced' if class_weight else None
    model = LogisticRegression(max_iter=1000, class_weight=cw)
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    return {
        'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc_score(y_test, y_score),
        'precision': precision, 'recall': recall, 'avg_prec': average_precision_score(y_test, y_score)
    }

comparison = {
    'baseline': model_run(X, y, class_weight=False, smote=False),
    'balanced': model_run(X, y, class_weight=True, smote=False),
    'smote': model_run(X, y, class_weight=True, smote=True)
}

# Plot belt
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
for key, data in comparison.items():
    plt.plot(data['fpr'], data['tpr'], label=f'{key} ROC (AUC={data["roc_auc"]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Comparison')
plt.legend()
plt.grid()
plt.tight_layout()
compare_roc_path = os.path.join(OUT_DIR, 'roc_comparison.png')
plt.savefig(compare_roc_path)
plt.close()

plt.figure(figsize=(10, 8))
for key, data in comparison.items():
    plt.plot(data['recall'], data['precision'], label=f'{key} PR (AP={data["avg_prec"]:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Comparison')
plt.legend()
plt.grid()
plt.tight_layout()
compare_pr_path = os.path.join(OUT_DIR, 'pr_comparison.png')
plt.savefig(compare_pr_path)
plt.close()

# Build a styled HTML report
def df_to_table_html(path, index=False, header=True, caption=None):
    try:
        df = pd.read_csv(path, header=0 if header else None)
        html = df.to_html(index=index, header=header, classes='table')
        if caption:
            html = f'<div class="table-caption">{caption}</div>' + html
        # Wrap in scrollable div
        html = f'<div class="table-scroll">{html}</div>'
        return html
    except Exception as e:
        return f'<p class="error">Unable to load {os.path.basename(path)}: {e}</p>'

def format_metrics_table(metrics_dict):
    """Format metrics dict as HTML table."""
    import json
    html = '<div class="table-scroll"><table class="table"><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>ROC-AUC</th><th>Avg Precision</th></tr></thead><tbody>'
    
    for model_name, metrics in metrics_dict.items():
        accuracy = metrics.get('accuracy', 'N/A')
        roc_auc = metrics.get('roc_auc', 'N/A')
        avg_prec = metrics.get('average_precision', 'N/A')
        
        # Parse classification_report to get weighted avg precision, recall, f1
        report = metrics.get('classification_report', '{}')
        try:
            report_dict = json.loads(report) if isinstance(report, str) else report
            weighted_avg = report_dict.get('weighted avg', {})
            precision = weighted_avg.get('precision', 'N/A')
            recall = weighted_avg.get('recall', 'N/A')
            f1 = weighted_avg.get('f1-score', 'N/A')
        except:
            precision = recall = f1 = 'N/A'
        
        # Format values
        accuracy_str = f'{accuracy:.4f}' if isinstance(accuracy, float) else str(accuracy)
        precision_str = f'{precision:.4f}' if isinstance(precision, float) else str(precision)
        recall_str = f'{recall:.4f}' if isinstance(recall, float) else str(recall)
        f1_str = f'{f1:.4f}' if isinstance(f1, float) else str(f1)
        roc_auc_str = f'{roc_auc:.4f}' if isinstance(roc_auc, float) else str(roc_auc)
        avg_prec_str = f'{avg_prec:.4f}' if isinstance(avg_prec, float) else str(avg_prec)
        
        html += f'<tr><td><strong>{model_name}</strong></td><td>{accuracy_str}</td><td>{precision_str}</td><td>{recall_str}</td><td>{f1_str}</td><td>{roc_auc_str}</td><td>{avg_prec_str}</td></tr>'
    
    html += '</tbody></table></div>'
    return html

css = '''
<style>
* { box-sizing: border-box }
html, body { margin: 0; padding: 0; width: 100%; overflow-x: hidden }
body { font-family: Arial, Helvetica, sans-serif; margin: 0; padding: 10px; color: #222; background: #f9f9f9 }
.container { max-width: 95%; margin: 0 auto; padding: 0 5px }
h1, h2, h3 { margin-top: 0; word-wrap: break-word }
h1 { color: #2b6cb0; margin-bottom: 20px }
.grid { display: grid; grid-template-columns: 1fr; gap: 20px; overflow: hidden }
.full { grid-column: 1 / -1 }
.card { background: #fff; padding: 10px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow: hidden }
.table-scroll { overflow-x: auto; width: 100%; -webkit-overflow-scrolling: touch }
.table { border-collapse: collapse; width: 100%; min-width: 100%; table-layout: auto; font-size: 11px }
.table th, .table td { padding: 6px 4px; border: 1px solid #ddd; text-align: left; white-space: nowrap; min-width: 50px }
.table th { background: #f5f5f5; font-weight: bold }
.table caption { caption-side: top; font-weight: bold; margin-bottom: 8px; text-align: left }
.plot { text-align: center; margin: 10px 0; overflow: hidden }
.plot img { max-width: 100%; height: auto; border-radius: 4px; display: block; margin: 0 auto }
.metrics { overflow: auto }
.metrics pre { background: #f7fafc; padding: 8px; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; font-size: 10px; max-width: 100%; min-width: 0 }
.table-caption { font-weight: bold; margin-bottom: 6px; text-align: left }
.error { color: #c53030 }
p { line-height: 1.6; word-wrap: break-word }
</style>
'''

html = ['<html><head><meta charset="utf-8"><title>EDA Report</title>', css, '</head><body>']
html.append('<div class="container">')
html.append('<h1>EDA Report — Stroke Dataset</h1>')

# Top grid with stats and missing/class balance
html.append('<div class="grid">')
html.append('<div class="card">')
html.append('<h2>Summary statistics</h2>')
html.append(df_to_table_html(os.path.join(OUT_DIR, 'summary_statistics.csv'), index=False, header=True))
html.append('</div>')

html.append('<div class="card">')
html.append('<h2>Missing values</h2>')
html.append(df_to_table_html(os.path.join(OUT_DIR, 'missing_values.csv'), index=False, header=False))
html.append('<h2>Class balance</h2>')
html.append(df_to_table_html(os.path.join(OUT_DIR, 'class_balance.csv'), index=False, header=True))
html.append('</div>')
html.append('</div>')

# Plots
html.append('<div class="grid">')
for img in ['correlation_heatmap.png', 'roc_logistic.png', 'pr_logistic.png', 'roc_comparison.png', 'pr_comparison.png']:
    p = os.path.join(OUT_DIR, img)
    if os.path.exists(p):
        html.append('<div class="card plot">')
        html.append(f'<h3>{img.replace("_", " ").replace(".png", "")}</h3>')
        html.append(f'<img src="{img}" alt="{img}">')
        html.append('</div>')
html.append('</div>')

# Metrics
html.append('<div class="full card metrics">')
html.append('<h2>Model metrics comparison</h2>')
html.append('<h3>Baseline vs Balanced vs SMOTE</h3>')

# Combine all three runs into one comparison table
all_metrics = {
    'Baseline (Logistic)': baseline_metrics['logistic'],
    'Balanced (Logistic)': balanced_metrics['logistic'],
    'SMOTE+Balanced (Logistic)': smote_metrics['logistic']
}
html.append(format_metrics_table(all_metrics))
html.append('<p style="font-size: 10px; margin-top: 10px; color: #666;">Metrics shown: weighted averages from classification report. ROC-AUC available for Logistic model only.</p>')

# Best threshold
try:
    best = balanced_metrics['logistic'].get('best_threshold', None)
except Exception:
    best = None

if best is not None:
    html.append('<h3>Best threshold (balanced logistic)</h3>')
    html.append(f'<p>Best threshold: {best}</p>')

# Feature importance section (RandomForest from balanced run)
fi = balanced_metrics.get('random_forest', {}).get('feature_importances', [])
if fi:
    html.append('<h2>Feature importance (Balanced Random Forest)</h2>')
    html.append('<table class="table"><thead><tr><th>Feature</th><th>Importance</th></tr></thead><tbody>')
    for feature, imp in fi:
        html.append(f'<tr><td>{feature}</td><td>{imp:.4f}</td></tr>')
    html.append('</tbody></table>')

# Conclusion text
html.append('<h2>Conclusion</h2>')
html.append('<p>The models were evaluated on an imbalanced dataset. The class-weighted and SMOTE strategies improved recall for the minority stroke class, ' \
             'but at a cost in precision. RandomForest feature importances reveal the most predictive features. ' \
             'Use the precision-recall curve and best-F1 threshold in production to balance false positives and false negatives in line with clinical risk tolerance.</p>')

html.append('</div>')

html.append('</div>')
html.append('</body></html>')

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html))

print('Report written to', REPORT_PATH)
  