import os
import pandas as pd
from Disease_Prediction_from_medical_data import run_eda, preprocess, train_and_eval


def test_run_eda_creates_outputs(tmp_path):
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    out_dir = tmp_path / 'eda_test'
    out_dir = str(out_dir)
    result = run_eda(df, out_dir=out_dir)
    assert result['rows'] == df.shape[0]
    assert os.path.exists(os.path.join(out_dir, 'summary_statistics.csv'))
    assert os.path.exists(os.path.join(out_dir, 'correlation_heatmap.png'))


def test_preprocess_and_train():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    X, y = preprocess(df)
    metrics = train_and_eval(X, y, use_class_weight=False, use_smote=False)
    assert 'logistic' in metrics
    assert 'roc_auc' in metrics['logistic']
