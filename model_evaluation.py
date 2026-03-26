import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from Disease_Prediction_from_medical_data import preprocess, train_and_eval


def evaluate_with_options(data_path='healthcare-dataset-stroke-data.csv', balance=False, smote=False):
    df = pd.read_csv(data_path)
    X, y = preprocess(df)
    metrics = train_and_eval(X, y, use_class_weight=balance, use_smote=smote)

    # Use balanced logistic model for confusion matrix demonstration
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    if smote:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced' if balance else None)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    os.makedirs('eda_outputs', exist_ok=True)
    pd.DataFrame(cm, index=['Actual_0', 'Actual_1'], columns=['Pred_0', 'Pred_1']).to_csv('eda_outputs/confusion_matrix.csv')

    with open('eda_outputs/classification_report.txt', 'w') as f:
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(report)

    print('Confusion matrix saved to eda_outputs/confusion_matrix.csv')
    print('Classification report saved to eda_outputs/classification_report.txt')
    return metrics


if __name__ == '__main__':
    print('Running model evaluation (balanced+smote recommended)')
    m = evaluate_with_options(balance=True, smote=True)
    print('Evaluation metrics:', m)
