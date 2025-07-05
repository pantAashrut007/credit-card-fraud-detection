import os
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Load test data from preprocessed .npz file, NOT CSV
test_data_path = os.path.join('data', 'test_processed.npz')
test_data = np.load(test_data_path, allow_pickle=True)
X_test, y_test = test_data['X'], test_data['y']

# Models to evaluate with optional scaler paths
model_files = {
    "Logistic Regression": {
        "model": "models/trained_models/logistic_regression.joblib",
        "scaler": None
    },
    "Random Forest": {
        "model": "models/trained_models/rf_model.joblib",
        "scaler": "models/trained_models/rf_scaler.joblib"
    },
    "XGBoost": {
        "model": "models/trained_models/xgboost_model.pkl",
        "scaler": None
    },
    "LightGBM": {
        "model": "models/trained_models/lightgbm_model.pkl",
        "scaler": None
    }
}

results = []

for name, files in model_files.items():
    model_path = files["model"]
    scaler_path = files["scaler"]

    if os.path.exists(model_path):
        model = joblib.load(model_path)

        if scaler_path and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        results.append((name, precision, recall, f1, roc_auc))
    else:
        print(f"[!] Model not found: {model_path}")

print("\n=== Model Comparison ===")
print("{:<20} {:<10} {:<10} {:<10} {:<10}".format("Model", "Precision", "Recall", "F1", "ROC-AUC"))
print("-" * 60)
for name, p, r, f1, roc in results:
    print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(name, p, r, f1, roc))
