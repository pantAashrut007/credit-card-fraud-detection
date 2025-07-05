import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load preprocessed data
train_data = np.load("data/train_processed.npz")
test_data = np.load("data/test_processed.npz")

X_train, y_train = train_data['X'], train_data['y']
X_test, y_test = test_data['X'], test_data['y']

# Train the model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print("=== XGBoost Classifier ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# Save the model
joblib.dump(model, "models/trained_models/xgboost_model.pkl")
