import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

#Do not run as it will take a lot of time to run


# Load preprocessed data
train_data = np.load("data/train_processed.npz")
test_data = np.load("data/test_processed.npz")

X_train, y_train = train_data['X'], train_data['y']
X_test, y_test = test_data['X'], test_data['y']

# Train the model
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print("=== Support Vector Machine (RBF Kernel) ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# Save the model
joblib.dump(model, "models/trained_models/svm_model.pkl")
