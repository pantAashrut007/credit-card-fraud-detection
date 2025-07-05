import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load the cleaned data (update the path as needed)
data_path = os.path.join('data', 'creditcard.csv')
df = pd.read_csv(data_path)
# df = pd.read_csv("data/creditcard.csv")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for future inference use
joblib.dump(scaler, "models/trained_models/scaler.pkl")

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Log class distribution
print("Before resampling:", np.bincount(y_train))
print("After resampling:", np.bincount(y_train_resampled))

# Save preprocessed data for model training
np.savez("data/train_processed.npz", X=X_train_resampled, y=y_train_resampled)
np.savez("data/test_processed.npz", X=X_test_scaled, y=y_test)