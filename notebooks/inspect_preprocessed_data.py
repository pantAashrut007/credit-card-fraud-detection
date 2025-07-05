# inspect_preprocessed_data.py

import numpy as np
import matplotlib.pyplot as plt

# Load preprocessed training data
train_data = np.load('data/train_processed.npz')
X_train = train_data['X']
y_train = train_data['y']

# Load preprocessed test data
test_data = np.load('data/test_processed.npz')
X_test = test_data['X']
y_test = test_data['y']

# Helper function to plot class distribution
def plot_class_distribution(y, title, subplot_index):
    unique, counts = np.unique(y, return_counts=True)
    plt.subplot(1, 2, subplot_index)
    plt.bar(unique.astype(str), counts, color=['green', 'red'])
    plt.title(title)
    plt.xlabel('Class (0 = Non-Fraud, 1 = Fraud)')
    plt.ylabel('Count')
    for i in range(len(counts)):
        plt.text(i, counts[i] + 100, str(counts[i]), ha='center')

# Plot both training and test class distributions
plt.figure(figsize=(12, 5))
plot_class_distribution(y_train, 'Training Set (After SMOTE)', 1)
plot_class_distribution(y_test, 'Test Set (Original Distribution)', 2)

plt.tight_layout()
plt.show()