import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
sns.set(style='darkgrid')

# Load the dataset
data_path = os.path.join('data', 'creditcard.csv')
df = pd.read_csv(data_path)

# Basic info
print("✅ Dataset Loaded Successfully")
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# Class distribution
class_counts = df['Class'].value_counts()
print("\nClass Distribution:\n", class_counts)

# Plot class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Class Distribution (0 = Legit, 1 = Fraud)")
plt.savefig('notebooks/class_distribution.png')
plt.close()

# Summary stats for numeric fields
print("\nAmount and Time Summary:\n", df[['Amount', 'Time']].describe())

# Correlation matrix
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", vmax=1.0, square=True, linewidths=0.1)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig('notebooks/correlation_matrix.png')
plt.close()

print("📊 EDA plots saved as PNGs in the 'notebooks/' folder.")
