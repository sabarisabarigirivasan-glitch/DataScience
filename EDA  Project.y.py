import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset.csv")

# Display first 5 rows
print("Dataset Preview:")
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing numeric values
df.fillna(df.mean(numeric_only=True), inplace=True)

# ----------------------------
# Data Visualization
# ----------------------------

# Histogram
df.hist(figsize=(12, 8))
plt.suptitle("Histogram of Features")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

# Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df.select_dtypes(include=np.number))
plt.title("Boxplot for Numerical Features")
plt.show()

print("EDA Completed Successfully")