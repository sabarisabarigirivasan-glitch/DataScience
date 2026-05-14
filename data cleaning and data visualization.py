import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("sales_data.csv")

# Display dataset
print("Original Dataset")
print(df.head())

# --------------------------------
# 1. Handling Missing Values
# --------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing numeric values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# --------------------------------
# 2. Remove Duplicates
# --------------------------------
df.drop_duplicates(inplace=True)

# --------------------------------
# 3. Remove Outliers using IQR
# --------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower_limit) & (df[col] <= upper_limit)]

print("\nCleaned Dataset Shape:", df.shape)

# Save cleaned file
df.to_csv("cleaned_data.csv", index=False)

# --------------------------------
# 4. Data Visualization
# --------------------------------

# Histogram
plt.figure(figsize=(8,5))
df[numeric_cols[0]].hist(bins=20)
plt.title("Histogram")
plt.xlabel(numeric_cols[0])
plt.ylabel("Frequency")
plt.show()

# Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot")
plt.show()

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Bar chart for categorical column
cat_cols = df.select_dtypes(include='object').columns

if len(cat_cols) > 0:
    plt.figure(figsize=(8,5))
    df[cat_cols[0]].value_counts().plot(kind='bar')
    plt.title("Category Count")
    plt.xlabel(cat_cols[0])
    plt.ylabel("Count")
    plt.show()

print("Project Completed Successfully")