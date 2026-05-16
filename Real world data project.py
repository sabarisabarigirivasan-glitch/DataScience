import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("retail_sales.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert date column
df['Date'] = pd.to_datetime(df['Date'])

# Create month column
df['Month'] = df['Date'].dt.month

# Total sales by month
monthly_sales = df.groupby('Month')['Sales'].sum()

print("\nMonthly Sales:")
print(monthly_sales)

# Plot monthly sales
plt.figure(figsize=(8, 5))
monthly_sales.plot(kind='bar')
plt.title("Monthly Sales Analysis")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

# Prediction model
X = df[['Month']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print("\nModel Error:", mse)

# Predict future month sales
future = pd.DataFrame({'Month': [11, 12]})
future_sales = model.predict(future)

print("\nPredicted Future Sales:")
print(future_sales)