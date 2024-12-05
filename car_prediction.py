import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

file_path = 'C:/Users/p.ammu/Desktop/CAR PREDICTION/CAR-PREDICTION-USING-PYTHON/car data.csv'  # Replace with the path to your dataset
data = pd.read_csv(file_path)

print("Dataset Overview:")
print(data.head())
print("\nDataset Info:")
print(data.info())

data['Car_Age'] = 2024 - data['Year'] 
data = data.drop(['Year', 'Car_Name'], axis=1) 

data = pd.get_dummies(data, drop_first=True)

print("\nCleaned Dataset:")
print(data.head())

X = data.drop('Selling_Price', axis=1) 
y = data['Selling_Price']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
plt.xlabel('Actual Selling Prices')
plt.ylabel('Predicted Selling Prices')
plt.title('Actual vs Predicted Selling Prices')
plt.show()

import joblib
joblib.dump(model, 'car_price_model.pkl')

print("\nModel saved as 'car_price_model.pkl'.")
