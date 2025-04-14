import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Load the dataset
data = pd.read_csv('SynchronousMachine.csv')
# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())
# Define features (X) and target (y)
X = data[['I_y', 'PF', 'e_PF', 'd_if']] # Features
y = data['I_f'] # Target variable (Excitation current)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the linear regression model
model = LinearRegression()
# Train the model on the training data
model.fit(X_train, y_train)
# Predict the excitation current for the test data
y_pred = model.predict(X_test)
# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Display the evaluation metrics in a DataFrame (table format)
metrics = pd.DataFrame({
"Metric": ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R-squared (R2)"],
"Value": [mse, mae, r2]
})
print("\nModel Evaluation Metrics:")
print(metrics)
# Display model coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nModel Coefficients:")
print(coefficients)
# Combine test set and predictions for comparison in a table
results = pd.DataFrame({
"Actual I_f": y_test.values,
"Predicted I_f": y_pred
}).reset_index(drop=True)
print("\nActual vs Predicted I_f:")
print(results)
# Plotting Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.plot(results["Actual I_f"], label='Actual I_f', color='b', marker='o')
plt.plot(results["Predicted I_f"], label='Predicted I_f', color='r', linestyle='--', marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Excitation Current (I_f)")
plt.title("Actual vs Predicted Excitation Current (I_f)")
plt.legend()
plt.grid(True)
plt.show()