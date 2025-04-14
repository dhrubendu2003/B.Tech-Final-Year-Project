import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import plot_tree
# Load the dataset
data = pd.read_csv('SynchronousMachine.csv')
# Display dataset preview
print("Dataset Preview:")
print(data.head())
# Define features (X) and target (y)
X = data[['I_y', 'PF', 'e_PF', 'd_if']] # Input features
y = data['I_f'] # Target variable (Excitation current)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Decision Tree Regressor model
model = DecisionTreeRegressor(random_state=42)
# Train the model on the training data
model.fit(X_train, y_train)
# Predict the excitation current on the test data
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Print the evaluation metrics
print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")
# Perform cross-validation to check generalization
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("\nCross-Validation R2 Scores:", cv_scores)
print(f"Average R2 Score: {cv_scores.mean()}")
# Combine test set and predictions for comparison
results = pd.DataFrame({
 "Actual I_f": y_test.values,
 "Predicted I_f": y_pred
}).reset_index(drop=True)
print("\nActual vs Predicted Excitation Current (I_f):")
print(results)
# Visualize Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.plot(results["Actual I_f"], label='Actual I_f', color='blue', marker='o')
plt.plot(results["Predicted I_f"], label='Predicted I_f', color='red', linestyle='--',
marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Excitation Current (I_f)")
plt.title("Actual vs Predicted Excitation Current (Decision Tree Regressor)")
plt.legend()
plt.grid(True)
plt.show()
# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, filled=True, fontsize=10)
plt.title("Decision Tree Structure")
plt.show()