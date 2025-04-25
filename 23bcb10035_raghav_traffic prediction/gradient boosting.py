import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def predict_traffic(junction, hour, day_of_week, month):
    input_data = pd.DataFrame([[hour, day_of_week, month, junction]], columns=features)
    prediction = model.predict(input_data)[0]
    print(f"Traffic predicted at {hour}:00 is {int(prediction)} vehicles.")

# Load and preprocess data
df = pd.read_csv(r"C:\Users\DELL\.cache\kagglehub\datasets\fedesoriano\traffic-prediction-dataset\versions\1\traffic.csv")
df.rename(columns={"DateTime": "datetime", "Vehicles": "traffic_volume"}, inplace=True)
df["datetime"] = pd.to_datetime(df["datetime"])
df[["hour", "day_of_week", "month"]] = df["datetime"].apply(lambda x: [x.hour, x.dayofweek, x.month]).apply(pd.Series)

# Define features and target
features, target = ["hour", "day_of_week", "month", "Junction"], "traffic_volume"
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model
mae, rmse = mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)  # Adding R^2

predict_traffic(junction=1, hour=9, day_of_week=0, month=3)
print(f"MAE: {mae}\nRMSE: {rmse}\nR^2: {r2}")  # Displaying R^2

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Scatter Plot
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=axes[0])
axes[0].set(xlabel="Actual Traffic Volume", ylabel="Predicted Traffic Volume", title="Actual vs Predicted Traffic")

# Line Plot
axes[1].plot(y_test.values, label="Actual", linestyle='dashed')
axes[1].plot(y_pred, label="Predicted", alpha=0.75)
axes[1].set(xlabel="Sample Index", ylabel="Traffic Volume", title="Actual vs Predicted Over Time")
axes[1].legend()

# Histogram
sns.histplot(y_test - y_pred, bins=30, kde=True, ax=axes[2])
axes[2].set(xlabel="Prediction Error", ylabel="Frequency", title="Distribution of Errors")

# Display plots
plt.tight_layout()
plt.show()
