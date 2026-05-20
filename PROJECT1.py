import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#DATASET

df = pd.read_csv(
    r"D:\MCA 2ND SEM\ml\powerdemand_5min_2021_to_2024_with weather.csv"
)
df.head()
print(df)

# DATA PREPROCESSING


df['datetime'] = pd.to_datetime(df['datetime'])


print("Missing Values:\n")
print(df.isnull().sum())

df = df.dropna()


# FEATURES AND TARGET


features = [
    'temp',
    'dwpt',
    'rhum',
    'wdir',
    'wspd',
    'pres',
    'year',
    'month',
    'day',
    'hour',
    'minute',
    'moving_avg_3'
]

X = df[features]

y = df['Power demand']


# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# RANDOM FOREST MODEL


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

# PREDICTION


y_pred = model.predict(X_test)


# MODEL EVALUATION

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"MAE       : {mae:.2f}")

print(f"MSE       : {mse:.2f}")

print(f"RMSE      : {rmse:.2f}")

print(f"R2 Score  : {r2:.4f}")


# FEATURE IMPORTANCE


importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(feature_importance)

# ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(12,6))

plt.plot(
    y_test.values[:100],
    label='Actual Demand'
)

plt.plot(
    y_pred[:100],
    label='Predicted Demand'
)

plt.xlabel("Samples")

plt.ylabel("Power Demand")

plt.title("Actual vs Predicted Electricity Demand")

plt.legend()

plt.grid(True)

plt.show()


# 5 DAYS FUTURE FORECAST



print("5 DAYS POWER DEMAND FORECAST")


# Last datetime
last_datetime = df['datetime'].max()

# Create future timestamps
future_dates = pd.date_range(
    start=last_datetime,
    periods=1440,
    freq='5min'
)

# Create dataframe
future_df = pd.DataFrame({
    'datetime': future_dates
})

# Time features
future_df['year'] = future_df['datetime'].dt.year

future_df['month'] = future_df['datetime'].dt.month

future_df['day'] = future_df['datetime'].dt.day

future_df['hour'] = future_df['datetime'].dt.hour

future_df['minute'] = future_df['datetime'].dt.minute

# Weather features using average values
future_df['temp'] = df['temp'].mean()

future_df['dwpt'] = df['dwpt'].mean()

future_df['rhum'] = df['rhum'].mean()

future_df['wdir'] = df['wdir'].mean()

future_df['wspd'] = df['wspd'].mean()

future_df['pres'] = df['pres'].mean()

future_df['moving_avg_3'] = df['moving_avg_3'].mean()

# Future feature dataset
X_future = future_df[features]

# Predict future demand
future_predictions = model.predict(X_future)

# Add prediction column
future_df['Predicted Demand'] = future_predictions


# PRINT FUTURE PREDICTIONS


print(
    future_df[
        ['datetime', 'Predicted Demand']
    ].head(20)
)


# SAVE CSV FILE


future_df.to_csv(
    "5_days_power_demand_prediction.csv",
    index=False
)

print("\nPrediction file saved successfully!")

# FUTURE FORECAST GRAPH


plt.figure(figsize=(15,6))

plt.plot(
    future_df['datetime'],
    future_df['Predicted Demand']
)

plt.title("5-Day Electricity Demand Forecast")

plt.xlabel("Datetime")

plt.ylabel("Predicted Power Demand")

plt.xticks(rotation=45)

plt.grid(True)

plt.tight_layout()

plt.show()