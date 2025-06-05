# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 21:13:47 2025

@author: shiva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as stats
import seaborn as sns
# -----------------------------
# 📥 Load and preprocess data
# -----------------------------

forecast_df = pd.read_csv(
    r"C:\Users\shiva\Documents\OneDrive_1_2-19-2025\forecastdemand_nsw.csv",
    parse_dates=['DATETIME', 'LASTCHANGED'],
    usecols=['DATETIME', 'LASTCHANGED', 'FORECASTDEMAND'],
    dtype={'FORECASTDEMAND': np.float32}
)
actual_df = pd.read_csv(
    r"C:\Users\shiva\Documents\OneDrive_1_2-19-2025\totaldemand_nsw.csv",
    parse_dates=['DATETIME'],
    usecols=['DATETIME', 'TOTALDEMAND'],
    dtype={'TOTALDEMAND': np.float32}
)
temperature_df = pd.read_csv(
    r"C:\Users\shiva\Documents\OneDrive_1_2-19-2025\temperature_nsw.csv",
    parse_dates=['DATETIME'],
    usecols=['DATETIME', 'TEMPERATURE'],
    dtype={'TEMPERATURE': np.float32}
)

# Parse datetime correctly
forecast_df['DATETIME'] = pd.to_datetime(forecast_df['DATETIME'], dayfirst=True, errors='coerce')
actual_df['DATETIME'] = pd.to_datetime(actual_df['DATETIME'], dayfirst=True, errors='coerce')
temperature_df['DATETIME'] = pd.to_datetime(temperature_df['DATETIME'], dayfirst=True, errors='coerce')

# Clean and merge data
latest_forecast_df = forecast_df.sort_values(['DATETIME', 'LASTCHANGED']).groupby('DATETIME', as_index=False).tail(1)
actual_df.set_index('DATETIME', inplace=True)
latest_forecast_df.set_index('DATETIME', inplace=True)
merged_data = actual_df.join(latest_forecast_df[['FORECASTDEMAND']], how='left').reset_index()

temperature_df = temperature_df.groupby('DATETIME', as_index=False)['TEMPERATURE'].mean()
merged_data = pd.merge(merged_data, temperature_df[['DATETIME', 'TEMPERATURE']], on='DATETIME', how='left')
merged_data['TEMPERATURE'] = merged_data['TEMPERATURE'].fillna(method='ffill')
merged_data.set_index('DATETIME', inplace=True)
merged_data.sort_index(inplace=True)
# -----------------------------
# 📥 Load and preprocess data
# -----------------------------

# Assuming you have already loaded and merged the datasets

merged_data = merged_data.asfreq('H')

# Check if data is stationary
from statsmodels.tsa.stattools import adfuller

result = adfuller(merged_data['TOTALDEMAND'].dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] < 0.5:
    print("Data is stationary")
else:
    print("Data is not stationary")

# -------------------------------------
# 🧠 Feature engineering
# -------------------------------------

def create_features(df, target_col='TOTALDEMAND'):
    df = df.copy()
    for lag in [1, 2, 24, 48, 72]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    df['temp_lag_1'] = df['TEMPERATURE'].shift(1)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['rolling_mean_24h'] = df[target_col].rolling(window=24, min_periods=1).mean()
    df['lag_1_hour'] = df['lag_1'] * df['hour']
    df['temp_hour'] = df['TEMPERATURE'] * df['hour']
    return df

data_with_features = create_features(merged_data)
data_with_features.dropna(inplace=True)



# -------------------------------------
# ✂️ Train-test split
# -------------------------------------

split_index = int(len(data_with_features) * 0.7)
train = data_with_features[:split_index]
test = data_with_features[split_index:]

# -------------------------------------
# 🏁 Model Training - ARIMAX via SARIMAX
# -------------------------------------

# Define target and exogenous variables
target_col = 'TOTALDEMAND'
exog_cols = ['TEMPERATURE', 'hour', 'day_of_week', 'lag_1', 'lag_24', 'temp_lag_1']

X_train = train[exog_cols]
y_train = train[target_col]
X_test = test[exog_cols]
y_test = test[target_col]

from statsmodels.tsa.stattools import adfuller

result = adfuller(merged_data['TOTALDEMAND'].dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])
if result[1]<0.5:
    print("Data is stationary")
else:
    print("Data is not stationary")
    


# Fit ARIMAX model using SARIMAX
model = SARIMAX(
    endog=y_train,
    exog=X_train,
    order=(2, 0, 2),  # These can be tuned or selected via auto_arima
    enforce_stationarity=False,
    enforce_invertibility=False
)
model_fit = model.fit(disp=False)

# Forecast
forecast = model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=X_test)
forecast = pd.Series(forecast, index=y_test.index)

# -------------------------------------
# 📊 Evaluation Metrics
# -------------------------------------

mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, forecast)
mape = np.mean(np.abs((y_test - forecast) / y_test)) * 100

print("\n📈 ARIMAX Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")
print(f"MAPE: {mape:.2f}%")

# -------------------------------------
# 📉 Plot Results
# -------------------------------------

plt.figure(figsize=(15,5))
plt.plot(y_test.index, y_test, label='Actual Demand')
plt.plot(y_test.index, forecast, label='ARIMAX Forecast', color='orange')
plt.title('Electricity Demand Forecast using ARIMAX')
plt.xlabel('Time')
plt.ylabel('Demand (MW)')
plt.legend()
plt.tight_layout()
plt.show()

# Slice 7 days (168 hours)

# 7-Day Forecast Comparison Plot (Simple)
# Last 7 days comparison
# Slice 7 days (168 hours) from forecast and test
seven_days_arima = forecast[:168]
seven_days_test = y_test[:168]  # Only the target column
seven_days_aemo = merged_data.loc[seven_days_test.index, 'FORECASTDEMAND']

plt.figure(figsize=(16, 8), dpi=120)
plt.plot(seven_days_test.index, seven_days_test, label='Actual Demand', linewidth=2, color='blue')
plt.plot(seven_days_test.index, seven_days_arima, label='ARIMA Forecast', linestyle='--', color='orange', linewidth=2)
plt.plot(seven_days_test.index, seven_days_aemo, label='AEMO Forecast', linestyle='--', color='gray', linewidth=2)
plt.title('Forecast Comparison for Last 7 Days', fontsize=16, fontweight='bold')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Demand (MW)', fontsize=14)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Define the start index for the 7-day period (e.g., shift to a different 168-point chunk)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, DateFormatter

# -------------------------------------
# 📉 7-Day Forecast Comparison Plot
# -------------------------------------

# Print index ranges to verify available dates
print("y_test index range:", y_test.index.min(), "to", y_test.index.max())
print("merged_data index range:", merged_data.index.min(), "to", merged_data.index.max())
print("forecast index range:", forecast.index.min(), "to", forecast.index.max())

# Select a valid start date from the test set
# Use the first date in y_test or a specific date within y_test.index
start_date = '2011-11-07'  # Start from the beginning of the test set
# Alternatively, specify a known date in y_test, e.g., '2023-01-01' (adjust based on your data)
# start_date = '2023-01-01'
end_date = pd.to_datetime(start_date) + pd.Timedelta(days=7)

# Select the 7-day period
seven_days_test = y_test.loc[start_date:end_date]
seven_days_aemo = merged_data.loc[start_date:end_date, 'FORECASTDEMAND']
seven_days_arima = forecast.loc[start_date:end_date]  # Use forecast, not seven_days_arima

# Print shapes to diagnose
print("seven_days_test shape:", seven_days_test.shape)
print("seven_days_aemo shape:", seven_days_aemo.shape)
print("seven_days_arima shape:", seven_days_arima.shape)

# Check if any array is empty
if seven_days_test.empty or seven_days_aemo.empty or seven_days_arima.empty:
    raise ValueError(f"One or more data arrays are empty for date range {start_date} to {end_date}. Check the date range or data availability.")

# Plotting
plt.figure(figsize=(16, 8), dpi=120)
plt.plot(seven_days_test.index, seven_days_test, label='Actual Demand', linewidth=2, color='blue')
plt.plot(seven_days_test.index, seven_days_arima, label='ARIMA Forecast', linestyle='--', color='orange', linewidth=2)
plt.plot(seven_days_test.index, seven_days_aemo, label='AEMO Forecast', linestyle='--', color='gray', linewidth=2)
plt.title(f'Forecast Comparison for 7 Days ', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Demand (MW)', fontsize=14)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Format x-axis to show one tick per day
plt.gca().xaxis.set_major_locator(DayLocator())  # One tick per day
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))  # Format as YYYY-MM-DD
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
# Plotting
plt.figure(figsize=(16, 8), dpi=120)
plt.plot(seven_days_test.index, seven_days_test, label='Actual Demand', linewidth=2, color='blue')
plt.plot(seven_days_test.index, seven_days_arima, label='ARIMA Forecast', linestyle='--', color='orange', linewidth=2)
plt.plot(seven_days_test.index, seven_days_aemo, label='AEMO Forecast', linestyle='--', color='gray', linewidth=2)
plt.title(f'Forecast Comparison for 7 Days Starting {start_date}', fontsize=16, fontweight='bold')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Demand (MW)', fontsize=14)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Residuals
residuals = y_test - forecast

# Residual Summary
print("\n📉 Residual Summary Statistics:")
print(residuals.describe())

# Residual Plot
plt.figure(figsize=(12, 4))
plt.plot(residuals.index, residuals, label='Residuals', color='red')
plt.axhline(0, linestyle='--', color='black')
plt.title('Residuals Over Time')
plt.xlabel('Date')
plt.ylabel('Residuals (MW)')
plt.xticks(residuals.index[::24], [d.strftime('%Y-%m-%d') for d in residuals.index[::24]], rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Ljung-Box Test
ljung_box_result = acorr_ljungbox(residuals, lags=[24], return_df=True)
print("\n🔍 Ljung-Box Test for Residual Autocorrelation (lag=24):")
print(ljung_box_result)

# ACF and PACF Plots
plt.figure(figsize=(12, 4))
plot_acf(residuals, lags=48)
plt.title('ACF of Residuals')
plt.xlabel('Lag (Hours)')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plot_pacf(residuals, lags=48)
plt.title('PACF of Residuals')
plt.xlabel('Lag (Hours)')
plt.ylabel('Partial Autocorrelation')
plt.grid(True)
plt.tight_layout()
plt.show()

# Histogram of Residuals
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=30, kde=True, color='purple', edgecolor='black')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals (MW)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Q-Q Plot for Residuals
plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.grid(True)
plt.tight_layout()
plt.show()

