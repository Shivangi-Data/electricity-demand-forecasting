import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as stats
import seaborn as sns
warnings.filterwarnings('ignore')

# Note: If using Spyder and plots are not appearing inline, go to Tools > Preferences > IPython Console > Graphics,
# and uncheck "Mute inline plotting" to enable inline plotting in the console.

# Load data with reduced memory usage by specifying dtypes
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

# Parse datetime with dayfirst=True to handle format mismatches
forecast_df['DATETIME'] = pd.to_datetime(forecast_df['DATETIME'], dayfirst=True, errors='coerce')
actual_df['DATETIME'] = pd.to_datetime(actual_df['DATETIME'], dayfirst=True, errors='coerce')
temperature_df['DATETIME'] = pd.to_datetime(temperature_df['DATETIME'], dayfirst=True, errors='coerce')

# Debug: Check for duplicates before merging
print("Number of duplicates in actual_df['DATETIME']:", actual_df['DATETIME'].duplicated().sum())
print("Number of duplicates in forecast_df['DATETIME']:", forecast_df['DATETIME'].duplicated().sum())
print("Number of duplicates in temperature_df['DATETIME']:", temperature_df['DATETIME'].duplicated().sum())

# Select the latest forecast
latest_forecast_df = forecast_df.sort_values(['DATETIME', 'LASTCHANGED']).groupby('DATETIME', as_index=False).tail(1)

# Merge actual demand with forecast demand using indices for efficiency
actual_df.set_index('DATETIME', inplace=True)
latest_forecast_df.set_index('DATETIME', inplace=True)
merged_data = actual_df.join(latest_forecast_df[['FORECASTDEMAND']], how='left')
merged_data.reset_index(inplace=True)

# Debug: Check for duplicates after merging
print("Number of duplicates in merged_data['DATETIME']:", merged_data['DATETIME'].duplicated().sum())

# Remove duplicates by taking the mean for numeric columns
numeric_cols = ['TOTALDEMAND', 'FORECASTDEMAND']
if merged_data['DATETIME'].duplicated().sum() > 0:
    merged_data = merged_data.groupby('DATETIME')[numeric_cols].mean().reset_index()
    print("Number of duplicates after grouping:", merged_data['DATETIME'].duplicated().sum())

# Handle duplicates in temperature_df
temperature_df = temperature_df.groupby('DATETIME', as_index=False)['TEMPERATURE'].mean()

# Debug: Verify no duplicates in temperature_df after grouping
print("Number of duplicates in temperature_df after grouping:", temperature_df['DATETIME'].duplicated().sum())

# Merge temperature data
merged_data = pd.merge(merged_data, temperature_df[['DATETIME', 'TEMPERATURE']], on='DATETIME', how='left')

# Debug: Check for duplicates after merging with temperature data
print("Number of duplicates after merging with temperature:", merged_data['DATETIME'].duplicated().sum())

# If duplicates exist, remove them
if merged_data['DATETIME'].duplicated().sum() > 0:
    merged_data = merged_data.groupby('DATETIME')[numeric_cols + ['TEMPERATURE']].mean().reset_index()
    print("Duplicates removed after merging with temperature. New duplicates:", merged_data['DATETIME'].duplicated().sum())

# Handle missing temperature values
merged_data['TEMPERATURE'] = merged_data['TEMPERATURE'].fillna(method='ffill')

# Set index and ensure hourly frequency
merged_data.set_index('DATETIME', inplace=True)
if merged_data.index.duplicated().sum() > 0:
    merged_data = merged_data[~merged_data.index.duplicated(keep='first')]
    print("Duplicate index labels removed. New duplicates:", merged_data.index.duplicated().sum())

merged_data.sort_index(inplace=True)
merged_data.to_csv(r"C:\Users\shiva\Documents\merged_dataset.csv", index=True)

expected_index = pd.date_range(start=merged_data.index.min(), end=merged_data.index.max(), freq='H')
merged_data = merged_data.reindex(expected_index, method='ffill')
merged_data.index.freq = 'H'

# Simplified Feature Engineering
def create_features(df, target_col='TOTALDEMAND'):
    df = df.copy()
    # Reduced lagged features for demand (only key lags)
    for lag in [1, 2, 24, 48, 72]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    # Reduced lagged features for temperature
    df['temp_lag_1'] = df['TEMPERATURE'].shift(1)
    # Time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    # Simplified rolling statistics
    df['rolling_mean_24h'] = df[target_col].rolling(window=24, min_periods=1).mean()
    # Key interaction feature
    df['lag_1_hour'] = df['lag_1'] * df['hour']
    # Interaction between temperature and hour
    df['temp_hour'] = df['TEMPERATURE'] * df['hour']
    return df

# Apply feature engineering
data_with_features = create_features(merged_data)
data_with_features.dropna(inplace=True)

# 70/30 train-test split
split_index = int(len(data_with_features) * 0.7)
train = data_with_features[:split_index]
test = data_with_features[split_index:]

# Define features and target
features = [col for col in train.columns if col not in ['TOTALDEMAND', 'FORECASTDEMAND', 'error', 'abs_error']]
X_train = train[features]
y_train = train['TOTALDEMAND']
X_test = test[features]
y_test = test['TOTALDEMAND']

# Faster hyperparameter tuning with RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=5,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
best_xgb_model = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)

# Forecast with the best model
xgb_forecast = best_xgb_model.predict(X_test)
xgb_forecast = pd.Series(xgb_forecast, index=y_test.index)

# Evaluation Metrics
mae_xgb = mean_absolute_error(y_test, xgb_forecast)
mse_xgb = mean_squared_error(y_test, xgb_forecast)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, xgb_forecast)
mape_xgb = np.mean(np.abs((y_test - xgb_forecast) / y_test)) * 100

print("\n📊 Optimized XGBoost with Temperature Evaluation Metrics:")
print(f"MAE: {mae_xgb:.2f}")
print(f"MSE: {mse_xgb:.2f}")
print(f"RMSE: {rmse_xgb:.2f}")
print(f"R²: {r2_xgb:.2f}")
print(f"MAPE: {mape_xgb:.2f}%")


plt.figure(figsize=(15,5))
plt.plot(y_test.index, y_test, label='Actual Demand')
plt.plot(y_test.index, xgb_forecast, label='XGBoost Forecast', color='red')
plt.title('Electricity Demand Forecast using XGBoost')
plt.xlabel('Time')
plt.ylabel('Demand (MW)')
plt.legend()
plt.tight_layout()
plt.show()

# SHAP Analysis (on a subset to speed up)
subset_size = 3000
X_test_subset = X_test.sample(n=min(subset_size, len(X_test)), random_state=42)
explainer = shap.TreeExplainer(best_xgb_model)
shap_values = explainer.shap_values(X_test_subset)

# SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_subset, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Subset)')
plt.tight_layout()
plt.show()

# SHAP Dependence Plot for TEMPERATURE with hour interaction
plt.figure(figsize=(10, 6))
shap.dependence_plot('TEMPERATURE', shap_values, X_test_subset, interaction_index='hour', show=False)
plt.title('SHAP Dependence Plot for TEMPERATURE with hour Interaction (Subset)')
plt.tight_layout()
plt.show()

# 7-Day Forecast Comparison Plot
seven_days_test = y_test[:168]
seven_days_xgb = xgb_forecast[:len(seven_days_test)]
seven_days_aemo = test['FORECASTDEMAND'].iloc[:len(seven_days_test)]
seven_days_aemo.index = seven_days_test.index

plt.figure(figsize=(12, 6))
plt.plot(seven_days_test.index, seven_days_test, label='Actual Demand', color='blue')
plt.plot(seven_days_test.index, seven_days_xgb, label='XGBoost Forecast', linestyle='--', color='purple')
plt.plot(seven_days_test.index, seven_days_aemo, label='AEMO Forecast', linestyle='--', color='gray')
plt.title('Forecast Comparison for 7 Days')
plt.xlabel('Date')
plt.ylabel('Demand (MW)')
date_ticks = pd.date_range(start=seven_days_test.index.min().normalize(), periods=7, freq='D')
plt.xticks(date_ticks, [d.strftime('%Y-%m-%d') for d in date_ticks], rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Residuals
residuals = y_test - xgb_forecast

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
plt.title('Distribution of Residuals ')
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

# AEMO 1-Hour Ahead Forecast Verification
verification_df = pd.DataFrame({
    'Forecast Time (t)': test['FORECASTDEMAND'].index,
    'AEMO Forecast at t': test['FORECASTDEMAND'].values,
    'Actual Demand at t+1': y_test.shift(-1).values
})

# Drop rows with NaNs due to shifting
verification_df.dropna(inplace=True)

# Show first 10 rows
print("\n🕒 Sample Verification of AEMO 1-Hour Ahead Forecast:\n")
print(verification_df.head(10).to_string(index=False))

# AEMO Forecast Evaluation
aemo_forecast_values = verification_df['AEMO Forecast at t'].values
actual_values_t1 = verification_df['Actual Demand at t+1'].values
mae_aemo = mean_absolute_error(actual_values_t1, aemo_forecast_values)
rmse_aemo = np.sqrt(mean_squared_error(actual_values_t1, aemo_forecast_values))
mape_aemo = np.mean(np.abs((actual_values_t1 - aemo_forecast_values) / actual_values_t1)) * 100
r2_aemo = r2_score(actual_values_t1, aemo_forecast_values)

print("\n📊 AEMO 1-Hour Ahead Forecast Evaluation Metrics:")
print(f"MAE:  {mae_aemo:.2f}")
print(f"RMSE: {rmse_aemo:.2f}")
print(f"MAPE: {mape_aemo:.2f}%")
print(f"R²:   {r2_aemo:.2f}")