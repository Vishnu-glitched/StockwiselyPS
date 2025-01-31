import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from home.linear_regression_gd import LinearRegressionGD

# Load data
df = pd.read_excel("yahoo_data.xlsx")

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate EMA
def calculate_ema(data, window=14):
    return data.ewm(span=window, adjust=False).mean()

# Add RSI and EMA columns
df["RSI"] = calculate_rsi(df["Close*"])
df["EMA"] = calculate_ema(df["Close*"])
df.dropna(inplace=True)

# Prepare data
features = ["RSI", "EMA", "Open", "High", "Low", "Volume"]
X = df[features].values
y = df["Close*"].values.reshape(-1, 1)

# Scale features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Add bias term (intercept) to X
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Train the model
lr_gd = LinearRegressionGD(learning_rate=0.009, epochs=10000)
lr_gd.fit(X_train, y_train)

import joblib
from home.linear_regression_gd import LinearRegressionGD

# Assuming 'model' is an instance of LinearRegressionGD
joblib.dump(lr_gd, 'static/Linearregressionmodel1')
