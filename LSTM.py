# Import necessary libraries
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# 1. Data Collection
data = yf.download("SPY", start="2021-09-21", end="2023-09-21")

# 2. Data Preprocessing
close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

X, y = [], []
for i in range(60, len(scaled_data)-7):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i:i+7, 0])

X, y = np.array(X), np.array(y)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 3. Model Creation
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=7))
model.compile(optimizer='adam', loss='mean_squared_error')

# 4. Training
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

# 5. Prediction
last_60_days = scaled_data[-60:]
last_60_days = last_60_days.reshape(-1, 60, 1)
predicted_prices = model.predict(last_60_days)
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(1, -1))

print("Predicted prices for the next week:")
print(predicted_prices)

