# Import necessary libraries
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=700)).strftime('%Y-%m-%d')
data = yf.download("SPY", start=start_date, end=end_date, interval="1h")

close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

sequence_length = 60 
prediction_length = 168  
X, y = [], []
for i in range(sequence_length, len(scaled_data)-prediction_length):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i:i+prediction_length, 0])

X, y = np.array(X), np.array(y)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

def create_model(lstm_units, dropout_rate):
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=prediction_length))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

lstm_units_list = [30, 50, 80]
dropout_rates = [0.1, 0.2, 0.3]
batch_sizes = [16, 32, 64]

best_model = None
best_score = float('inf')

for lstm_units in lstm_units_list:
    for dropout_rate in dropout_rates:
        for batch_size in batch_sizes:
            print(f"Training with lstm_units={lstm_units}, dropout_rate={dropout_rate}, batch_size={batch_size}")
            model = create_model(lstm_units, dropout_rate)
            model.fit(X_train, y_train, epochs=10, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
            predictions = model.predict(X_test)
            mse_score = mean_squared_error(y_test, predictions)
            print(f"MSE Score: {mse_score}")
            if mse_score < best_score:
                best_score = mse_score
                best_model = model
            if best_score < 0.013:
                break
        if best_score < 0.013:
                break
    if best_score < 0.013:
                break

print("\nBest Model:")
print(f"MSE Score: {best_score}")

print("Predicted prices for the next week (hourly):")
current_date = datetime.today() + timedelta(days=1)
for i in range(0, prediction_length, 24):
    print("\nDate:", current_date.strftime('%Y-%m-%d'))
    daily_prices = predictions[0][i:i+24]
    for price in daily_prices:
        print(f"{price:.2f}")
    current_date += timedelta(days=1)