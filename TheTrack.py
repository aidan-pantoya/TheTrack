# -*- coding: utf-8 -*-
"""
Author: Aidan Pantoya, ChatGPT
9/2023
Application predicts the next day for a given ticker symbol
"""
import datetime
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas_market_calendars as mcal
import pytz
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


target_accuracy = 0.997

SYMBL = 'SPY'

tz = pytz.timezone('America/New_York')
now = datetime.datetime.now(tz)


now_str = now.strftime('%Y-%m-%d_%H-%M-%S')
save_graph = f'C:/Users/Aidan Pantoya/Pictures/Market/{SYMBL}{now_str}.jpg'

data = yf.download(SYMBL, period="2y", interval="1h")
data.index = data.index.tz_convert('America/New_York')
prices = data[['Open', 'High', 'Low', 'Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

best_score = float('-inf')
best_window = None
best_model = None
best_mape = float('inf')

def is_trading_day(date):
    """
    Returns True if the given date is a trading day, else False.
    """
    nyse = mcal.get_calendar('NYSE')
    naive_date = date.tz_localize(None)
    trading_days = nyse.valid_days(start_date=naive_date, end_date=naive_date)
    return not trading_days.empty

def mean_absolute_percentage_error(y_true, y_pred): 
    """Compute MAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_dataset(dataset, look_back=1):
    """Function to create a dataset for LSTM."""
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, :])  
    return np.array(X), np.array(Y)

def predict_next_24_hours(model, last_sequence, scaler, look_back):
    """Function to predict the next 24 hours."""
    next_24 = []
    current_seq = last_sequence.copy()
    for _ in range(24):
        prediction = model.predict(current_seq.reshape(1, look_back, 4))
        next_24.append(scaler.inverse_transform(prediction)[0])
        current_seq = np.vstack([current_seq[1:], prediction])
    return np.array(next_24)

BigWindow = int(8 * 52 * 4)
StartWindow = int(8 * 3)
wndwCnt = int(BigWindow/8)

early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

best_accuracy = 0

learning_rates = np.arange(0.0001, 0.0012, 0.0002)
batch_sizes = [8,16, 32, 64]  
window_sizes = range(BigWindow,StartWindow, int(-1 * wndwCnt) )

for window_size in window_sizes:
    X, Y = create_dataset(scaled_data, window_size)
    X = X.reshape(X.shape[0], X.shape[1], 4)
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = Y[:train_size], Y[train_size:]
    for lr in learning_rates:
        optimizer = Adam(learning_rate=lr)
        model = Sequential()
        model.add(LSTM(50, input_shape=(window_size, 4)))
        model.add(Dense(4))
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        for batch in batch_sizes:
            history = model.fit(X_train, y_train, epochs=24, batch_size=batch, validation_split=0.3, callbacks=[early_stop])
            predictions = model.predict(X_test)
            dummy_preds = np.zeros((predictions.shape[0], 4))
            dummy_preds[:, :] = predictions
            inverse_scaled_preds = scaler.inverse_transform(dummy_preds)[:, -1]
            test_predict = inverse_scaled_preds
            dummy_true = np.zeros((y_test.shape[0], 4))
            dummy_true[:, :] = y_test
            true_values = scaler.inverse_transform(dummy_true)[:, -1]
            mape = mean_absolute_percentage_error(true_values, test_predict)
            accuracy = 1 - mape / 100 
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_window = window_size
                best_model = model 
                best_mape = mape 
            
            print(f"LR: {lr}, Batch Size: {batch}, Window Size: {window_size}, Accuracy: {accuracy:.5f}")
            
            if best_accuracy >= target_accuracy:
                break
        if best_accuracy >= target_accuracy:
            break
    if best_accuracy >= target_accuracy:
        break

print(f"Best Parameters: Window Size: {best_window}, MAPE: {best_mape:.2f}%, Accuracy: {100-best_mape:.2f}%")

last_sequence = scaled_data[-best_window:]
next_24_predictions = predict_next_24_hours(best_model, last_sequence, scaler, best_window)

last_date = data.index[-1]
next_24_dates = []
last_date += datetime.timedelta(hours=1)  
while len(next_24_dates) < 24:
    if last_date.tzinfo is None:
        last_date = last_date.tz_localize('America/New_York')
    if is_trading_day(last_date):
        next_24_dates.append(last_date)
    last_date += datetime.timedelta(hours=1)

all_dates = data.index.to_list()

one_week_ago = last_date - datetime.timedelta(days=7)
week_indexes = [i for i, date in enumerate(all_dates) if date >= one_week_ago]
week_prices = prices[week_indexes]
week_dates = [all_dates[i] for i in week_indexes]

week_sequences = [scaled_data[i - best_window:i] for i in week_indexes]
week_sequences = np.array(week_sequences)
week_sequences = week_sequences.reshape(week_sequences.shape[0], best_window, 4)

week_predictions = best_model.predict(week_sequences)
week_predictions = scaler.inverse_transform(week_predictions)

plt.figure(figsize=(14, 7))
plt.plot(week_dates, week_prices, label='True Prices', color='blue')
plt.plot(week_dates, week_predictions, label='Predicted Prices for Past Week', color='red')
plt.plot(next_24_dates, next_24_predictions, label='Predictions for Next 24 Hours', color='green', linestyle='--')

plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.title(f'Tckr: {SYMBL}, Best Parameters: Window Size: {best_window}, MAPE: {best_mape:.2f}%, Accuracy: {100-best_mape:.2f}%')
plt.tight_layout()
plt.savefig(save_graph, dpi=300)
plt.show()