import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from yahooquery import Ticker
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta

tickers = ['TSLA']

for ticker in tickers:
    date = '2024-04-29'
    
    endDate = datetime.strptime(date, '%Y-%m-%d')
    startDate = endDate - timedelta(days=2*365)
    startDate = startDate.strftime('%Y-%m-%d')
    
    data = yf.download(ticker, start=startDate, end=endDate)
    
    epcs = 5000
    
    stock = Ticker(ticker)
    earnings = stock.earnings[ticker]['earningsChart']['quarterly']
    earnings_df = pd.DataFrame(earnings)
    
    def parse_quarter_date(qtr):
        year, quarter = int(qtr[-4:]), qtr[:-4]
        quarter_month_map = {'1Q': '01-01', '2Q': '04-01', '3Q': '07-01', '4Q': '10-01'}
        date_str = f"{year}-{quarter_month_map[quarter]}"
        return pd.to_datetime(date_str)
    
    earnings_df['date'] = earnings_df['date'].apply(parse_quarter_date)
    earnings_df.set_index('date', inplace=True)
    
    data = data[['Close', 'Volume']].merge(earnings_df[['actual']], left_index=True, right_index=True, how='left')
    data['actual'].fillna(method='ffill', inplace=True)
    data.fillna(0, inplace=True)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close', 'Volume', 'actual']])
    
    training_data_len = math.ceil(len(scaled_data) * 0.75)
    x_train, y_train = [], []
    for i in range(60, training_data_len):
        x_train.append(scaled_data[i-60:i])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 3))
    
    x_test, y_test = [], []
    test_data = scaled_data[training_data_len - 60:]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i])
        y_test.append(test_data[i, 0])
    x_test = np.array(x_test)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 3))
    y_test = np.array(y_test)
    
    model = keras.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        layers.LSTM(250, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        layers.LSTM(100, return_sequences=False),
        layers.Dense(50),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=8, epochs=epcs)
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 2))), axis=1))[:, 0]
    
    rmse = np.sqrt(np.mean((predictions - scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 2))), axis=1))[:, 0]) ** 2))
    print('Root mean squared error:', rmse)
    
    validation = data['Close'][training_data_len:].copy()
    validation = pd.DataFrame(validation)
    validation.reset_index(drop=True, inplace=True)
    validation['Predictions'] = np.ravel(predictions)
    
    future_steps = 30
    last_60_days = scaled_data[-60:]
    current_batch = last_60_days.reshape((1, last_60_days.shape[0], last_60_days.shape[1]))
    
    future_predictions = []
    
    for i in range(future_steps):
        future_pred = model.predict(current_batch)[0, 0]
        future_predictions.append(future_pred)
        future_pred = np.array([[future_pred, 0, 0]])
        current_batch = np.append(current_batch[:, 1:, :], future_pred.reshape((1, 1, 3)), axis=1)
    
    future_predictions = scaler.inverse_transform(np.concatenate((np.array(future_predictions).reshape(-1, 1), np.zeros((future_steps, 2))), axis=1))[:, 0]
    validation_extended = pd.concat([validation, pd.DataFrame(future_predictions, columns=['Predictions'])], ignore_index=True)
    
    fig, axs = plt.subplots(2, figsize=(14, 14))
    
    axs[0].plot(validation['Close'], label='Actual Prices', color='blue')
    axs[0].plot(validation_extended['Predictions'], label='Predicted Prices', color='red')
    axs[0].set_title(f'Actual and Predicted Stock Prices: {ticker} - {date}, 50-250-50')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    axs[0].legend()
    
    axs[1].plot(data['Volume'][training_data_len:].reset_index(drop=True), label='Volume', color='green')
    axs[1].set_title('Volume')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Volume')
    axs[1].legend() 
    
    #axs[2].plot(data['actual'][training_data_len:].reset_index(drop=True), label='Earnings', color='purple')
    #axs[2].set_title('Earnings')
    #axs[2].set_xlabel('Date')
    #axs[2].set_ylabel('Earnings')
    #axs[2].legend() 
        
    plt.tight_layout()
    plt.show()
