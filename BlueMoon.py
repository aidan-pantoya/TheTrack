import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

ticker = 'SPY'


# date = '2023-10-05' # datetime.now()

# end_date = datetime.strptime(date, '%Y-%m-%d')

date =  datetime.now()

end_date = date

start_date = end_date - timedelta(days=365)
start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')

data = yf.download(ticker, start=start_date, end=end_date)

print("Before preprocessing:")
print(data.head())

epcs = 500

data['PctChange'] = data['Close'].pct_change()

data = data[['Close', 'Volume', 'PctChange']]
data.fillna(0, inplace=True)  

print("After preprocessing:")
print(data.head())


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close', 'Volume', 'PctChange']])

if np.isnan(np.sum(scaled_data)):
    print("Found NaN values in scaled_data. Check data preprocessing.")
else:
    print("No NaN values found in scaled_data. Proceeding with model training.")

training_data_len = len(scaled_data) - 30

x_train, y_train = [], []
x_test, y_test = [], []

for i in range(30, training_data_len):
    x_train.append(scaled_data[i-30:i])
    y_train.append(scaled_data[i, -1]) 
    
x_train, y_train = np.array(x_train), np.array(y_train)

for i in range(training_data_len, len(scaled_data)):
    x_test.append(scaled_data[i-30:i])
    y_test.append(scaled_data[i, -1])  
    
x_test, y_test = np.array(x_test), np.array(y_test)

model = models.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.LSTM(100, return_sequences=True),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.LSTM(150, return_sequences=True),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.LSTM(100),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(1)  
])

opt = optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='mean_squared_error')

model.summary()

if np.isnan(np.sum(x_train)) or np.isnan(np.sum(y_train)) or np.isnan(np.sum(x_test)) or np.isnan(np.sum(y_test)):
    print("Found NaN values in the data. Check data preprocessing.")
else:
    
    history = model.fit(x_train, y_train, batch_size=8, epochs=epcs, validation_data=(x_test, y_test), verbose=1)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 2)), predictions), axis=1))[:, -1]

    y_test_actual = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 2)), y_test.reshape(-1, 1)), axis=1))[:, -1]
    rmse = np.sqrt(np.mean((predictions - y_test_actual) ** 2))
    print('Root mean squared error:', rmse)

    validation = data[['Close', 'PctChange']][training_data_len:].copy()
    validation.reset_index(drop=True, inplace=True)
    validation['PredictedPctChange'] = predictions
    validation['PredictedClose'] = validation['Close'].shift(1) * (1 + validation['PredictedPctChange'])

    future_steps = 30
    last_30_days = scaled_data[-30:]
    current_batch = last_30_days.reshape((1, last_30_days.shape[0], last_30_days.shape[1]))

    future_pct_changes = []

    for i in range(future_steps):
        future_pct_change = model.predict(current_batch)[0, 0]
        future_pct_changes.append(future_pct_change)
        future_pred = np.array([[0, 0, future_pct_change]])
        current_batch = np.append(current_batch[:, 1:, :], future_pred.reshape((1, 1, 3)), axis=1)

    future_pct_changes = scaler.inverse_transform(np.concatenate((np.zeros((future_steps, 2)), np.array(future_pct_changes).reshape(-1, 1)), axis=1))[:, -1]

    last_close = validation['Close'].iloc[-1]
    future_closes = [last_close * (1 + pct_change) for pct_change in future_pct_changes]
    future_close_df = pd.DataFrame(future_closes, columns=['PredictedClose'])

    future_close_df.index = range(len(validation), len(validation) + future_steps)
    validation_extended = pd.concat([validation, future_close_df])

    fig, axs = plt.subplots(2, figsize=(14, 14))

    axs[0].plot(validation['PctChange'], label='Actual PctChange', color='blue')
    axs[0].plot(validation.index, validation['PredictedPctChange'], label='Predicted PctChange', color='red')
    axs[0].plot(range(len(validation), len(validation) + future_steps), future_pct_changes, label='Future Predicted PctChange', color='green')
    axs[0].set_title(f'Actual and Predicted Percentage Changes: {ticker} - {end_date}')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Percentage Change')
    axs[0].legend()

    axs[1].plot(validation['Close'], label='Actual Close Prices', color='blue')
    axs[1].plot(validation_extended.index, validation_extended['PredictedClose'], label='Predicted Close Prices', color='red')
    axs[1].set_title(f'Actual and Predicted Stock Prices: {ticker} - {end_date}')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Price')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
