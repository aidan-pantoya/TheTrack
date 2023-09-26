import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime

def get_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    return df['Close']

def create_features(data, window_size):
    data_lag = pd.DataFrame(data)
    for i in range(window_size):
        data_lag[f"lag_{i+1}"] = data_lag['Close'].shift(i+1)
    data_lag.dropna(inplace=True)
    return data_lag.drop(columns=['Close']), data_lag['Close']

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Model R^2 score: {score}")
    return model

def predict_next_day(model, latest_data):
    prediction = model.predict([latest_data])
    return prediction[0]

if __name__ == '__main__':
    symbol = 'SPY'  # stock symbol
    start_date = '2002-01-01'
    end_date = '2023-08-27'
    window_size = 50  # use the previous X days to predict the next day

    data = get_data(symbol, start_date, end_date)
    X, y = create_features(data, window_size)
    
    model = train_model(X, y)
    
    # Predict the next day's closing price based on the latest available data
    latest_data = X.iloc[-1].values
    prediction = predict_next_day(model, latest_data)
    print(f"Predicted next day closing price for {symbol}: ${prediction:.2f}")

