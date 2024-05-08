from app.main import *

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, InputLayer
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from datetime import datetime

# mlflow.set_experiment("stocks-lstm")

def perform_lstm(stock_df, ticker):

    features = stock_df[['High', 'Low', 'Volume']].values
    target = stock_df['Close'].values.reshape(-1, 1)
    
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler_features.fit_transform(features)
    target_scaled = scaler_target.fit_transform(target)

    def create_lstm_data(features, target, time_steps=1):
        X, Y = [], []
        for i in range(len(target) - time_steps):
            X.append(features[i:(i + time_steps)])
            Y.append(target[i + time_steps])
        return np.array(X), np.array(Y)

    time_steps = 30
    X, Y = create_lstm_data(features_scaled, target_scaled, time_steps)

    model = Sequential([
        InputLayer(shape=(X.shape[1], X.shape[2])),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    
    model.summary()
    early_stop = EarlyStopping(monitor = 'loss', patience = 2)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, epochs=50, batch_size=32, callbacks=[early_stop])

    last_inputs = features_scaled[-time_steps:]
    future_data = []
    
    forecast_steps = 30
    
    for _ in range(forecast_steps):  
        last_input = np.array([last_inputs])
        next_price_scaled = model.predict(last_input)
        future_data.append(next_price_scaled[0][0])
        new_row = [[last_inputs[-1, 0], last_inputs[-1, 1], next_price_scaled[0][0]]]
        last_inputs = np.vstack([last_inputs, new_row])[1:]

    future_data = scaler_target.inverse_transform(np.array(future_data).reshape(-1, 1))


    historical_data = stock_df[['Close']].tail(60)

    future_indices = range(1, 1 + len(future_data))  
    future_df = pd.DataFrame(future_data, columns=['Close'], index=future_indices)

    formatted_historical_labels = [datetime.strptime(label, '%Y-%m-%d %H:%M:%S%z').strftime('%Y-%m-%d %H:%M') if '+' in label 
                                   else label for label in historical_data.index.astype(str)]

    plt.figure(figsize=(14, 7))
    plt.plot(formatted_historical_labels, historical_data['Close'], marker='o', label='Historical Data')
    plt.plot(future_df.index.astype(str), future_df['Close'], marker='o', color='orange', label='Forecasted Data')

    plt.title('Historical and Predicted Stock Prices')
    plt.xlabel('Data Points')
    plt.ylabel('Stock Price')
    plt.grid(True)
    plt.legend()

    all_labels = formatted_historical_labels + list(future_df.index.astype(str))
    label_step = 5
    visible_labels = all_labels[::label_step]  
    plt.xticks(range(len(all_labels))[::label_step], visible_labels, rotation=45) 
    plt.tight_layout()   
    
    plot_bytes_lstm = BytesIO()
    plot_bytes_lstm.seek(0)
    plt.savefig(plot_bytes_lstm, format='png')
    plot_base64_lstm = base64.b64encode(plot_bytes_lstm.getvalue()).decode('utf-8')
    
    plt.close()

    return plot_base64_lstm
