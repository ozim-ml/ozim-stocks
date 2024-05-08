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

# mlflow.set_experiment("stocks-lstm")

def perform_lstm(df, ticker):




    
    # Create a BytesIO object to store the plot
    plot_bytes_lstm = BytesIO()

    # Save the plot to BytesIO and encode as base64
    plot_bytes_lstm.seek(0)
    plt.savefig(plot_bytes_lstm, format='png')
    plot_base64_lstm = base64.b64encode(plot_bytes_lstm.getvalue()).decode('utf-8')

    plt.close()

    return plot_base64_lstm




