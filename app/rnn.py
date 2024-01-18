from app.main import *
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def perform_lstm(df, ticker):

    tf.random.set_seed(1)

    df_adj = df.filter(['Adj Close'])
    lstm_df = df_adj.values

    train_len_val = .80
    train_len = int(np.ceil(len(lstm_df) * train_len_val))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(lstm_df)

    train_data = scaled_data[0:int(train_len), :]

    x_train, y_train = [],[]

    interval = 60

    for i in range(interval, len(train_data)):
        x_train.append(train_data[i - interval:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Dropout(0.2))

    model.compile(optimizer='adam', loss='mean_squared_error')

    batch_size = 16
    epochs = 10

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    test_data = scaled_data[train_len - interval:, :]
    x_test = []
    y_test = lstm_df[train_len:, :]

    for i in range(interval, len(test_data)):
        x_test.append(test_data[i - interval:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    mean = np.mean(y_test)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    cv_rmse = rmse/mean

    train = df_adj[:train_len]
    valid = df_adj[train_len:]
    valid['Predictions'] = predictions

    plt.figure(figsize=(12, 6))
    plt.title(f'Model of {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Adj Close Price')
    plt.plot(train['Adj Close'])
    plt.plot(valid[['Adj Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

    # Create a BytesIO object to store the plot
    plot_bytes_lstm = BytesIO()

    # Save the plot to BytesIO and encode as base64
    plot_bytes_lstm.seek(0)
    plt.savefig(plot_bytes_lstm, format='png')
    plot_base64_lstm = base64.b64encode(plot_bytes_lstm.getvalue()).decode('utf-8')

    plt.close()

    return plot_base64_lstm




