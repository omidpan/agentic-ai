# filename: train_lstm.py
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Set random seed for reproducibility
np.random.seed(2505)
tf.random.set_seed(2505)

def download_and_preprocess_data(ticker='IONQ', period='730d', interval='1h'):
    print(f"Downloading historical data for {ticker}...")
    df = yf.download(ticker, period=period, interval=interval, prepost=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel('Ticker')
        
    df.index = df.index.tz_localize(None)
    df.index = df.index.astype('int64') // 10**9
    
    # Reorder columns to ensure 'Close' is last
    new_order = [col for col in df.columns if col != 'Close'] + ['Close']
    df = df[new_order]
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    
    return df_scaled, scaler

def create_sequences(df, window_size=30):
    X, Y = [], []
    data_vals = df.values
    for i in range(len(data_vals) - window_size):
        X.append(data_vals[i:i + window_size, :-1]) # All features except Close
        Y.append(data_vals[i + window_size, -1])  # Close price target
    return np.array(X), np.array(Y)

def main():
    ticker_symbol = 'IONQ'
    window_size = 30
    
    df_scaled, scaler = download_and_preprocess_data(ticker_symbol)
    
    # Save the scaler for realtime usage
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved to scaler.pkl")
    
    X, Y = create_sequences(df_scaled, window_size=window_size)
    print(f'Dimension of X: {X.shape}, Dimension of Y: {Y.shape}')
    
    # Train/Test Split (90/10)
    threshold = int(0.9 * len(X))
    trainX, trainY = X[:threshold], Y[:threshold]
    testX, testY = X[threshold:], Y[threshold:]
    
    print(f'Training Length: {trainX.shape}, Testing Length: {testX.shape}')
    
    # Build LSTM Model
    model = keras.Sequential([
        layers.LSTM(units=30, activation='tanh', use_bias=True, input_shape=(trainX.shape[1], trainX.shape[2])),
        layers.Dropout(rate=0.2),
        layers.Dense(1)
    ])
    
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    
    history = model.fit(
        trainX, trainY,
        shuffle=False,
        epochs=50,
        batch_size=32,
        validation_split=0.20,
        verbose=1
    )
    
    # Evaluate model on test set
    test_loss = model.evaluate(testX, testY)
    print(f"Test Loss (MSE): {test_loss[0]}")
    
    # Save the model
    model.save('lstm_model.keras')
    print("Model saved to lstm_model.keras")

if __name__ == '__main__':
    main()