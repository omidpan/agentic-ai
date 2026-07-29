# filename: predict_realtime.py
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from kafka import KafkaConsumer, KafkaProducer
import yfinance as yf
import time

# Configurations
KAFKA_SERVER = 'localhost:9092'
RAW_DATA_TOPIC = 'ionq_raw_candles'
PREDICTION_TOPIC = 'ionq_predictions'
TICKER = 'IONQ'
WINDOW_SIZE = 30

def load_artifacts():
    print("Loading model and scaler...")
    model = keras.models.load_model('lstm_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

def fetch_initial_window(ticker=TICKER, window_size=WINDOW_SIZE):
    print("Fetching initial rolling window data...")
    df = yf.download(ticker, period="5d", interval="1h", prepost=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel('Ticker')
    df.index = df.index.tz_localize(None)
    
    new_order = [col for col in df.columns if col != 'Close'] + ['Close']
    df = df[new_order]
    return df.tail(window_size)

def main():
    model, scaler = load_artifacts()
    
    # Setup Kafka Producer to push predictions
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_SERVER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    # Setup Kafka Consumer to read completed 1-hour candles
    consumer = KafkaConsumer(
        RAW_DATA_TOPIC,
        bootstrap_servers=[KAFKA_SERVER],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    # Maintain rolling window buffer
    rolling_df = fetch_initial_window()
    print("Real-time prediction engine started. Listening for completed candles...")
    
    for message in consumer:
        candle = message.value
        # Expected candle format: {"Open": ..., "High": ..., "Low": ..., "Volume": ..., "Close": ...}
        new_row = pd.DataFrame([candle], index=[pd.to_datetime(candle.get('Timestamp', time.time()), unit='s')])
        
        # Reorder columns to match training schema
        new_order = [col for col in rolling_df.columns if col != 'Close'] + ['Close']
        new_row = new_row[new_order]
        
        # Append and keep rolling window size limit
        rolling_df = pd.concat([rolling_df, new_row]).iloc[-WINDOW_SIZE:]
        
        if len(rolling_df) < WINDOW_SIZE:
            continue
            
        # Preprocess / Scale window using loaded scaler
        scaled_values = scaler.transform(rolling_df.values)
        
        # Extract features (exclude Close target column which is last)
        features = scaled_values[:, :-1]
        X_input = np.expand_dims(features, axis=0) # Shape: (1, window_size, n_features)
        
        # Make Prediction
        scaled_pred = model.predict(X_input, verbose=0)[0][0]
        
        # Inverse transform prediction to find absolute price change
        # Construct a dummy array to invert scale specifically for the Close column
        dummy_array = np.zeros((1, scaled_values.shape[1]))
        dummy_array[0, -1] = scaled_pred
        predicted_close = scaler.inverse_transform(dummy_array)[0, -1]
        
        current_close = rolling_df['Close'].iloc[-2] # Previous candle close
        predicted_return = (predicted_close - current_close) / current_close
        
        payload = {
            "timestamp": time.time(),
            "ticker": TICKER,
            "current_close": float(current_close),
            "predicted_close": float(predicted_close),
            "predicted_return": float(predicted_return)
        }
        
        producer.send(PREDICTION_TOPIC, value=payload)
        print(f"Published Prediction -> Return: {predicted_return*100:.2f}% | Current: {current_close} | Pred: {predicted_close}")

if __name__ == '__main__':
    main()