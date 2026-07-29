# filename: utils.py
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

from config import TICKER, PERIOD, INTERVAL, MODEL_PATH, SCALER_PATH, RANDOM_SEED
from feature_engineering import add_technical_indicators, prepare_features_and_target

def set_seeds(seed=RANDOM_SEED):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

def download_raw_data(ticker=TICKER, period=PERIOD, interval=INTERVAL) -> pd.DataFrame:
    """Downloads raw market history from Yahoo Finance and formats indices."""
    df = yf.download(ticker, period=period, interval=interval, prepost=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel('Ticker')
    
    # Make timezone naive and convert index to epoch seconds if necessary
    df.index = df.index.tz_localize(None)
    return df

def fetch_and_process_historical_data(ticker=TICKER, period=PERIOD, interval=INTERVAL):
    """Downloads raw data, computes features, scales dataset, and saves scaler."""
    df_raw = download_raw_data(ticker, period, interval)
    df_featured = add_technical_indicators(df_raw)
    df_prepared = prepare_features_and_target(df_featured)
    
    scaler = MinMaxScaler()
    scaled_vals = scaler.fit_transform(df_prepared.values)
    df_scaled = pd.DataFrame(scaled_vals, columns=df_prepared.columns, index=df_prepared.index)
    
    # Save scaler for runtime usage
    joblib.dump(scaler, SCALER_PATH)
    return df_scaled, scaler

def create_sequences(df: pd.DataFrame, window_size: int):
    """Creates rolling windows (X) and future target values (Y)."""
    X, Y = [], []
    data_vals = df.values
    for i in range(len(data_vals) - window_size):
        X.append(data_vals[i:i + window_size, :-1])  # All feature columns except Close
        Y.append(data_vals[i + window_size, -1])     # Target Close price value
    return np.array(X), np.array(Y)

def load_model_and_scaler():
    """Loads the trained Keras model and MinMaxScaler object from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or scaler files are missing. Please run train_lstm.py first.")
    
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler