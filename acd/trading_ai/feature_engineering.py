# filename: feature_engineering.py
import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends technical indicators (e.g., Moving Averages, RSI, Volatility) 
    to the dataframe to enrich feature dimensionality for the LSTM model.
    """
    df = df.copy()
    
    # Ensure columns are standard
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # Simple Moving Averages
    df['SMA_5'] = close.rolling(window=5).mean()
    df['SMA_20'] = close.rolling(window=20).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    
    # Relative Strength Index (RSI) 14-period
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Volatility (Rolling Standard Deviation of Returns)
    returns = close.pct_change()
    df['Volatility_10'] = returns.rolling(window=10).std()
    
    # Price Rate of Change (ROC)
    df['ROC_5'] = close.pct_change(periods=5)
    
    # Drop rows containing NaN values created by rolling windows/indicators
    df.dropna(inplace=True)
    
    return df

def prepare_features_and_target(df: pd.DataFrame):
    """
    Reorders columns so 'Close' is positioned as the final target variable,
    which matches the pipeline sequence convention.
    """
    if 'Close' in df.columns:
        other_cols = [col for col in df.columns if col != 'Close']
        df = df[other_cols + ['Close']]
    return df