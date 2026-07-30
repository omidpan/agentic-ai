# filename: feature_engineering.py
import pandas as pd
import numpy as np
import argparse
from config import DATA_DIR
parser = argparse.ArgumentParser(description="Process a stock symbol.")
parser.add_argument("-s" ,"--symbol",
                    type=str,
                    required=True,
                    help="The stock symbol to process. default is NVDA.", 
                    nargs='?', default="NVDA")
parser.add_argument("-bs", "--bar_size",
                    type=str,
                    required=True,
                    help="candle size of historical data. default is 1 hour.", 
                    nargs='?', default="1 hour")
args = parser.parse_args()

# Access the value using dot notation
stock_symbol = args.symbol.lower()
bar_size = args.bar_size
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends technical indicators to enrich feature dimensionality.
    Every indicator here is computed using only past/current rows (rolling
    windows), so no future information leaks into a given row.
    """
    df = df.copy()

    close = df['close']
    high = df['high']
    low = df['low']

    df['SMA_5'] = close.rolling(window=5).mean()
    df['SMA_20'] = close.rolling(window=20).mean()
    df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    returns = close.pct_change()
    df['Volatility_10'] = returns.rolling(window=10).std()
    df['ROC_5'] = close.pct_change(periods=5)

    # Bollinger band width (normalized) - relative volatility measure
    sma20_std = close.rolling(window=20).std()
    df['BB_Width'] = (4 * sma20_std) / (df['SMA_20'] + 1e-8)

    # Average True Range (volatility)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()

    # longest lookback used above is 26 (EMA_26) / 20 (rolling std) -> keep
    # INDICATOR_LOOKBACK in predict_realtime.py >= this so live buffers match
    df.dropna(inplace=True)
    return df


def create_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Adds a forward-looking percentage-return target column: the return from
    the current row's close to the close `horizon` rows ahead.

    This replaces predicting the raw close price level. Raw price is
    non-stationary and an LSTM trained on it mostly learns to echo the last
    known price (looks accurate on MSE, carries little real signal). The
    forward return is stationary and is also what the trading strategy
    actually consumes.
    """
    df = df.copy()
    df['Target_Return'] = df['close'].shift(-horizon) / df['close'] - 1.0
    df.dropna(inplace=True)  # drops the last `horizon` rows (no future close yet)
    return df


def prepare_features_and_target(df: pd.DataFrame, horizon: int = 1):
    """
    Returns (df_full, feature_columns, raw_close).
    - df_full contains all feature columns plus 'Target_Return'.
    - feature_columns is every column except the target (this list gets
      saved at training time and reused at inference time, so live feature
      order/composition can never silently drift from what the model saw
      during training).
    - raw_close is the unscaled close series, aligned to df_full's index,
      for use in live P&L / stop-loss calcs.
    """
    df = create_target(df, horizon=horizon)
    raw_close = df['close'].copy()
    feature_columns = [c for c in df.columns if c != 'Target_Return']
    return df, feature_columns, raw_close

if __name__ == "__main__":
    # read data from CSV, add indicators, and save to new CSV for training
    df = pd.read_csv(f"{DATA_DIR}/{stock_symbol}_{bar_size}_init.csv")
    df = add_technical_indicators(df)
    df, feature_columns, raw_close = prepare_features_and_target(df, horizon=1)
    df.to_csv(f"{DATA_DIR}/{stock_symbol}_{bar_size}_features.csv", index=False)
    
    
