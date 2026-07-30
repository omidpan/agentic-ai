import pandas as pd
import pandas_ta as ta
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
stock_symbol = args.symbol.lower()
bar_size = args.bar_size

# 1. Load your dataset
print(f"Loading data for {DATA_DIR}/{stock_symbol}_{bar_size}_init.csv")
df = pd.read_csv(f"{DATA_DIR}/{stock_symbol}_{bar_size}_init.csv")
df["Date"] = pd.to_datetime(df["datetime"], unit='s')
df.set_index("Date", inplace=True)

# 2. Ensure vital columns are lowercase for pandas-ta
# ATR requires high, low, and close columns to be lowercase
df.rename(columns={
    "close": "close",
    "high": "high",
    "low": "low",
    "open": "open"
}, inplace=True)

# 3. Calculate Existing Indicators
# macd = df.ta.macd(fast=12, slow=26, signal=9)# original code
macd = df.ta.macd(fast=20, slow=30, signal=10,ma_type='ema')  # Adjusted parameters for MACD
bbands_all = df.ta.bbands(length=36, std=2)  # Adjusted parameters for Bollinger Bands
# Select only the lower, basis (middle), and upper bands
bbands = bbands_all[['BBL_36_2.0', 'BBM_36_2.0', 'BBU_36_2.0']]

# ATR (Average True Range) - Measures market volatility
atr = df.ta.atr(length=21)  # Adjusted length for ATR

# RSI (Relative Strength Index) - Measures momentum (overbought/oversold)
rsi = df.ta.rsi(length=24)

# EMA (Exponential Moving Average) - Faster reaction to recent price changes
ema = df.ta.ema(length=20)

# WMA (Weighted Moving Average) - Puts more weight on recent data points
wma = df.ta.wma(length=20)

# 5. Combine everything into your original dataset
# We pass all indicators as a list to concat them in one single step
df = pd.concat([df, macd, bbands, atr, rsi, ema, wma], axis=1)
# --- DETECT CROSSOVERS ---

# 1. MACD Crossovers
# Bullish Cross: MACD line crosses ABOVE the Signal line
df['macd_bullish_cross'] = (df['MACD_20_30_10'] > df['MACDs_20_30_10']) & (df['MACD_20_30_10'].shift(1) <= df['MACDs_20_30_10'].shift(1))

# Bearish Cross: MACD line crosses BELOW the Signal line
df['macd_bearish_cross'] = (df['MACD_20_30_10'] < df['MACDs_20_30_10']) & (df['MACD_20_30_10'].shift(1) >= df['MACDs_20_30_10'].shift(1))


# 2. Bollinger Band Crosses
# Price breaks ABOVE Upper Band (Potential overbought / momentum breakout)
df['bb_upper_breakout'] = (df['close'] > df['BBU_36_2.0']) & (df['close'].shift(1) <= df['BBU_36_2.0'].shift(1))

# Price breaks BELOW Lower Band (Potential oversold / mean-reversion buy)
df['bb_lower_breakout'] = (df['close'] < df['BBL_36_2.0']) & (df['close'].shift(1) >= df['BBL_36_2.0'].shift(1))


# --- EXTRACT THE CROSSOVER ROWS ---

# Filter rows where ANY of these conditions are True
crossover_events = df[
    df['macd_bullish_cross'] | 
    df['macd_bearish_cross'] | 
    df['bb_upper_breakout'] | 
    df['bb_lower_breakout']
]

# Print out your filtered event table
print("\n--- Detected Crossover Events ---")
print(crossover_events[['close', 'MACD_20_30_10', 'MACDs_20_30_10', 'BBL_36_2.0', 'BBU_36_2.0']].tail(10))

# 6. View your updated dataset with indicators
print(df.head())
