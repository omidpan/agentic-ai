# filename: config.py
import os

# Kafka Configuration
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
RAW_DATA_TOPIC = "ionq_raw_candles"
PREDICTION_TOPIC = "ionq_predictions"

# Ticker & Modeling Settings
TICKER = "IONQ"
PERIOD = "730d"
INTERVAL = "1h"
WINDOW_SIZE = 30
RANDOM_SEED = 2505

# File Paths
MODEL_PATH = "lstm_model.keras"
SCALER_PATH = "scaler.pkl"

# Strategy Parameters
CONFIDENCE_THRESHOLD = 0.005      # 0.5% minimum expected return to trigger trade
NO_TRADE_ZONE_LOWER = -0.002
NO_TRADE_ZONE_UPPER = 0.002
STOP_LOSS_PCT = 0.015             # 1.5% Stop Loss
TAKE_PROFIT_PCT = 0.03            # 3.0% Take Profit
BASE_POSITION_SIZE = 100          # Base units to trade