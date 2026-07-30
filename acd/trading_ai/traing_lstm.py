# filename: train_lstm.py
'''
how to run the file python  traing_lstm.py -s nvda  -bs "1 hour"
argument -s is the stock symbol, argument -bs is the bar size of historical data.
This script trains an LSTM model for stock price 
prediction using historical data and technical indicators.

'''

import os
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler
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
from config import (
    TICKER, PERIOD, INTERVAL, WINDOW_SIZE, HORIZON,
    TRAIN_FRAC, VAL_FRAC, MODEL_PATH, SCALER_PATH, FEATURE_META_PATH, RANDOM_SEED,
    DATA_DIR,MODEL_PATH, SCALER_PATH, FEATURE_META_PATH, RANDOM_SEED,
)
from feature_engineering import add_technical_indicators, prepare_features_and_target
from utils.utils import set_seeds, download_raw_data, chronological_split, create_sequences

set_seeds(RANDOM_SEED)


def directional_accuracy(y_true, y_pred):
    """Fraction of predictions whose sign matches the actual return's sign."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def naive_baseline_accuracy(y_true):
    """
    Sanity check: what accuracy would you get by just always predicting the
    sign of the PREVIOUS realized return (a persistence/random-walk model)?
    If your LSTM's directional accuracy isn't meaningfully above this AND
    above 50%, it isn't demonstrating real predictive edge yet.
    """
    prev_sign = np.sign(y_true[:-1])
    actual_sign = np.sign(y_true[1:])
    return float(np.mean(prev_sign == actual_sign))



def main():
    os.makedirs(os.path.dirname(MODEL_PATH) or '.', exist_ok=True)

    print(f"loading data {TICKER}...")
    df = pd.read_csv(f"{DATA_DIR}/{stock_symbol}_{bar_size}_features.csv")
    df_full, feature_columns, _ = prepare_features_and_target(df, horizon=HORIZON)

    # --- Split chronologically BEFORE fitting the scaler (avoids leakage) ---
    train_df, val_df, test_df = chronological_split(df_full, TRAIN_FRAC, VAL_FRAC)
    print(f"Train/Val/Test rows: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_columns].values)  # fit on TRAIN ONLY

    def transform(split_df):
        scaled_feats = scaler.transform(split_df[feature_columns].values)
        return scaled_feats, split_df['Target_Return'].values

    train_X_scaled, train_Y = transform(train_df)
    val_X_scaled, val_Y = transform(val_df)
    test_X_scaled, test_Y = transform(test_df)

    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_META_PATH, 'w') as f:
        json.dump({
            'feature_columns': feature_columns,
            'window_size': WINDOW_SIZE,
            'horizon': HORIZON
        }, f, indent=2)
    print(f"Scaler saved to {SCALER_PATH}, feature meta saved to {FEATURE_META_PATH}")

    trainX, trainY = create_sequences(pd.DataFrame(train_X_scaled), pd.Series(train_Y), WINDOW_SIZE)
    valX, valY = create_sequences(pd.DataFrame(val_X_scaled), pd.Series(val_Y), WINDOW_SIZE)
    testX, testY = create_sequences(pd.DataFrame(test_X_scaled), pd.Series(test_Y), WINDOW_SIZE)

    print(f"Train seq: {trainX.shape}, Val seq: {valX.shape}, Test seq: {testX.shape}")

    model = keras.Sequential([
        layers.LSTM(units=64, activation='tanh', return_sequences=True,
                    input_shape=(trainX.shape[1], trainX.shape[2])),
        layers.Dropout(rate=0.2),
        layers.LSTM(units=32, activation='tanh'),
        layers.Dropout(rate=0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.summary()

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)

    model.fit(
        trainX, trainY,
        validation_data=(valX, valY),
        shuffle=False,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    test_loss, test_mae = model.evaluate(testX, testY, verbose=0)
    test_pred = model.predict(testX, verbose=0).flatten()
    dir_acc = directional_accuracy(testY, test_pred)
    baseline_acc = naive_baseline_accuracy(testY)

    print(f"Test MSE: {test_loss:.6f} | Test MAE (return): {test_mae:.6f}")
    print(f"Directional accuracy on test set: {dir_acc*100:.2f}%")
    print(f"Naive persistence baseline directional accuracy: {baseline_acc*100:.2f}%")
    print("If the model's directional accuracy isn't clearly above both 50% "
          "and the naive baseline, treat any backtest P&L with suspicion --"
          "it likely isn't capturing real predictive signal yet.")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == '__main__':
    main()
