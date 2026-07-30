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
import matplotlib.pyplot as plt
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
    """
    Directional Accuracy measures how often your model correctly predicts the direction of the next price movement,
    regardless of the size of the move.
    Fraction of predictions whose sign matches the actual return's sign.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def naive_baseline_accuracy(y_true):
    """
    Sanity check: what accuracy would you get by just always predicting the
    sign of the PREVIOUS realized return (a persistence/random-walk model)?
    If your LSTM's directional accuracy isn't meaningfully above this AND
    above 50%, it isn't demonstrating real predictive edge yet.
    """
    y_true = np.asarray(y_true)

    prev_sign = np.sign(y_true[:-1])
    next_sign = np.sign(y_true[1:])

    mask = (prev_sign != 0) & (next_sign != 0)

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(prev_sign[mask] == next_sign[mask]))
def chronological_split(df: pd.DataFrame, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """
    Splits a time-ordered dataframe into train/val/test by position, never
    shuffling. Fit any scaler ONLY on the train slice this returns -- fitting
    on the full dataset first (as the original code did) leaks test-period
    statistics into training.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
def create_sequences(feature_df: pd.DataFrame, target_series: pd.Series, window_size: int):
    """
    Builds rolling windows X (window_size x n_features) and matching targets Y.
    X[i] uses rows [i : i+window_size) of features; Y[i] is the target value
    aligned to the LAST row in that window (row i+window_size-1) -- i.e. the
    return already computed from that candle to `horizon` candles later.
    No future rows are read past that point.
    """
    X, Y = [], []
    feat_vals = feature_df.values
    targ_vals = target_series.values
    for i in range(len(feat_vals) - window_size + 1):
        X.append(feat_vals[i:i + window_size, :])
        Y.append(targ_vals[i + window_size - 1])
    return np.array(X), np.array(Y)
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
    df['Target'] = (df['Target_Return'] > 0).astype(int)
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
    feature_columns = [
        c for c in df.columns
        if c not in ['Target_Return', 'Target']]
    
    return df, feature_columns, raw_close


def main():
    os.makedirs(os.path.dirname(MODEL_PATH) or '.', exist_ok=True)
    clean_bar_name = str(bar_size).replace(" ", "").lower()
    print(f"loading data {TICKER}...")
    df = pd.read_csv(f"{DATA_DIR}/{stock_symbol}_{clean_bar_name}_features.csv")
    df_full, feature_columns, _ = prepare_features_and_target(df, horizon=HORIZON)

# --------------------------------------------------------
# ADD THIS HERE
# --------------------------------------------------------
    print("\n========== TARGET RETURN STATISTICS ==========")
    print(df_full["Target_Return"].describe())

    print(f"Positive returns : {(df_full['Target_Return'] > 0).mean() * 100:.2f}%")
    print(f"Negative returns : {(df_full['Target_Return'] < 0).mean() * 100:.2f}%")
    print(f"Zero returns     : {(df_full['Target_Return'] == 0).mean() * 100:.2f}%")

    print(f"Mean abs return  : {np.mean(np.abs(df_full['Target_Return'])):.6f}")
    print("=============================================\n")
# --------------------------------------------------------

    # --- Split chronologically BEFORE fitting the scaler (avoids leakage) ---
    train_df, val_df, test_df = chronological_split(df_full, TRAIN_FRAC, VAL_FRAC)
    print(f"Train/Val/Test rows: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_columns].values)  # fit on TRAIN ONLY

    def transform(split_df):
        scaled_feats = scaler.transform(split_df[feature_columns].values)
        return scaled_feats, split_df['Target'].values

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
        layers.Dropout(rate=0.05),
        layers.LSTM(units=32, activation='tanh'),
        layers.Dropout(rate=0.05),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc')
        ]
    )
    model.summary()

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
)
    model.fit(
        trainX, trainY,
        validation_data=(valX, valY),
        shuffle=False,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1
    )

    test_loss, test_mae = model.evaluate(testX, testY, verbose=0)
    probability = model.predict(testX).flatten()
    test_pred = (probability >= 0.5).astype(int)    
    print("\n========== PREDICTION STATISTICS ==========")
    print(f"Actual mean      : {np.mean(testY):.6f}")
    print(f"Predicted mean   : {np.mean(test_pred):.6f}")

    print(f"Actual std       : {np.std(testY):.6f}")
    print(f"Predicted std    : {np.std(test_pred):.6f}")

    print(f"Actual min/max   : {testY.min():.6f} / {testY.max():.6f}")
    print(f"Pred min/max     : {test_pred.min():.6f} / {test_pred.max():.6f}")

    corr = np.corrcoef(testY, test_pred)[0, 1]
    print(f"Correlation      : {corr:.4f}")
    print("==========================================\n")
# -------------------------------------------------------
# Plot predictions vs actual returns
# -------------------------------------------------------

    plt.figure(figsize=(15,5))
    plt.plot(testY[:300], label="Actual Return")
    plt.plot(test_pred[:300], label="Predicted Return")
    plt.title("Actual vs Predicted Returns (First 300 Test Samples)")
    plt.xlabel("Test Sample")
    plt.ylabel("Return")
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------------------------------------------------------
    # Metrics
    # -------------------------------------------------------
    accuracy = np.mean(test_pred == testY)
    baseline_acc = naive_baseline_accuracy(testY)
    print(f"Classification Accuracy : {accuracy*100:.2f}%")
    

    print(f"Test MSE: {test_loss:.6f} | Test MAE (return): {test_mae:.6f}")
    print(f"Naive persistence baseline directional accuracy: {baseline_acc*100:.2f}%")    
    print("If the model's directional accuracy isn't clearly above both 50% "
          "and the naive baseline, treat any backtest P&L with suspicion --"
          "it likely isn't capturing real predictive signal yet.")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == '__main__':
    main()
