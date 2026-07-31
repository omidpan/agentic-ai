import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import callbacks, layers

from config import (
    DATA_DIR,
    FEATURE_META_PATH,
    HORIZON,
    MODEL_PATH,
    RANDOM_SEED,
    SCALER_PATH,
    TRAIN_FRAC,
    VAL_FRAC,
    WINDOW_SIZE,
)
from utils.utils import set_seeds

parser = argparse.ArgumentParser(description="Process a stock symbol.")
parser.add_argument(
    "-s", "--symbol", type=str, required=False, default="NVDA", help="The stock symbol."
)
parser.add_argument(
    "-bs", "--bar_size", type=str, required=False, default="1 hour", help="Candle size."
)
args = parser.parse_args()
stock_symbol = args.symbol.lower()
bar_size = args.bar_size

set_seeds(RANDOM_SEED)


def chronological_split(df: pd.DataFrame, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def create_multi_target_sequences(
    feature_df: pd.DataFrame,
    target_return: pd.Series,
    target_direction: pd.Series,
    window_size: int,
):
    """Builds rolling sequences for multi-output LSTM (return & direction)."""
    X, Y_ret, Y_dir = [], [], []
    feat_vals = feature_df.values
    ret_vals = target_return.values
    dir_vals = target_direction.values

    for i in range(len(feat_vals) - window_size + 1):
        X.append(feat_vals[i : i + window_size, :])
        Y_ret.append(ret_vals[i + window_size - 1])
        Y_dir.append(dir_vals[i + window_size - 1])

    return np.array(X), np.array(Y_ret), np.array(Y_dir)

def prepare_data(df: pd.DataFrame):
    # 1. Ensure data is strictly sorted chronologically by epoch datetime
    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)

    # 2. Convert categorical text columns (e.g., session/trading period) to one-hot dummies
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=float)

    # 3. OPTIONAL: Extract temporal features from epoch if useful
    # (Assuming datetime is in seconds; use unit='ms' if in milliseconds)
    dt_series = pd.to_datetime(df['datetime'], unit='s')
    df['hour_sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24.0)

    # 4. Strictly exclude 'datetime' and targets from the feature set
    ignore = {
        "datetime",
        "Target_Return",
        "Target_LogReturn",
        "Target_Direction",
        "Target",
    }

    feature_columns = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]

    print(f"+++ Feature length: {len(feature_columns)} +++")
    # Quick check to ensure datetime is excluded!
    print(f"Sample features (first 5): {feature_columns[:5]}")  
    
    return df, feature_columns

def main():
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    clean_bar_name = str(bar_size).replace(" ", "").lower()
    print(f"Loading data {stock_symbol}...")
    df = pd.read_csv(f"{DATA_DIR}/{stock_symbol}_{clean_bar_name}_features.csv")

    df_full, feature_columns = prepare_data(df)

    # Automatically derive Target_Direction if not present
    if "Target_Direction" not in df_full.columns and "Target_Return" in df_full.columns:
        df_full["Target_Direction"] = (df_full["Target_Return"] > 0).astype(int)

    # Split chronologically BEFORE fitting scaler
    train_df, val_df, test_df = chronological_split(df_full, TRAIN_FRAC, VAL_FRAC)
    print(f"Train/Val/Test rows: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    # Fit scaler strictly on numeric training features
    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns].values)

    def transform(split_df):
        scaled_feats = scaler.transform(split_df[feature_columns].values)
        return (
            scaled_feats,
            split_df["Target_Return"].values,
            split_df["Target_Direction"].values,
        )

    train_X_s, train_Y_ret, train_Y_dir = transform(train_df)
    val_X_s, val_Y_ret, val_Y_dir = transform(val_df)
    test_X_s, test_Y_ret, test_Y_dir = transform(test_df)

    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_META_PATH, "w") as f:
        json.dump(
            {
                "feature_columns": feature_columns,
                "window_size": WINDOW_SIZE,
                "horizon": HORIZON,
            },
            f,
            indent=2,
        )

    # Create sequences
    trainX, trainY_ret, trainY_dir = create_multi_target_sequences(
        pd.DataFrame(train_X_s),
        pd.Series(train_Y_ret),
        pd.Series(train_Y_dir),
        WINDOW_SIZE,
    )
    valX, valY_ret, valY_dir = create_multi_target_sequences(
        pd.DataFrame(val_X_s), pd.Series(val_Y_ret), pd.Series(val_Y_dir), WINDOW_SIZE
    )
    testX, testY_ret, testY_dir = create_multi_target_sequences(
        pd.DataFrame(test_X_s), pd.Series(test_Y_ret), pd.Series(test_Y_dir), WINDOW_SIZE
    )

    # Model Architecture
    inputs = keras.Input(shape=(WINDOW_SIZE, len(feature_columns)))
    x = layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
    x = layers.LSTM(32, dropout=0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    reg = layers.Dense(16, activation="relu")(x)
    reg_output = layers.Dense(1, activation="linear", name="return")(reg)

    cls = layers.Dense(16, activation="relu")(x)
    cls_output = layers.Dense(1, activation="sigmoid", name="direction")(cls)

    model = keras.Model(inputs, [reg_output, cls_output])

    metrics = {
        "return": [keras.metrics.MeanAbsoluteError(), keras.metrics.RootMeanSquaredError()],
        "direction": ["accuracy", keras.metrics.AUC()],
    }

 
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    checkpoint = callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_loss", save_best_only=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-4, verbose=1
    )
# -------------------------------------------------------------
    # Model Compilation & Loss Setup
    # -------------------------------------------------------------
    # Explicitly name your output layers in the functional model:
    # reg_output = layers.Dense(1, activation="linear", name="return")(reg)
    # cls_output = layers.Dense(1, activation="sigmoid", name="direction")(cls)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss={"return": "huber", "direction": "binary_crossentropy"},
        loss_weights={"return": 1.0, "direction": 0.5},
        metrics={
            "return": [
                keras.metrics.MeanAbsoluteError(),
                keras.metrics.RootMeanSquaredError(),
            ],
            "direction": ["accuracy", keras.metrics.AUC()],
        },
    )
    model.summary()
    # -------------------------------------------------------------
    # Compute Class/Sample Weights for Direction Head
    # -------------------------------------------------------------
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(trainY_dir), y=trainY_dir
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Convert y values into sample weights for direction output
    dir_sample_weights = np.array(
        [class_weight_dict[int(y)] for y in trainY_dir], dtype=np.float32
    )

    # -------------------------------------------------------------
    # Fit Model (Using Tuples/Lists or Clean Dicts)
    # -------------------------------------------------------------
    # Option A (Recommended for Keras 3 multi-output compatibility):
    model.fit(
        trainX,
        [trainY_ret, trainY_dir],
        validation_data=(valX, [valY_ret, valY_dir]),
        sample_weight=[
            np.ones_like(trainY_ret, dtype=np.float32),  # Weight 1.0 for returns
            dir_sample_weights,  # Balanced weights for direction
        ],
        shuffle=False,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1,
    )
    # Predictions
    pred_return, probability = model.predict(testX)
    test_pred_dir = (probability >= 0.5).astype(int).flatten()

    # Signal Check on Latest Candle
    latest_prob = probability[-1][0]
    latest_ret = pred_return[-1][0]

    if latest_prob > 0.65 and latest_ret > 0:
        print("Signal: BUY")
    elif latest_prob < 0.35 and latest_ret < 0:
        print("Signal: SELL")
    else:
        print("Signal: No Trade")

    accuracy = np.mean(test_pred_dir == testY_dir)
    print(f"\nClassification Accuracy: {accuracy * 100:.2f}%")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()