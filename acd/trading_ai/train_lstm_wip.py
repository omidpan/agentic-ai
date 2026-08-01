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
from tensorflow.keras import callbacks, layers, Model
import tensorflow as tf

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
def chronological_split(
    df: pd.DataFrame,
    train_frac=TRAIN_FRAC,
    val_frac=VAL_FRAC
):
    if train_frac <= 0 or val_frac <= 0:
        raise ValueError("train_frac and val_frac must be positive.")

    if train_frac + val_frac >= 1:
        raise ValueError(
            "train_frac + val_frac must be less than 1."
        )

    df = (
        df.copy()
        .sort_values(["datetime", "ticker"])
        .reset_index(drop=True)
    )

    unique_datetimes = (
        df["datetime"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )

    n_dates = len(unique_datetimes)

    train_end = int(n_dates * train_frac)
    val_end = int(n_dates * (train_frac + val_frac))

    train_dates = unique_datetimes[:train_end]
    validation_dates = unique_datetimes[train_end:val_end]
    test_dates = unique_datetimes[val_end:]

    train_df = df[df["datetime"].isin(train_dates)].copy()
    validation_df = df[
        df["datetime"].isin(validation_dates)
    ].copy()
    test_df = df[df["datetime"].isin(test_dates)].copy()

    return train_df, validation_df, test_df

def create_sequences(X, targets, sequence_length):
    """
    Create LSTM sequences for either:

    1. A single NumPy target array
    2. A dictionary containing multiple target arrays
    """
    X_sequences = []

    if isinstance(targets, dict):
        target_sequences = {
            target_name: []
            for target_name in targets
        }
    else:
        target_sequences = []

    for end_index in range(
        sequence_length - 1,
        len(X)
    ):
        start_index = end_index - sequence_length + 1

        X_sequences.append(
            X[start_index:end_index + 1]
        )

        if isinstance(targets, dict):
            for target_name, target_values in targets.items():
                target_sequences[target_name].append(
                    target_values[end_index]
                )
        else:
            target_sequences.append(
                targets[end_index]
            )

    X_sequences = np.asarray(
        X_sequences,
        dtype=np.float32
    )

    if isinstance(targets, dict):
        target_sequences = {
            target_name: np.asarray(
                values,
                dtype=np.float32
            )
            for target_name, values in target_sequences.items()
        }
    else:
        target_sequences = np.asarray(
            target_sequences,
            dtype=np.float32
        )

    return X_sequences, target_sequences
###### important for multiple stock sequeces, 
# we need to create sequences for each stock separately and then concatenate them together. 
# This ensures that the sequences are not mixed across different stocks, 
# which could lead to data leakage and incorrect model training.
def create_ticker_sequences(
    split_df,
    transform_function,
    sequence_length
):
    all_X = []
    all_targets = None

    for ticker, ticker_df in split_df.groupby("ticker",sort=False):
        ticker_df = (ticker_df.sort_values("datetime").reset_index(drop=True))
        if len(ticker_df) < sequence_length:
            continue

        X, targets = transform_function(ticker_df)

        X_seq, target_seq = create_sequences(X,targets,sequence_length)
        all_X.append(X_seq)

        if isinstance(target_seq, dict):
            if all_targets is None:
                all_targets = {
                    name: []
                    for name in target_seq
                }

            for name, values in target_seq.items():
                all_targets[name].append(values)

        else:
            if all_targets is None:
                all_targets = []

            all_targets.append(target_seq)

    if not all_X:
        raise ValueError(
            "No sequences were created. Check sequence length "
            "and the number of rows per ticker."
        )

    combined_X = np.concatenate(all_X,axis=0)

    if isinstance(all_targets, dict):
        combined_targets = {
            name: np.concatenate(values, axis=0)
            for name, values in all_targets.items()
        }
    else:
        combined_targets = np.concatenate(
            all_targets,
            axis=0
        )

    return combined_X, combined_targets


def prepare_data(df: pd.DataFrame):


    # 4. Strictly exclude 'datetime' and targets from the feature set
    ignore = {
        "datetime",
        "ticker",
        "stock_Return",
        "stock_LogReturn",
        "stock_Direction",
        "smh_Return",
        "smh_LogReturn",
        "smh_Direction",
        "spy_Return",
        "spy_LogReturn",
        "spy_Direction",
    }

    feature_columns = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]

    print(f"+++ Feature length: {len(feature_columns)} +++")
    # Quick check to ensure datetime is excluded!
    print(f"Sample features (first 5): {feature_columns[:5]}")  
    
    return df, feature_columns
from tensorflow import keras
from tensorflow.keras import layers


def build_classification_model(
    sequence_length,
    number_of_features
):
    inputs = keras.Input(
        shape=(
            sequence_length,
            number_of_features
        )
    )

    x = layers.LSTM(
        64,
        return_sequences=True
    )(inputs)

    x = layers.Dropout(0.2)(x)

    x = layers.LSTM(
        32,
        return_sequences=False
    )(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Dense(
        16,
        activation="relu"
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="stock_Direction"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc")
        ]
    )

    return model
def main():
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    clean_bar_name = str(bar_size).replace(" ", "").lower()
    print(f"Loading data {stock_symbol}...")
    df = pd.read_csv(f"{DATA_DIR}/semiconductor/combined_dataset.csv")

    df_full, feature_columns = prepare_data(df)



    # Split chronologically BEFORE fitting scaler
    train_df, validation_df, test_df = chronological_split(df_full, TRAIN_FRAC, VAL_FRAC)
    print(f"Train/Val/Test rows: {len(train_df)}/{len(validation_df)}/{len(test_df)}")
    print("Train:", train_df["datetime"].min()," to ", train_df["datetime"].max())

    print("Validation:", validation_df["datetime"].min()," to", validation_df["datetime"].max())

    print("Test:", test_df["datetime"].min()," to", test_df["datetime"].max())

    assert train_df["datetime"].max() < validation_df["datetime"].min()
    assert validation_df["datetime"].max() < test_df["datetime"].min()

    # Fit scaler strictly on numeric training features after dropping datetime and target columns
    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns])
   ### IMPORTANAT approuch:using stock_Direction as target for classification
   # and stock_Return as target for regression
   
   # classification target: stock_Direction (binary: 1 for positive return, 0 for negative return)
    def transform_classification(split_df):
        X = scaler.transform(split_df[feature_columns])
        y = (split_df["stock_Direction"].to_numpy().astype("float32"))
        return X, y
####################################################################################
 
 ############# regrassion target: stock_LogReturn (continuous) or stock_Return (continuous)
    def transform_regression(split_df):
        X = scaler.transform(split_df[feature_columns])
        y = split_df["stock_LogReturn"].to_numpy().astype("float32")
        return X, y
    ####################################################################################
    
############# 2 stockDirection and stock_LogReturn as multi-targets
    def transform_multi_target(split_df):
        X = scaler.transform(split_df[feature_columns])
        y_ret = split_df["stock_LogReturn"].to_numpy().astype("float32")
        y_dir = split_df["stock_Direction"].to_numpy().astype("float32")
        target = {"stock_LogReturn": y_ret, "stock_Direction": y_dir}
        return X, target
    ######################################################################
    
    
    #classification target:
    X_train_classify, y_train_classify = transform_classification(train_df)
    X_val_classify, y_val_classify = transform_classification(validation_df)
    X_test_classify, y_test_classify = transform_classification(test_df)
    ### regression target:
    X_train_regression, y_train_regression = transform_regression(train_df)
    X_val_regression, y_val_regression = transform_regression(validation_df)
    X_test_regression, y_test_regression = transform_regression(test_df)
    
    ##### multi-targets
    X_train_multi, y_train_multi = transform_multi_target(train_df)
    X_val_multi, y_val_multi = transform_multi_target(validation_df)
    X_test_multi, y_test_multi = transform_multi_target(test_df)
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


########### classification experiment   ###############################
    X_train_classify_seq, y_train_classify_seq = create_ticker_sequences(
        train_df,
        transform_classification,
        WINDOW_SIZE
    )

    X_validation_classify_seq, y_validation_classify_seq = create_ticker_sequences(
        validation_df,
        transform_classification,
        WINDOW_SIZE
    )

    X_test_classify_seq, y_test_classify_seq = create_ticker_sequences(
        test_df,
        transform_classification,
        WINDOW_SIZE
    )
 ############ regression experiment   ###############################
    X_train_regression_seq, y_train_regression_seq = create_ticker_sequences(
        train_df,
        transform_regression,
        WINDOW_SIZE
    )
    X_validation_regression_seq, y_validation_regression_seq = create_ticker_sequences(
        validation_df,
        transform_regression,
        WINDOW_SIZE
    )
    X_test_regression_seq, y_test_regression_seq = create_ticker_sequences(
        test_df,
        transform_regression,
        WINDOW_SIZE
    )
 ############ multi-target experiment   ###############################
    X_train_multi_seq, y_train_multi_seq = create_ticker_sequences(
        train_df,
        transform_multi_target,
        WINDOW_SIZE
    )
    X_validation_multi_seq, y_validation_multi_seq = create_ticker_sequences(
        validation_df,
        transform_multi_target,
        WINDOW_SIZE
    )
    X_test_multi_seq, y_test_multi_seq = create_ticker_sequences(
        test_df,
        transform_multi_target,
        WINDOW_SIZE
    )   
    ####################################################################   

    # Model Architecture
    model = build_classification_model(
        sequence_length=X_train_classify_seq.shape[1],
        number_of_features=X_train_classify_seq.shape[2]
    )
    model.summary()
    # Callbacks
    early_stop = callbacks.EarlyStopping(
         monitor="val_loss",
        patience=12,
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint = callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_loss", save_best_only=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )

    print("Number of tickers:", train_df["ticker"].nunique())
    print("Training rows:", len(train_df))
    print("Training sequences:", len(X_train_classify_seq))

    # assert X_train_classify_seq.shape == (10064, 20, 50)
    # assert y_train_classify_seq.shape == (10064,)
    # assert len(X_train_classify_seq) == len(y_train_classify_seq)
    # -------------------------------------------------------------
    # Fit Model (Using Tuples/Lists or Clean Dicts)
    # -------------------------------------------------------------
    # Option A (Recommended for Keras 3 multi-output compatibility):
   # 3. Train it
    history = model.fit(
        X_train_classify_seq,
        y_train_classify_seq,
        validation_data=(
            X_validation_classify_seq,
            y_validation_classify_seq
        ),
        epochs=100,
        batch_size=64,
        shuffle=False,
        callbacks=[
            early_stop,
            checkpoint,
            reduce_lr
        ]
    )


    # 4. Evaluate only after training is finished
    test_results = model.evaluate(
        X_test_classify_seq,
        y_test_classify_seq,
        verbose=1
    )

    print("Test results:", test_results)
    positive_rate = np.mean(y_test_classify_seq)
    majority_accuracy = max(
        positive_rate,
        1 - positive_rate
    )

    print("Positive rate:", positive_rate)
    print("Majority baseline accuracy:", majority_accuracy)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()