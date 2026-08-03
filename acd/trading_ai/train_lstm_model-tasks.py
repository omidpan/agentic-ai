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
from utils.utils import set_seeds

## configs
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



parser = argparse.ArgumentParser(description="Process a stock symbol.")
parser.add_argument(
    "-s", "--symbol", type=str, required=False, default="NVDA", help="The stock symbol."
)
parser.add_argument(
    "-bs", "--bar_size", type=str, required=False, default="1 hour", help="Candle size."
)
# REGRESSION CHANGE 1: choose the experiment without editing the source code.
parser.add_argument(
    "--task",
    choices=["classification", "regression", "multi_regression"],
    default="regression",
    help="Model type to train (default: regression).",
)
parser.add_argument(
    "--regression-targets",
    default=None,
    help=(
        "Optional comma-separated precomputed future-target columns. If omitted, "
        "the script creates future log-return targets from --return-horizons."
    ),
)
parser.add_argument(
    "--return-horizons",
    default=str(HORIZON),
    help="Comma-separated future return horizons, for example 1,5,20.",
)
args = parser.parse_args()
stock_symbol = args.symbol.lower()
bar_size = args.bar_size
task = args.task
__MODEL_PATH=f"{MODEL_PATH}-{task}-{bar_size}.keras"
__SCALER_PATH=f"{SCALER_PATH}-{task}-{bar_size}.pkl"
__FEATURE_META_PATH=f"{FEATURE_META_PATH}-{task}-{bar_size}.json"
return_horizons = [int(value.strip()) for value in args.return_horizons.split(",")]
if any(horizon <= 0 for horizon in return_horizons):
    raise ValueError("Every return horizon must be a positive integer.")
regression_targets = (
    [name.strip() for name in args.regression_targets.split(",") if name.strip()]
    if args.regression_targets
    else [f"stock_FutureLogReturn_{horizon}" for horizon in return_horizons]
)

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
    ignore = [
    column
    for column in df.columns if column.endswith((
        "_open",
        "_high",
        "_low",
        # "_close",
        # "_volume",
        "_Direction",
        "_Return",
        "_LogReturn",
        "ock_Count15",    
        "mh_Count15"    
    ))]
    ignore.append("datetime")  # Ensure datetime is excluded
    ignore.append("ticker")  # Ensure ticker is excluded
    ignore.append("ticker_id")  # Ensure ticker_id is excluded
    # Never allow a target column to become an input feature.
    ignore.extend(regression_targets)

    feature_columns = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]

    print(f"+++ Feature length: {len(feature_columns)} +++")
    # Quick check to ensure datetime is excluded!
    print(f"Sample features (first 5): {feature_columns}")  
    
    return df, feature_columns



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


# REGRESSION CHANGE 2: linear output; no sigmoid and no binary cross-entropy.
def build_regression_model(sequence_length, number_of_features, target_name):
    inputs = keras.Input(shape=(sequence_length, number_of_features))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear", name=target_name)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


# REGRESSION CHANGE 3: one linear output head for every continuous return target.
def build_multi_regression_model(
    sequence_length,
    number_of_features,
    target_names,
):
    inputs = keras.Input(shape=(sequence_length, number_of_features))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    shared = layers.Dense(16, activation="relu")(x)

    outputs = {
        name: layers.Dense(1, activation="linear", name=name)(shared)
        for name in target_names
    }
    losses = {name: "mse" for name in target_names}
    metrics = {
        name: [
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ]
        for name in target_names
    }

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=losses,
        metrics=metrics,
    )
    return model
def main():
    os.makedirs(os.path.dirname(__MODEL_PATH) or ".", exist_ok=True)
    clean_bar_name = str(bar_size).replace(" ", "").lower()
    print(f"Loading data {stock_symbol}...")
    df = pd.read_csv(f"{DATA_DIR}/semiconductor/combined_dataset_{clean_bar_name}.csv")

    # REGRESSION CHANGE 0: explicitly generate leakage-safe FUTURE returns.
    # At row t: target = log(close[t + horizon] / close[t]), calculated per ticker.
    if not args.regression_targets:
        df = df.sort_values(["ticker", "datetime"]).copy()
        for horizon, target_name in zip(return_horizons, regression_targets):
            future_close = df.groupby("ticker")["stock_close"].shift(-horizon)
            df[target_name] = np.log(future_close / df["stock_close"])

    if task in {"regression", "multi_regression"}:
        # The final rows of each ticker do not yet have a known future label.
        df = df.dropna(subset=regression_targets).copy()

    df_full, feature_columns = prepare_data(df)



    # Split chronologically BEFORE fitting scaler
    train_df, validation_df, test_df = chronological_split(df_full, TRAIN_FRAC, VAL_FRAC)

    # REGRESSION CHANGE 0b: purge boundary rows whose future target reaches
    # into the following split. This prevents train/validation label overlap.
    if task in {"regression", "multi_regression"} and not args.regression_targets:
        purge_size = max(return_horizons)
        train_boundary_indices = train_df.groupby("ticker").tail(purge_size).index
        validation_boundary_indices = validation_df.groupby("ticker").tail(purge_size).index
        train_df = train_df.drop(index=train_boundary_indices).copy()
        validation_df = validation_df.drop(index=validation_boundary_indices).copy()
    print(f"Train/Val/Test rows: {len(train_df)}/{len(validation_df)}/{len(test_df)}")
    print("Train:", train_df["datetime"].min()," to ", train_df["datetime"].max())

    print("Validation:", validation_df["datetime"].min()," to", validation_df["datetime"].max())

    print("Test:", test_df["datetime"].min()," to", test_df["datetime"].max())

    assert train_df["datetime"].max() < validation_df["datetime"].min()
    assert validation_df["datetime"].max() < test_df["datetime"].min()

    # Fit every scaler on TRAINING data only.
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_df[feature_columns])

    missing_targets = [
        name for name in regression_targets if name not in df_full.columns
    ]
    if missing_targets:
        raise ValueError(f"Missing regression target columns: {missing_targets}")
    if task == "regression" and len(regression_targets) != 1:
        raise ValueError("regression requires exactly one --regression-targets column")
    if task == "multi_regression" and len(regression_targets) < 2:
        raise ValueError("multi_regression requires at least two target columns")

    # REGRESSION CHANGE 4: scale each continuous target independently.
    # This is especially helpful when targets use different horizons/scales.
    target_scalers = {}
    if task in {"regression", "multi_regression"}:
        for name in regression_targets:
            target_scaler = StandardScaler()
            target_scaler.fit(train_df[[name]])
            target_scalers[name] = target_scaler

    def transform_classification(split_df):
        X = feature_scaler.transform(split_df[feature_columns]).astype("float32")
        y = split_df["stock_Direction"].to_numpy(dtype="float32")
        return X, y

    def transform_regression(split_df):
        target_name = regression_targets[0]
        X = feature_scaler.transform(split_df[feature_columns]).astype("float32")
        y = target_scalers[target_name].transform(
            split_df[[target_name]]
        ).ravel().astype("float32")
        return X, y

    def transform_multi_regression(split_df):
        X = feature_scaler.transform(split_df[feature_columns]).astype("float32")
        targets = {
            name: target_scalers[name].transform(
                split_df[[name]]
            ).ravel().astype("float32")
            for name in regression_targets
        }
        return X, targets

    scaler_bundle = {
        "feature_scaler": feature_scaler,
        "target_scalers": target_scalers,
    }
    joblib.dump(scaler_bundle, f"{__SCALER_PATH}")
    with open(__FEATURE_META_PATH, "w") as f:
        json.dump(
            {
                "feature_columns": feature_columns,
                "window_size": WINDOW_SIZE,
                "horizon": HORIZON,
                "task": task,
                "regression_targets": regression_targets,
                "targets_are_scaled": bool(target_scalers),
            },
            f,
            indent=2,
        )


    # REGRESSION CHANGE 5: create and train only the selected experiment.
    if task == "classification":
        transform_function = transform_classification
    elif task == "regression":
        transform_function = transform_regression
    else:
        transform_function = transform_multi_regression

    X_train_seq, y_train_seq = create_ticker_sequences(
        train_df, transform_function, WINDOW_SIZE
    )
    X_validation_seq, y_validation_seq = create_ticker_sequences(
        validation_df, transform_function, WINDOW_SIZE
    )
    X_test_seq, y_test_seq = create_ticker_sequences(
        test_df, transform_function, WINDOW_SIZE
    )

    if task == "classification":
        model = build_classification_model(
            X_train_seq.shape[1], X_train_seq.shape[2]
        )
    elif task == "regression":
        model = build_regression_model(
            X_train_seq.shape[1],
            X_train_seq.shape[2],
            regression_targets[0],
        )
    else:
        model = build_multi_regression_model(
            X_train_seq.shape[1],
            X_train_seq.shape[2],
            regression_targets,
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
        __MODEL_PATH, monitor="val_loss", save_best_only=True
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
    print("Training sequences:", len(X_train_seq))
    print("Task:", task)
    if task != "classification":
        print("Regression targets:", regression_targets)

    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(
            X_validation_seq,
            y_validation_seq
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


    test_results = model.evaluate(
        X_test_seq,
        y_test_seq,
        verbose=1,
        return_dict=True,
    )
    print("Test results:", test_results)

    if task == "classification":
        positive_rate = np.mean(y_test_seq)
        majority_accuracy = max(positive_rate, 1 - positive_rate)
        print("Positive rate:", positive_rate)
        print("Majority baseline accuracy:", majority_accuracy)
    else:
        # REGRESSION CHANGE 6: compare against a zero-return baseline and
        # report errors in the ORIGINAL return units, not standardized units.
        predictions = model.predict(X_test_seq, verbose=0)
        if task == "regression":
            predictions = {regression_targets[0]: np.asarray(predictions).ravel()}
            actuals = {regression_targets[0]: y_test_seq}
        else:
            if not isinstance(predictions, dict):
                predictions = dict(zip(model.output_names, predictions))
            actuals = y_test_seq

        for name in regression_targets:
            predicted_original = target_scalers[name].inverse_transform(
                np.asarray(predictions[name]).reshape(-1, 1)
            ).ravel()
            actual_original = target_scalers[name].inverse_transform(
                np.asarray(actuals[name]).reshape(-1, 1)
            ).ravel()
            mae = np.mean(np.abs(actual_original - predicted_original))
            rmse = np.sqrt(np.mean((actual_original - predicted_original) ** 2))
            zero_baseline_mae = np.mean(np.abs(actual_original))
            directional_accuracy = np.mean(
                np.sign(predicted_original) == np.sign(actual_original)
            )
            print(f"{name} original-scale MAE: {mae:.8f}")
            print(f"{name} original-scale RMSE: {rmse:.8f}")
            print(f"{name} zero-return baseline MAE: {zero_baseline_mae:.8f}")
            print(f"{name} directional accuracy: {directional_accuracy:.4f}")
    model.save(f"{__MODEL_PATH}")
    print(f"Model saved to {__MODEL_PATH}")


if __name__ == "__main__":
    main()