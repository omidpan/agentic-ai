from pathlib import Path

import pandas as pd
import numpy as np
WINDOW_SIZE=15
HORIZON=1
BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# Configuration
# --------------------------------------------------

dataset_names = {
    "semiconductors": [
        "nvda",
        "amd",
        "avgo",
        "intc",
        "mrvl",
        "mu",
        "tsm",
        "amat",
    ]
}

dataset_context = {
    # "semiconductors": ["smh", "spy"]
    # "semiconductors": ['smh']
}

# bar_size = "1day"
bar_size = "4hours"
session = "extended"

REQUIRED_COLUMNS = {
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def safe_divide(
        numerator: pd.Series,
        denominator: pd.Series
    ) -> pd.Series:
        denominator = denominator.replace(0, np.nan)
        return numerator.div(denominator)
# --------------------------------------------------
def add_trend_feature(df: pd.DataFrame, feature_prefix: str) -> pd.DataFrame:
    df = df.copy()
    close = df["close"].astype(float)

    ema20 = close.ewm(span=20,adjust=False).mean()
    ema50 = close.ewm(span=50,adjust=False).mean()
    df[f"{feature_prefix}_EMA_Ratio"] = safe_divide(ema20,ema50) - 1
    return df
def add_candle_features(
    df: pd.DataFrame,
    feature_prefix: str
) -> pd.DataFrame:

    df = df.copy()

    body = df["close"] - df["open"]
    candle_range = df["high"] - df["low"]

    upper_shadow = (
        df["high"]
        - df[["open", "close"]].max(axis=1)
    )

    lower_shadow = (
        df[["open", "close"]].min(axis=1)
        - df["low"]
    )

    df[f"{feature_prefix}_BodyPct"] = safe_divide(
        body,
        df["open"]
    )

    df[f"{feature_prefix}_Range_Pct"] = safe_divide(
        candle_range,
        df["close"]
    )

    df[f"{feature_prefix}_ShadowImbalance"] = safe_divide(
        lower_shadow - upper_shadow,
        candle_range
    )
    
    df[f"{feature_prefix}_BodyToRange"] = safe_divide(
            np.abs(body),
            candle_range
        )
    
    df[f"{feature_prefix}_Bullish"] = (
            df["close"] > df["open"]
        ).astype(int)
    return df
def add_gap_features(
    df: pd.DataFrame,
    feature_prefix: str
) -> pd.DataFrame:

    df = df.copy()

    df[f"{feature_prefix}_GapPct"] = safe_divide(
        df["open"] - df["close"].shift(1),
        df["close"].shift(1))
    # df[f"{feature_prefix}_GapPct_Missing"] = df[f"{feature_prefix}_GapPct"].isna().astype(int)
    df[f"{feature_prefix}_GapPct"] = df[f"{feature_prefix}_GapPct"].fillna(0.0)

    return df
def add_rolling_feature(
    df: pd.DataFrame,
    window: int = WINDOW_SIZE,
    feature_prefix: str = "stock"
) -> pd.DataFrame:
    """
    Add rolling volatility using current and previous log returns.

    Keeps all rows:
    - Uses partial history when fewer than `window` returns exist.
    - Records how much history was available.
    - Replaces unavoidable initial NaN with a neutral value.
    """

    df = df.copy()

    log_return_column = f"{feature_prefix}_LogReturn"
    rolling_std_column = (
        f"{feature_prefix}_RollingStd{window}"
    )
    history_count_column = (
        f"{feature_prefix}_HistoryCount{window}"
    )
    full_history_column = (
        f"{feature_prefix}_FullHistory{window}"
    )

    returns = df[log_return_column]

    # Number of valid returns currently available
    df[history_count_column] = (
        returns
        .rolling(
            window=window,
            min_periods=1
        )
        .count()
    )

    # 1 when the complete rolling window is available
    df[full_history_column] = (
        df[history_count_column] >= window
    ).astype(int)

    # Standard deviation requires at least two valid returns
    df[rolling_std_column] = (
        returns
        .rolling(
            window=window,
            min_periods=2
        )
        .std()
    )

    # Neutral replacement for unavoidable initial NaN values
    df[rolling_std_column] = (
        df[rolling_std_column]
        .fillna(0.0)
    )

    return df

def add_zscore_feature(
    df: pd.DataFrame,
    window: int = WINDOW_SIZE,
    feature_prefix: str = "stock"
) -> pd.DataFrame:
    """
    Add the rolling z-score of the current log return.

    Uses only the current and previous observations.
    Preserves all rows without using future information.
    """

    df = df.copy()

    return_column = f"{feature_prefix}_LogReturn"
    zscore_column = f"{feature_prefix}_ReturnZ{window}"
    # missing_column = f"{zscore_column}_Missing"

    returns = df[return_column]

    # Partial windows preserve the early rows.
    # At least two valid returns are needed for standard deviation.
    rolling_mean = returns.rolling(
        window=window,
        min_periods=2
    ).mean()

    rolling_std = returns.rolling(
        window=window,
        min_periods=2
    ).std()

    # Treat zero or extremely small standard deviation as unavailable.
    valid_std = rolling_std.mask(
        rolling_std.abs() < 1e-12
    )

    df[zscore_column] = safe_divide(
        returns - rolling_mean,
        valid_std
    )


    # A z-score of zero is the neutral replacement.
    df[zscore_column] = (
        df[zscore_column]
        .fillna(0.0)
    )  
    return df
def add_spy_features(
    df: pd.DataFrame,
    feature_prefix: str,
) -> pd.DataFrame:
    '''
    Add features for SPY that are relevant to semiconductor stocks.
    '''
    spy_df = (
    df
    .sort_values("datetime")
    .drop_duplicates(subset="datetime", keep="last")
    .reset_index(drop=True)
    .copy())
    previous_close = spy_df["close"].shift(1)
    # ---------------------------------------------------------
    # Normalized SPY candle features
    # ---------------------------------------------------------

    # Open-to-close percentage movement
    spy_df[f"{feature_prefix}_intraday_return"] = (
        spy_df["close"] - spy_df["open"]
    ) / spy_df["open"]

    # Intraday high-low range relative to the opening price
    spy_df[f"{feature_prefix}_high_low_range"] = (
        spy_df["high"] - spy_df["low"]
    ) / spy_df["open"]

    # Position of the closing price inside the daily range
    # 0.0 = closed at the low
    # 0.5 = closed in the middle
    # 1.0 = closed at the high
    daily_range = spy_df["high"] - spy_df["low"]

    spy_df[f"{feature_prefix}_close_position"] = np.where(
        daily_range.ne(0),
        (spy_df["close"] - spy_df["low"]) / daily_range,
        0.5
    )

    # Overnight gap compared with the previous trading day's close
    
    spy_df[f"{feature_prefix}_GapPct"] = safe_divide(
        spy_df["open"] - spy_df["close"].shift(1),
        spy_df["close"].shift(1))
    spy_df[f"{feature_prefix}_GapPct"] = spy_df[f"{feature_prefix}_GapPct"].fillna(0.0)    
    candle_range = spy_df["high"] - spy_df["low"]
    spy_df[f"{feature_prefix}_Range_Pct"] = safe_divide(
            candle_range,
            spy_df["close"]
        )
    df=spy_df.copy()
    df= create_target(spy_df, horizon=HORIZON, feature_prefix=feature_prefix)
    return df 
def add_features(df: pd.DataFrame, feature_prefix: str) -> pd.DataFrame:
    '''We don't need all the features for spy because of heatmap
        correlation with stock datasets.
    '''
    if(feature_prefix=='spy'):
        df=add_spy_features(df, feature_prefix)
    else:
        df=create_target(df, horizon=HORIZON, feature_prefix=feature_prefix)
        df=add_trend_feature(df,feature_prefix=feature_prefix)
        df=add_candle_features(df,feature_prefix=feature_prefix)
        df=add_gap_features(df,feature_prefix=feature_prefix)
        df=add_rolling_feature(df, window=WINDOW_SIZE,feature_prefix=feature_prefix)
        df=add_zscore_feature(df, window=WINDOW_SIZE,feature_prefix=feature_prefix)

    return df



########## complete feature engineering and return targets+++++++++++++++
# 2 return targets
# ----------------------------------------------------------
# Target Generation
# ----------------------------------------------------------

def create_target(
    df: pd.DataFrame,
    horizon: int = HORIZON,
    feature_prefix: str | None = 'stock',
) -> pd.DataFrame:
    """
    Creates prediction targets.

    horizon=1

        Predict next candle.

    horizon=2

        Predict two candles ahead.
    """

    df = df.copy()

    # Future percentage return

    df[f"{feature_prefix}_Return"] = (df["close"].shift(-horizon)/df["close"]- 1.0)
    df[f"{feature_prefix}_Return"] = df[f"{feature_prefix}_Return"].fillna(0.0)

    # Future log return

    df[f"{feature_prefix}_LogReturn"] = np.log(df["close"].shift(-horizon)/df["close"])
    df[f"{feature_prefix}_LogReturn"] = df[f"{feature_prefix}_LogReturn"].fillna(0.0)

    # Binary direction

    df[f"{feature_prefix}_Direction"] = (df[f"{feature_prefix}_Return"] > 0).astype(int)
    return df


# --------------------------------------------------
# Load and validate one CSV
# --------------------------------------------------

def load_dataset(
    base_dir: Path,
    name: str,
    context_type: str | None = None,
    feature_prefix: str | None = None,
) -> tuple[str, pd.DataFrame]:
    """Load, validate, sort, and add features to one instrument CSV."""

    instrument_name = name.strip().lower()
    filename = f"{instrument_name}_{bar_size}_{session}.csv"

    if context_type is None:
        file_path = base_dir / filename
    else:
        file_path = base_dir.parent / "context" / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        parse_dates=["datetime"],
    )

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    duplicate_mask = df["datetime"].duplicated()

    if duplicate_mask.any():
        duplicate_dates = df.loc[
            duplicate_mask,
            "datetime",
        ].tolist()

        raise ValueError(
            f"{file_path.name} contains duplicate dates: "
            f"{duplicate_dates}"
        )

    if (df["close"] == 0).any():
        raise ValueError(
            f"{file_path.name} contains a zero close price."
        )

    df = (
        df
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    prefix = feature_prefix or instrument_name
    df = add_features(df, feature_prefix=prefix)
    return instrument_name, df


# --------------------------------------------------
# 1. Load stock datasets
# --------------------------------------------------

if __name__ == "__main__":
    stock_datasets: dict[str, pd.DataFrame] = {}

    for group, ticker_names in dataset_names.items():
        for ticker_name in ticker_names:
            dataset_name, df = load_dataset(
                BASE_DIR,
                ticker_name,
                feature_prefix="stock",
            )

            df["ticker"] = dataset_name.upper()
            df["group"] = group

            stock_datasets[dataset_name] = df


    # --------------------------------------------------
    # 2. Combine stocks vertically
    # --------------------------------------------------

    if not stock_datasets:
        raise ValueError("No stock datasets were loaded.")

    stocks = pd.concat(stock_datasets.values(),ignore_index=True,)
    # Final chronological order for training and splitting


    # --------------------------------------------------
    # 3. Rename shared stock OHLCV columns
    # --------------------------------------------------

    stocks = stocks.rename(columns={
        "open": "stock_open",
        "high": "stock_high",
        "low": "stock_low",
        "close": "stock_close",
        "volume": "stock_volume",
    })


    # --------------------------------------------------
    # 4. Create ticker IDs for an embedding layer
    # --------------------------------------------------

    unique_tickers = sorted(stocks["ticker"].unique())

    ticker_to_id = {
        ticker: ticker_id
        for ticker_id, ticker in enumerate(unique_tickers)
    }

    stocks["ticker_id"] = (
        stocks["ticker"]
        .map(ticker_to_id)
        .astype("int32")
    )

    print("Ticker mapping:", ticker_to_id)


    # --------------------------------------------------
    # 5. One-hot encode stock groups
    # --------------------------------------------------

    stocks = pd.get_dummies(
        stocks,
        columns=["group"],
        prefix="group",
        dtype="int8",
    )


    # --------------------------------------------------
    # 6. Load context datasets
    # --------------------------------------------------

    context_datasets: dict[str, pd.DataFrame] = {}

    for context_type, context_names in dataset_context.items():
        for context_name in context_names:
            dataset_name, df = load_dataset(
                BASE_DIR,
                context_name,
                context_type=context_type,
                feature_prefix=context_name.lower(),
            )
            ####### if the context dataset is spy ,we need to 
            # modify features because of correlation with stock datasets
            
            # Keep datetime unchanged for the merge. Prefix all
            # context OHLCV fields to prevent name collisions.
            df = df.rename(columns={
                "open": f"{dataset_name}_open",
                "high": f"{dataset_name}_high",
                "low": f"{dataset_name}_low",
                "close": f"{dataset_name}_close",
                "volume": f"{dataset_name}_volume",
            })

            context_datasets[dataset_name] = df


    # --------------------------------------------------
    # 7. Merge contexts horizontally by datetime
    # --------------------------------------------------

    combined = stocks.copy()

    for context_name, context_df in context_datasets.items():
        combined = combined.merge(
            context_df,
            on="datetime",
            how="left",
            validate="many_to_one",
        )


    # --------------------------------------------------
    # 8. Sort each stock time series chronologically
    # --------------------------------------------------

    combined = (
        combined
        .sort_values(
            ["datetime", "ticker"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------
    # 9. Verify that every stock date has context data
    # --------------------------------------------------

    context_feature_columns = [
    column
    for context_name in context_datasets
    for column in [
        f"{context_name}_open",
        f"{context_name}_high",
        f"{context_name}_low",
        f"{context_name}_close",
        f"{context_name}_volume",
        f"{context_name}_Range_Pct",
        ]
    ]

    missing_context_mask = combined[
        context_feature_columns
    ].isna().any(axis=1)

    if missing_context_mask.any():
        problem_rows = combined.loc[
            missing_context_mask,
            ["datetime", "ticker"],
        ]
        print("Some stock dates have no matching context data:")
        print(problem_rows.to_string(index=False))

        raise ValueError(
            "Some stock dates have no matching context data:\n"
            f"{problem_rows.to_string(index=False)}"
        )


    # --------------------------------------------------
    # 10. Inspect and save the completed dataset
    # --------------------------------------------------

    print("Combined columns:")
    # Convert empty or whitespace-only strings to NaN
    combined = combined.replace(r"^\s*$", np.nan, regex=True)

    # Check whether the DataFrame contains any missing value
    has_missing = combined.isna().any().any()
   ### for 4 hours dataset , drop stock_ShadowImbalance column because of missing values
    if bar_size=="4hours":
        combined.drop(columns=["stock_ShadowImbalance"], inplace=True)
    print("Has missing or empty values:", has_missing)
    rows_with_missing = combined[combined.isna().any(axis=1)]
    print(rows_with_missing)
    print(len(combined.columns))
    # print(combined.columns.tolist())

    # print("First two combined rows:")
    # # print(combined.head(2).to_string(index=False))

    output_path = BASE_DIR / f"combined_dataset_{bar_size}.csv"
    
################################## correlation matrix ###################
    import matplotlib.pyplot as plt
    import seaborn as sns
    correlation_columns = [
    column for column in combined.columns
     if column.startswith(("stock_", "smh_", "spy_")) 
     and not column.endswith(("_Direction", "_Return", "_LogReturn"))
    ]
    
    ## show number of columns and rows
    print(f"Number of columns: {len(combined.columns)}")
    print(f"Number of rows: {len(combined)}")
    
    correlation_matrix = combined[correlation_columns].corr()
    # correlation_matrix = df.corr(numeric_only=True,method='pearson')
    
    plt.figure(figsize=(30, 15))

    sns.heatmap(
        correlation_matrix,
        annot=True,       # Display correlation numbers
        fmt=".2f",        # Show two decimal places
        cmap="coolwarm",  # Blue-white-red colors
        center=0,
        vmin=-1,
        vmax=1
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()
###############################################################
    
    combined.to_csv(output_path, index=False)

    print(f"Combined dataset saved to: {output_path}")