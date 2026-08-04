"""
feature_engineering.py

Feature engineering utilities for LSTM stock prediction.

All features are computed WITHOUT future leakage.
Only current and historical candles are used.
"""
'''
I would redesign the entire feature engineering into an object-oriented class rather than a collection of standalone functions. For example:

fe = FeatureEngineer(
    horizon=1,
    lookback=20,
    market_context=True,
    add_volume=True,
    add_zscores=True,
    add_statistics=True,
)

df, feature_columns = fe.transform(
    stock_df,
    spy_df=spy_df,
    qqq_df=qqq_df,
    vix_df=vix_df,
    sector_df=sector_df,
)

This makes the pipeline much easier to configure,
test, and reuse for both training and real-time inference.

One more recommendation

Looking at your overall trading system architecture,
I think the next major improvement should not be adding more technical indicators.
Instead, I'd focus on making the feature pipeline configuration-driven (via a YAML or JSON config)
so you can enable or disable entire feature groups, experiment with different lookback windows,
and run ablation studies without changing the code. That will make it much easier
to identify which features genuinely 
improve directional accuracy and which ones simply add noise.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from config import DATA_DIR,WINDOW_SIZE
parser = argparse.ArgumentParser(description="Process a stock symbol.")
parser.add_argument("-s" ,"--symbol",
                    type=str,
                    required=False,
                    help="The stock symbol to process. default is NVDA.", 
                    nargs='?', default="NVDA")
parser.add_argument("-bs", "--bar_size",
                    type=str,
                    required=False,
                    help="candle size of historical data. default is 1 hour.", 
                    nargs='?', default="1 hour")
args = parser.parse_args()

# Access the value using dot notation
stock_symbol = args.symbol.lower()
bar_size = args.bar_size
# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------

EPSILON = 1e-10


# ----------------------------------------------------------
# Helper
# ----------------------------------------------------------

def safe_divide(a, b):
    """
    Safe division that prevents divide-by-zero.
    """
    return a / (b + EPSILON)


# ----------------------------------------------------------
# Log Return Features
# ----------------------------------------------------------

def add_log_returns(df: pd.DataFrame,rotate:int) -> pd.DataFrame:
    """
    Adds logarithmic return features.

    Features
    --------
    LogReturn1
    LogReturn2
    LogReturn3
    LogReturn5
    LogReturn10
    """
    if(rotate>5):
        rotate=5
    df = df.copy()
  
    close = df["close"]
    for i in range(1,rotate):
        df[f"LogReturn{i}"] = np.log(close / close.shift(i))

    return df


# ----------------------------------------------------------
# Candle Features
# ----------------------------------------------------------

def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates candle structure features.

    These features often contain more predictive
    information than SMA/EMA indicators.
    """

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

    df["Body"] = body

    df["BodyPct"] = safe_divide(
        body,
        df["open"]
    )

    df["Range"] = candle_range

    df["RangePct"] = safe_divide(
        candle_range,
        df["close"]
    )

    df["UpperShadow"] = upper_shadow

    df["LowerShadow"] = lower_shadow

    df["UpperShadowPct"] = safe_divide(
        upper_shadow,
        candle_range
    )

    df["LowerShadowPct"] = safe_divide(
        lower_shadow,
        candle_range
    )

    df["BodyToRange"] = safe_divide(
        np.abs(body),
        candle_range
    )

    df["Bullish"] = (
        df["close"] > df["open"]
    ).astype(int)

    return df


# ----------------------------------------------------------
# Gap Features
# ----------------------------------------------------------

def add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gap features.

    Particularly useful for daily candles,
    but still informative for 4-hour bars.
    """

    df = df.copy()

    previous_close = df["close"].shift(1)

    gap = df["open"] - previous_close

    df["Gap"] = gap

    df["GapPct"] = safe_divide(
        gap,
        previous_close
    )

    df["GapDirection"] = np.sign(gap)

    df["GapUp"] = (gap > 0).astype(int)

    df["GapDown"] = (gap < 0).astype(int)

    return df


# ----------------------------------------------------------
# Build Part-1 Features
# ----------------------------------------------------------

def add_part1_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all Part-1 feature engineering.
    """

    df = add_log_returns(df,1)

    df = add_candle_features(df)

    df = add_gap_features(df)

    return df
# ----------------------------------------------------------
# Volume Features
# ----------------------------------------------------------

def add_volume_features(
    df: pd.DataFrame,
    window: int = WINDOW_SIZE
) -> pd.DataFrame:
    """
    Volume based features.

    These features measure whether the current
    volume is unusual relative to recent history.

    Features
    --------
    VolumeMA20
    VolumeEMA20
    RelativeVolume
    VolumeChange
    VolumeZ
    VolumeEMARatio
    DollarVolume
    DollarVolumeMA20
    DollarVolumeRatio
    """

    df = df.copy()

    volume = df["volume"]

    close = df["close"]

    # Moving average

    df["VolumeMA20"] = (
        volume
        .rolling(window)
        .mean()
    )

    # EMA

    df["VolumeEMA20"] = (
        volume
        .ewm(span=window, adjust=False)
        .mean()
    )

    # Relative Volume

    df["RelativeVolume"] = safe_divide(
        volume,
        df["VolumeMA20"]
    )

    # Volume change

    df["VolumeChange"] = volume.pct_change()

    # Volume Z-score

    vol_mean = volume.rolling(window).mean()

    vol_std = volume.rolling(window).std()

    df["VolumeZ"] = safe_divide(
        volume - vol_mean,
        vol_std
    )

    # EMA Ratio

    df["VolumeEMARatio"] = safe_divide(
        volume,
        df["VolumeEMA20"]
    )

    # Dollar Volume

    df["DollarVolume"] = close * volume

    df["DollarVolumeMA20"] = (
        df["DollarVolume"]
        .rolling(window)
        .mean()
    )

    df["DollarVolumeRatio"] = safe_divide(
        df["DollarVolume"],
        df["DollarVolumeMA20"]
    )

    return df


# ----------------------------------------------------------
# Volatility Features
# ----------------------------------------------------------

def add_volatility_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    """
    Volatility features based on returns.

    More useful than raw ATR alone.
    """

    df = df.copy()

    returns = np.log(
        df["close"] /
        df["close"].shift(1)
    )

    df["Volatility5"] = (
        returns
        .rolling(5)
        .std()
    )

    df["Volatility10"] = (
        returns
        .rolling(10)
        .std()
    )

    df["Volatility20"] = (
        returns
        .rolling(20)
        .std()
    )

    # ATR normalization

    if "ATR_14" in df.columns:

        df["ATRPct"] = safe_divide(
            df["ATR_14"],
            df["close"]
        )

    # High-Low %

    df["HighLowPct"] = safe_divide(
        df["high"] - df["low"],
        df["close"]
    )

    # True Range %

    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            abs(df["high"] - previous_close),
            abs(df["low"] - previous_close)
        ],
        axis=1
    ).max(axis=1)

    df["TrueRangePct"] = safe_divide(
        tr,
        previous_close
    )

    return df


# ----------------------------------------------------------
# Trend Features
# ----------------------------------------------------------

def add_trend_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    """
    Trend based features.

    Prefer these over multiple SMA's.
    """

    df = df.copy()

    close = df["close"]

    # EMA

    df["EMA20"] = (
        close
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["EMA50"] = (
        close
        .ewm(span=50, adjust=False)
        .mean()
    )

    # EMA Ratio

    df["EMA_Ratio"] = safe_divide(
        df["EMA20"],
        df["EMA50"]
    )

    # Price above EMA

    df["PriceEMA20"] = safe_divide(
        close,
        df["EMA20"]
    )

    df["PriceEMA50"] = safe_divide(
        close,
        df["EMA50"]
    )

    # EMA slopes

    df["EMA20Slope"] = (
        df["EMA20"]
        .diff()
    )

    df["EMA50Slope"] = (
        df["EMA50"]
        .diff()
    )

    # Rolling trend

    df["Trend5"] = (
        close
        .pct_change(5)
    )

    df["Trend10"] = (
        close
        .pct_change(10)
    )

    df["Trend20"] = (
        close
        .pct_change(20)
    )

    return df


# ----------------------------------------------------------
# Apply Part 2
# ----------------------------------------------------------

def add_part2_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = add_volume_features(df)

    df = add_volatility_features(df)

    df = add_trend_features(df)

    return df
# ----------------------------------------------------------
# Market Context
# ----------------------------------------------------------

def add_market_context(
    stock_df: pd.DataFrame,
    spy_df: pd.DataFrame = None,
    qqq_df: pd.DataFrame = None,
    vix_df: pd.DataFrame = None,
    sector_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Merge market context into stock dataframe.

    All dataframes MUST have

        datetime
        close

    columns.

    They should already be aligned to the same
    timeframe (4H or Daily).

    Features created

    SPY_Return

    QQQ_Return

    VIX_Return

    Sector_Return
    """

    df = stock_df.copy()

    if "datetime" not in df.columns:
        raise ValueError(
            "stock dataframe must contain datetime column"
        )

    def merge_return(df, market_df, feature_name):

        if market_df is None:
            return df

        temp = market_df.copy()

        temp = temp[["datetime", "close"]]

        temp[feature_name] = (
            temp["close"].pct_change()
        )

        temp = temp.drop(columns="close")

        df = df.merge(
            temp,
            on="datetime",
            how="left"
        )

        return df

    df = merge_return(df, spy_df, "SPY_Return")

    df = merge_return(df, qqq_df, "QQQ_Return")

    df = merge_return(df, vix_df, "VIX_Return")

    df = merge_return(df, sector_df, "Sector_Return")

    return df


# ----------------------------------------------------------
# Relative Strength
# ----------------------------------------------------------

def add_relative_strength(
    df: pd.DataFrame
) -> pd.DataFrame:

    """
    Relative performance versus market.

    Very powerful features.
    """

    df = df.copy()

    if "LogReturn1" not in df.columns:

        df["LogReturn1"] = np.log(
            df["close"] /
            df["close"].shift(1)
        )

    if "SPY_Return" in df.columns:

        df["RelativeStrength_SPY"] = (
            df["LogReturn1"]
            -
            df["SPY_Return"]
        )

    if "QQQ_Return" in df.columns:

        df["RelativeStrength_QQQ"] = (
            df["LogReturn1"]
            -
            df["QQQ_Return"]
        )

    if "Sector_Return" in df.columns:

        df["RelativeStrength_Sector"] = (
            df["LogReturn1"]
            -
            df["Sector_Return"]
        )

    return df


# ----------------------------------------------------------
# Rolling Statistics
# ----------------------------------------------------------

def add_rolling_statistics(
    df: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:

    """
    Rolling statistical features.
    """

    df = df.copy()

    returns = df["LogReturn1"]

    df["RollingMean"] = (
        returns
        .rolling(window)
        .mean()
    )

    df["RollingMedian"] = (
        returns
        .rolling(window)
        .median()
    )

    df["RollingStd"] = (
        returns
        .rolling(window)
        .std()
    )

    df["RollingVariance"] = (
        returns
        .rolling(window)
        .var()
    )

    df["RollingMin"] = (
        returns
        .rolling(window)
        .min()
    )

    df["RollingMax"] = (
        returns
        .rolling(window)
        .max()
    )

    df["RollingSkew"] = (
        returns
        .rolling(window)
        .skew()
    )

    df["RollingKurtosis"] = (
        returns
        .rolling(window)
        .kurt()
    )

    df["RollingQuantile25"] = (
        returns
        .rolling(window)
        .quantile(0.25)
    )

    df["RollingQuantile75"] = (
        returns
        .rolling(window)
        .quantile(0.75)
    )

    return df


# ----------------------------------------------------------
# Z-Score Features
# ----------------------------------------------------------

def add_zscore_features(
    df: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:

    """
    Rolling z-score features.
    """

    df = df.copy()

    def zscore(series):

        mean = (
            series
            .rolling(window)
            .mean()
        )

        std = (
            series
            .rolling(window)
            .std()
        )

        return (
            series - mean
        ) / (std + EPSILON)

    df["CloseZ"] = zscore(
        df["close"]
    )

    df["VolumeZ"] = zscore(
        df["volume"]
    )

    if "ATR_14" in df.columns:

        df["ATRZ"] = zscore(
            df["ATR_14"]
        )

    if "LogReturn1" in df.columns:

        df["ReturnZ"] = zscore(
            df["LogReturn1"]
        )

    return df


# ----------------------------------------------------------
# Apply Part 3
# ----------------------------------------------------------

def add_part3_features(
    df: pd.DataFrame,
    spy_df=None,
    qqq_df=None,
    vix_df=None,
    sector_df=None
):

    df = add_market_context(
        df,
        spy_df,
        qqq_df,
        vix_df,
        sector_df
    )

    df = add_relative_strength(df)

    df = add_rolling_statistics(df)

    df = add_zscore_features(df)

    return df
# ----------------------------------------------------------
# Target Generation
# ----------------------------------------------------------

def create_target(
    df: pd.DataFrame,
    horizon: int = 1
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

    df["Target_Return"] = (
        df["close"].shift(-horizon)
        /
        df["close"]
        - 1.0
    )

    # Future log return

    df["Target_LogReturn"] = np.log(
        df["close"].shift(-horizon)
        /
        df["close"]
    )

    # Binary direction

    df["Target_Direction"] = (
        df["Target_Return"] > 0
    ).astype(int)
    return df


# ----------------------------------------------------------
# Remove invalid rows
# ----------------------------------------------------------

def clean_dataset(
    df: pd.DataFrame
) -> pd.DataFrame:

    """
    Removes rows containing NaN or infinity.
    """
    df=df.copy()
#     feature_columns = [column for column in df.columns
#                        if column.startswith(("stock_", "smh_", "spy_"))
#         and not column.endswith((
#         "_open",
#         # "_close",
#         "_high",
#         "_low",
#         # "_volume",
#         "_Direction",
#         "_Return",
#         "_LogReturn",
#         "ock_Count15",    
#         "mh_Count15",
#         "_ShadowImbalance",   
#     ))
# ]


    feature_report = pd.DataFrame({
    "nan_count": df.isna().sum(),
    "positive_inf": df.eq(np.inf).sum(),
    "negative_inf": df.eq(-np.inf).sum(),
    # "min": df.replace(
    #     [np.inf, -np.inf], np.nan
    # ).min(),
    # "max": df.replace(
    #     [np.inf, -np.inf], np.nan
    # ).max(),
    'empty_string_count': (df == '').sum(),
})
    # Must be > 0 for nan, positive_inf, empty_string, AND < 0 for negative_inf
    filtered_report = feature_report[
        (feature_report["nan_count"] > 0) &
        (feature_report["positive_inf"] > 0) &
        (feature_report["empty_string_count"] > 0) &
        (feature_report["negative_inf"] < 0)
    ]

    # 2. Filter the COLUMNS of the report itself
    # This removes any metric column (like 'min' or 'max') if it only has 0s left
    filtered_report = filtered_report.loc[:, (filtered_report != 0).any(axis=0)]

    # Display the clean, issue-only report

    print(filtered_report)
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # df = df.dropna()

    df = df.reset_index(drop=True)
    print(df.isna().sum().sort_values(ascending=False))
    return df


# ----------------------------------------------------------
# Feature Columns
# ----------------------------------------------------------

def get_feature_columns(
    df: pd.DataFrame
):

    ignore = {

        "datetime",

        "Target_Return",

        "Target_LogReturn",

        "Target_Direction"

    }

    feature_columns = [

        c

        for c in df.columns

        if c not in ignore

    ]
    print(f'+++  Feature length: {len(feature_columns)}  +++')
    return feature_columns


# ----------------------------------------------------------
# Complete Pipeline
# ----------------------------------------------------------

def prepare_features(
    stock_df: pd.DataFrame,
    horizon: int = 1,
    spy_df: pd.DataFrame = None,
    qqq_df: pd.DataFrame = None,
    vix_df: pd.DataFrame = None,
    sector_df: pd.DataFrame = None,
    
):

    """
    Complete feature engineering pipeline.

    Returns

    --------

    df

    feature_columns
    """

    df = stock_df.copy()

    # -------------------------
    # Part 1
    # -------------------------

    df = add_part1_features(df)

    # -------------------------
    # Part 2
    # -------------------------

    # df = add_part2_features(df)

    # -------------------------
    # Part 3
    # -------------------------

    # df = add_part3_features(df,spy_df,qqq_df,vix_df,sector_df)

    # -------------------------
    # Target
    # -------------------------

    # df = create_target(df,horizon)

    # -------------------------
    # Cleanup
    # -------------------------

    df = clean_dataset(df)

    # -------------------------
    # Feature list
    # -------------------------

    feature_columns = get_feature_columns(df)

    return (df,feature_columns)

def refactor_columns(df):
    df=df.copy()
    exclude = {
        "ticker",
        "ticker_id",
        "group_semiconductors",
        "datetime",
    }
    
    df = df.rename(
            columns=lambda col: (
                col.lower()
                if col.lower() in exclude or col.lower().startswith("stock_") or (
                    col.lower().startswith("smm_") or col.lower().startswith("spy_")
                )
                else f"stock_{col.lower()}"
            )
        )
    return df
########## display ##########
def display(df):
    feature_columns=df.columns
    correlation_matrix = df[feature_columns].corr(numeric_only=True,method='pearson')
    
    plt.figure(figsize=(20, 15))

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
# --------------------------------------------------
    # 4. Create ticker IDs for an embedding layer
    # --------------------------------------------------
def feature_map_and_encoder(stocks:pd.DataFrame):
    stocks=stocks.copy()
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
    stocks['semiconductors']=stocks['group_semiconductors']
    stocks = pd.get_dummies(
            stocks,
            columns=["semiconductors"],
            prefix='group_semiconductor',
            dtype="int8",
        )
    stocks.sort_values(["datetime", "ticker"],ascending=[True, True],).reset_index(drop=True)
    return stocks



# ----------------------------------------------------------
# Executing features
# ----------------------------------------------------------
if __name__ == "__main__":
        # read data from CSV, add indicators, and save to new CSV for training
    clean_bar_name = str(bar_size).replace(" ", "").lower()
    df = pd.read_csv(f"{DATA_DIR}/semiconductor/dataset_{clean_bar_name}.csv")
    df=df.copy()
    df = df.rename(
    columns=lambda col: (
        col.lower()
        .removeprefix("stock_")
    )
)
    # spy_df=pd.read_csv(f"{DATA_DIR}/spy_{clean_bar_name}.csv")
    # qqq_df=pd.read_csv(f"{DATA_DIR}/qqq_{clean_bar_name}.csv")
    df, feature_columns = prepare_features(df,horizon=1)
    df=refactor_columns(df)
    df=feature_map_and_encoder(df)
    df=clean_dataset(df)
    display(df)
    print(df.head())

    print(feature_columns)

    print(len(feature_columns),"features")
    
    # df.to_csv(f"{DATA_DIR}/{stock_symbol}_{clean_bar_name}_WIP.csv", index=False)
    df.to_csv(f"{DATA_DIR}/dataset_{clean_bar_name}_features.csv", index=False)