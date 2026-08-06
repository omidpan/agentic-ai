import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
BASE_DIR = Path(__file__).resolve().parent
SESSION = "extended"
EPSILON = 1e-10

BAR_SIZE_ALIASES = {
    "1hour": "1hour",
    "1 hour": "1hour",
    "1h": "1hour",
    "4hours": "4hours",
    "4 hours": "4hours",
    "4hour": "4hours",
    "4 hour": "4hours",
    "4h": "4hours",
    "1day":"1day",
    "1d":"1day"
}
PRICE_VOLUME_COLUMNS = ["open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
REQUIRED_COLUMNS = {"datetime", *PRICE_VOLUME_COLUMNS}

def parse_arguments():
    parser = ArgumentParser(
        description="Repair missing candles in one stock CSV."
    )
    parser.add_argument(
        "-s",
        "--symbol",
        default="NVDA",
        help="Stock symbol; default: NVDA.",
    )
    parser.add_argument(
        "-bs",
        "--bar-size",
        default="1hour",
        help="Supported values: 1hour or 4hours; default: 1hour.",
    )
    parser.add_argument(
        "-c",
        "--context",
        action="store_true",
        help="Read the input CSV from the ../context directory.",
    )
    return parser.parse_args()


def normalize_bar_size(value: str) -> str:
    normalized = " ".join(value.lower().strip().split())
    if normalized not in BAR_SIZE_ALIASES:
        raise ValueError(
            f"Unsupported bar size {value!r}. Use 1hour or 4hours."
        )
    return BAR_SIZE_ALIASES[normalized]

######### candle features
def safe_divide(numerator, denominator, zero_value=EPSILON):
    numerator = pd.Series(
        numerator,
        index=denominator.index,
        dtype=float,
    )

    denominator = pd.Series(
        denominator,
        index=denominator.index,
        dtype=float,
    )

    result = pd.Series(
        zero_value,
        index=denominator.index,
        dtype=float,
    )

    valid = denominator.notna() & denominator.ne(0)

    result.loc[valid] = (
        numerator.loc[valid]
        / denominator.loc[valid]
    )

    # Preserve NaN when the original inputs are missing.
    missing_input = numerator.isna() | denominator.isna()
    result.loc[missing_input] = np.nan

    return result

######### gap features
def add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds gap features to one chronologically sorted stock DataFrame.
    """
    feature_candidate = [
               "GapPct",
               "GapDirection"
           ]
    df = df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    previous_close = df["close"].shift(1)
    gap = df["open"] - previous_close

    df["Gap"] = gap

    df["GapPct"] = safe_divide(
        gap,
        previous_close
    )

    df["GapDirection"] = np.sign(gap)

    df["GapUp"] = gap.gt(0).astype("int8")
    df["GapDown"] = gap.lt(0).astype("int8")

    return df

def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
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

    # Do not clip shadows until invalid OHLC rows are identified.
    invalid_ohlc = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df["high"] < df["low"])
    )

    flat_candle = (
        df["open"].eq(df["high"])
        & df["open"].eq(df["low"])
        & df["open"].eq(df["close"])
    )

    positive_volume = df["volume"].gt(0)
    zero_volume = df["volume"].eq(0)

    df["Body"] = body
    df["BodyPct"] = safe_divide(body, df["open"])

    df["Range"] = candle_range
    df["RangePct"] = safe_divide(
        candle_range,
        df["close"],
    )

    df["UpperShadow"] = upper_shadow
    df["LowerShadow"] = lower_shadow

    df["UpperShadowPct"] = safe_divide(
        upper_shadow,
        candle_range,
    )

    df["LowerShadowPct"] = safe_divide(
        lower_shadow,
        candle_range,
    )

    df["BodyToRange"] = safe_divide(
        body.abs(),
        candle_range,
    )

    df["Bullish"] = (
        df["close"] > df["open"]
    ).astype("int8")

    # Division and data-quality flags
    df["ZeroOpen"] = df["open"].eq(0).astype("int8")
    df["ZeroClose"] = df["close"].eq(0).astype("int8")
    df["ZeroRange"] = candle_range.eq(0).astype("int8")
    df["ZeroVolume"] = zero_volume.astype("int8")
    df["InvalidOHLC"] = invalid_ohlc.astype("int8")

    # Candle classification flags
    df["FlatCandle"] = flat_candle.astype("int8")

    df["StrangeFlatCandle"] = (
        flat_candle & positive_volume
    ).astype("int8")

    df["InactiveFlatCandle"] = (
        flat_candle & zero_volume
    ).astype("int8")

    return df

############ add log featues
def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds logarithmic return features.
     Direction
    ---------
     1 = Up
     0 = Flat
    -1 = Down

    Features
    --------
    LogReturn1
    LogReturn2
    LogReturn3
    LogReturn5
    LogReturn10
    This assumes each horizon has an independent binary class. 
    If every target has three classes such as Down, Neutral, and Up, 
    the output architecture should instead represent three horizons × three classes.
    My recommendation: start with target horizons 1, 3, and 10. 
    They provide clearly separated short-, medium-, and long-term objectives and avoid much of the redundancy between 1–2 and 3–5. 
    But confirm this by comparing validation/test performance against the full 1, 2, 3, 5, 10 target configuration.
    """
    feature_candidates = [
    "volume",
    "LogReturn1",
    "LogReturn2",
    "LogReturn3",
    "LogReturn5",
    "LogReturn10",
]
    df = df.copy()

    close = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df["LogReturn1"] = np.log(close / close.shift(1))
     # Direction of the current candle
    df["Direction"] = np.sign(df["LogReturn1"])

    df["LogReturn2"] = np.log(close / close.shift(2))

    df["LogReturn3"] = np.log(close / close.shift(3))

    df["LogReturn5"] = np.log(close / close.shift(5))

    df["LogReturn10"] = np.log(close / close.shift(10))

    return df

####### volumne

def add_volume_features(
    df: pd.DataFrame,
    window: int = 20 #WINDOW_SIZE
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
    
    df["VolumeChange"] = np.log1p(df["volume"]).diff()

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
    df["LogDollarVolume"] = np.log1p(df["DollarVolume"])
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
    features to be selected[
    "Volatility5",
    "Volatility10",
    "Volatility20",
    "TrueRangePct",
]
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
    ****** important
    for traning across multiple stock the best one is 
    [
     'EMA_Ratio', 
    'Trend5']

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
# Rolling Statistics
# ----------------------------------------------------------

def add_rolling_statistics(
    df: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:

    """
    Rolling statistical features.
    feature to be selected
    # Experiment A
[
    "RollingMean",
    "RollingStd",
    "RollingRange",
    "RollingSkew",
    "RollingKurtosis",
]

# Experiment B
[
    "RollingMean",
    "RollingStd",
    "RollingSkew",
    "RollingKurtosis",
]
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
    df["RollingRange"] = (
    df["RollingMax"] - df["RollingMin"]
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

    keep this features for training 
    ['CloseZ', 'VolumeZ', 'ReturnZ']
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
        return safe_divide((series-mean),std) 

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

def add_all_features(df):
     # group 1
    df=df.copy()
    df=add_log_returns(df)
    df=add_candle_features(df)
    df=add_gap_features(df)
    df=add_trend_features(df)
    #group 2
    df=add_volume_features(df)
    df=add_rolling_statistics(df)
    df=add_volatility_features(df)
    df=add_zscore_features(df)
    df = df.iloc[20:]
    print("Remaining rows:", len(df))
    print("Remaining NaN values:",df.isna().sum().sum(),)
    sorted_nan_counts = df.isna().sum().sort_values(ascending=False)
    print('columns with NaN: ',sorted_nan_counts[sorted_nan_counts > 0])
    ######## infiniy
    numeric_df = df.select_dtypes(include=np.number)

    number_of_inf = np.isinf(numeric_df.to_numpy(dtype=np.float64, na_value=np.nan)).sum()

    print(f"Number of infinity values: {number_of_inf}")
    numeric_df = df.select_dtypes(include=np.number)

    inf_mask = pd.DataFrame(
        np.isinf(
            numeric_df.to_numpy(
                dtype=np.float64,
                na_value=np.nan,
            )
        ),
        index=df.index,
        columns=numeric_df.columns,
    )

    inf_report = inf_mask.sum()
    inf_report = inf_report[inf_report > 0].sort_values(
        ascending=False
    )

    print(inf_report)
    # print(f'column with positive infinity: {df.columns[df.eq(np.inf).any()].tolist()}')
    # print("column with negative infinity: ", df.columns[df.eq(-np.inf).any()].tolist())

    feature_candidate=[##### from log feature
        #   "LogReturn1",
        
        #  "LogReturn3",
       
        #  "LogReturn10",
         ###### body
          "BodyPct","RangePct","UpperShadowPct","LowerShadowPct","BodyToRange",
                     "Bullish", 
                     # "ZeroVolume",
          ######### gap features
           "GapPct","GapDirection",
           ##### trending features
             'EMA_Ratio', 'PriceEMA20','EMA20' ,'Trend5',
             ####### volume
              "VolumeChange",
              "VolumeZ",
              #  "DollarVolume",
               'VolumeEMA20',
               'LogDollarVolume',
            ##### volatility
            "Volatility5",
            "Volatility10",
            "Volatility20",
            "TrueRangePct",
            ##### rolling 
             'RollingRange',
            "RollingMean",
             "RollingStd",
            "RollingSkew",
            "RollingKurtosis",
            ### z-score
            'CloseZ', 'VolumeZ', 'ReturnZ'        
         ]
    return df
def main():
    args = parse_arguments()
    symbol = args.symbol.strip().lower()
    bar_size = normalize_bar_size(args.bar_size)
        
    input_path = BASE_DIR / f"{symbol}_{bar_size}_{SESSION}.csv" if not  args.context else Path("../context") / f"{symbol}_{bar_size}_{SESSION}.csv"
    output_path = BASE_DIR / f"{symbol}_{bar_size}_{SESSION}_feng.csv" if not  args.context else Path("../context")/f"{symbol}_{bar_size}_{SESSION}_feng.csv"

    ######## read each ticker
    df = pd.read_csv(input_path,parse_dates=["datetime"],)
    df = (df.sort_values(by=["datetime"],ascending=[True],).reset_index(drop=True))
    ####### add features
    modified_df=add_all_features(df)
    #Remove first early candles because of nan
    modified_df = modified_df.iloc[20:]
    #### add ticker and group to dataframe
    modified_df = modified_df.assign(ticker=symbol,group='semiconductor',)
    modified_df.to_csv(f'{output_path}', index=False)
   
if __name__ == "__main__":
    main()

    

 





