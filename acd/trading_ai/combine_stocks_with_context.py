"""Merge SPY and SMH features into eight stocks, then stack and encode them.

Processing order
----------------
1. Load and validate SPY and SMH context CSV files.
2. Load each single-stock feature CSV.
3. Require every stock datetime to exist exactly once in both contexts.
4. Merge SPY and SMH horizontally into each stock by ``datetime``.
5. Concatenate all merged stocks vertically.
6. One-hot encode the eight stock tickers.
7. Encode group as 0=semiconductor and 1=context.

The output keeps ``ticker`` as metadata because it is needed to build LSTM
sequences without crossing from one stock into another. The text ``group``
column is replaced by ``group_code``. Do not pass the text ``ticker`` column
directly to the model; use the generated ``ticker_*`` columns instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# All relative input/output paths are resolved from the directory containing
# this Python file, not from the terminal's current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent


DEFAULT_TICKERS = (
    "nvda",
    "amd",
    "avgo",
    "intc",
    "mrvl",
    "mu",
    "tsm",
    "amat",
)

DATETIME_COLUMN = "datetime"
LABEL_COLUMNS = {"ticker", "group"}
GROUP_CODES = {
    "semiconductor": 0,
    "context": 1,
}


def resolve_from_script_dir(path: Path) -> Path:
    """Return an absolute path, resolving relative paths beside this script."""

    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (SCRIPT_DIR / path).resolve()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge SPY and SMH horizontally into each stock by datetime, "
            "stack the stocks vertically, and one-hot encode stock tickers."
        )
    )
    parser.add_argument(
        "--stock-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing the eight single-stock feature CSV files. "
            "Relative paths are resolved from this script's directory."
        ),
    )
    parser.add_argument(
        "--stock-template",
        default="{ticker}_1day_extended_feng.csv",
        help=(
            "Stock filename template relative to --stock-dir. It must contain "
            "{ticker}. Default: {ticker}_1day_extended_feng.csv"
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_TICKERS),
        help="Stock tickers to combine. Default: the configured eight tickers.",
    )
    parser.add_argument(
        "--spy",
        type=Path,
        required=True,
        help=(
            "Path to the engineered SPY context CSV. Relative paths are "
            "resolved from this script's directory."
        ),
    )
    parser.add_argument(
        "--smh",
        type=Path,
        required=True,
        help=(
            "Path to the engineered SMH context CSV. Relative paths are "
            "resolved from this script's directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Path of the combined output CSV. Relative paths are resolved "
            "from this script's directory."
        ),
    )
    parser.add_argument(
        "--allow-extra-context-datetimes",
        action="store_true",
        help=(
            "Allow SPY/SMH to contain datetimes outside the union of stock "
            "datetimes. Stock datetimes must still have exact context matches."
        ),
    )
    return parser.parse_args()


def read_feature_csv(path: Path, expected_ticker: str) -> pd.DataFrame:
    """Read, normalize, sort, and validate one engineered feature CSV."""

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    # Accept the common accidental header typo shown in some exported files.
    if "datetime" in df.columns and DATETIME_COLUMN not in df.columns:
        df = df.rename(columns={"datetime": DATETIME_COLUMN})

    if DATETIME_COLUMN not in df.columns:
        raise ValueError(f"{path.name} has no '{DATETIME_COLUMN}' column.")

    df[DATETIME_COLUMN] = pd.to_datetime(
        df[DATETIME_COLUMN],
        errors="raise",
    )

    if df[DATETIME_COLUMN].isna().any():
        raise ValueError(f"{path.name} contains missing datetime values.")

    duplicate_mask = df[DATETIME_COLUMN].duplicated(keep=False)
    if duplicate_mask.any():
        examples = (
            df.loc[duplicate_mask, DATETIME_COLUMN]
            .drop_duplicates()
            .head(10)
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"{path.name} contains duplicate datetimes. Examples: {examples}"
        )

    if "ticker" in df.columns:
        observed = set(
            df["ticker"].dropna().astype(str).str.strip().str.lower().unique()
        )
        if observed and observed != {expected_ticker.lower()}:
            raise ValueError(
                f"{path.name} should contain ticker {expected_ticker!r}, "
                f"but contains {sorted(observed)}."
            )

    df["ticker"] = expected_ticker.lower()
    df["group"] = (
        "context"
        if expected_ticker.lower() in {"spy", "smh"}
        else "semiconductor"
    )

    return (
        df.sort_values(DATETIME_COLUMN)
        .reset_index(drop=True)
    )


def validate_no_missing_or_infinite(df: pd.DataFrame, name: str) -> None:
    """Reject missing cells and numeric positive/negative infinity."""

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts.gt(0)]
    if not missing_counts.empty:
        raise ValueError(
            f"{name} contains NaN values:\n{missing_counts.to_string()}"
        )

    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return

    values = numeric_df.to_numpy(dtype=np.float64, na_value=np.nan)
    inf_counts = pd.Series(
        np.isinf(values).sum(axis=0),
        index=numeric_df.columns,
    )
    inf_counts = inf_counts[inf_counts.gt(0)]
    if not inf_counts.empty:
        raise ValueError(
            f"{name} contains infinity values:\n{inf_counts.to_string()}"
        )


def prefix_context_features(
    context_df: pd.DataFrame,
    context_name: str,
) -> pd.DataFrame:
    """Prefix context features and replace text labels with numeric metadata."""

    context_name = context_name.lower()
    feature_columns = [
        column
        for column in context_df.columns
        if column not in {DATETIME_COLUMN, *LABEL_COLUMNS}
    ]

    renamed = context_df[[DATETIME_COLUMN, *feature_columns]].rename(
        columns={column: f"{context_name}_{column}" for column in feature_columns}
    )

    # The prefix already identifies which context instrument owns the features.
    # These numeric codes preserve the requested group information.
    renamed[f"{context_name}_group_code"] = GROUP_CODES["context"]
    return renamed


def require_exact_datetime_matches(
    stock_df: pd.DataFrame,
    context_df: pd.DataFrame,
    stock_name: str,
    context_name: str,
) -> None:
    """Require every stock datetime to exist in a context dataset."""

    missing = stock_df.loc[
        ~stock_df[DATETIME_COLUMN].isin(context_df[DATETIME_COLUMN]),
        DATETIME_COLUMN,
    ]

    if missing.empty:
        return

    examples = missing.head(10).astype(str).tolist()
    stock_has_time = stock_df[DATETIME_COLUMN].dt.time.nunique() > 1 or any(
        timestamp.time().isoformat() != "00:00:00"
        for timestamp in stock_df[DATETIME_COLUMN].head(100)
    )
    context_has_time = context_df[DATETIME_COLUMN].dt.time.nunique() > 1 or any(
        timestamp.time().isoformat() != "00:00:00"
        for timestamp in context_df[DATETIME_COLUMN].head(100)
    )

    time_hint = ""
    if stock_has_time != context_has_time:
        time_hint = (
            " One dataset appears date-only while the other includes a time; "
            "make their candle labels consistent before merging."
        )

    raise ValueError(
        f"{stock_name.upper()} has {len(missing)} datetimes with no exact "
        f"{context_name.upper()} match. Examples: {examples}.{time_hint}"
    )


def require_compatible_stock_columns(
    stock_frames: dict[str, pd.DataFrame],
) -> None:
    """Prevent vertical concatenation from introducing schema-related NaNs."""

    first_ticker = next(iter(stock_frames))
    expected = list(stock_frames[first_ticker].columns)
    expected_set = set(expected)

    errors: list[str] = []
    for ticker, frame in stock_frames.items():
        actual_set = set(frame.columns)
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        if missing or extra:
            errors.append(
                f"{ticker.upper()}: missing={missing}, extra={extra}"
            )

    if errors:
        raise ValueError(
            "Stock feature columns are not identical:\n" + "\n".join(errors)
        )


def merge_one_stock(
    stock_df: pd.DataFrame,
    contexts: dict[str, pd.DataFrame],
    ticker: str,
) -> pd.DataFrame:
    """Horizontally merge both context frames into one stock."""

    combined = stock_df.copy()

    for context_name, context_df in contexts.items():
        require_exact_datetime_matches(
            stock_df=stock_df,
            context_df=context_df,
            stock_name=ticker,
            context_name=context_name,
        )

        combined = combined.merge(
            prefix_context_features(context_df, context_name),
            on=DATETIME_COLUMN,
            how="left",
            validate="one_to_one",
        )

    combined["group_code"] = GROUP_CODES["semiconductor"]
    combined = combined.drop(columns="group")
    return combined


def validate_context_coverage(
    stock_frames: Iterable[pd.DataFrame],
    contexts: dict[str, pd.DataFrame],
    allow_extra: bool,
) -> None:
    """Optionally require context datetime sets to equal the stock union."""

    if allow_extra:
        return

    stock_datetime_union = pd.Index(
        pd.concat(
            [frame[DATETIME_COLUMN] for frame in stock_frames],
            ignore_index=True,
        ).unique()
    )

    for context_name, context_df in contexts.items():
        extra = context_df.loc[
            ~context_df[DATETIME_COLUMN].isin(stock_datetime_union),
            DATETIME_COLUMN,
        ]
        if not extra.empty:
            examples = extra.head(10).astype(str).tolist()
            raise ValueError(
                f"{context_name.upper()} contains {len(extra)} datetimes not "
                f"used by any stock. Examples: {examples}. Use "
                "--allow-extra-context-datetimes if this is intentional."
            )


def one_hot_encode_tickers(combined: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic int8 one-hot columns for the stock tickers."""

    ticker_categories = sorted(combined["ticker"].unique())
    categorical_ticker = pd.Categorical(
        combined["ticker"],
        categories=ticker_categories,
    )
    encoded = pd.get_dummies(
        categorical_ticker,
        prefix="ticker",
        dtype="int8",
    )
    encoded.index = combined.index
    return pd.concat([combined, encoded], axis=1)


def build_combined_dataset(
    stock_frames: dict[str, pd.DataFrame],
    contexts: dict[str, pd.DataFrame],
    allow_extra_context_datetimes: bool = False,
) -> pd.DataFrame:
    """Run the complete horizontal-then-vertical combination pipeline."""

    if not stock_frames:
        raise ValueError("No stock datasets were supplied.")
    if set(contexts) != {"spy", "smh"}:
        raise ValueError("Exactly two contexts named 'spy' and 'smh' are required.")

    require_compatible_stock_columns(stock_frames)
    validate_context_coverage(
        stock_frames.values(),
        contexts,
        allow_extra=allow_extra_context_datetimes,
    )

    merged_stocks = [
        merge_one_stock(stock_df, contexts, ticker)
        for ticker, stock_df in stock_frames.items()
    ]

    combined = pd.concat(merged_stocks, ignore_index=True)
    combined = one_hot_encode_tickers(combined)

    # Order the complete panel chronologically, then deterministically by ticker.
    # The training script groups by ticker before creating LSTM windows, so the
    # CSV itself does not need to keep each ticker's rows contiguous.
    combined = (
        combined.sort_values([DATETIME_COLUMN, "ticker"])
        .reset_index(drop=True)
    )

    validate_no_missing_or_infinite(combined, "combined output")
    return combined


def main() -> None:
    args = parse_arguments()

    # argparse converts the arguments to Path objects, but Path alone does not
    # choose a base directory. Resolve them explicitly relative to this file.
    stock_dir = resolve_from_script_dir(args.stock_dir)
    spy_path = resolve_from_script_dir(args.spy)
    smh_path = resolve_from_script_dir(args.smh)
    output_path = resolve_from_script_dir(args.output)

    tickers = [ticker.strip().lower() for ticker in args.tickers]
    if len(tickers) != len(set(tickers)):
        raise ValueError("--tickers contains duplicate names.")
    if "{ticker}" not in args.stock_template:
        raise ValueError("--stock-template must contain the text {ticker}.")

    stock_frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        stock_path = stock_dir / args.stock_template.format(ticker=ticker)
        stock_frames[ticker] = read_feature_csv(stock_path, ticker)
        validate_no_missing_or_infinite(stock_frames[ticker], ticker.upper())

    contexts = {
        "spy": read_feature_csv(spy_path, "spy"),
        "smh": read_feature_csv(smh_path, "smh"),
    }
    for name, frame in contexts.items():
        validate_no_missing_or_infinite(frame, name.upper())

    combined = build_combined_dataset(
        stock_frames,
        contexts,
        allow_extra_context_datetimes=args.allow_extra_context_datetimes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")

    ticker_columns = sorted(
        column for column in combined.columns if column.startswith("ticker_")
    )
    rows_per_ticker = combined.groupby("ticker", sort=True).size()

    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Stock directory: {stock_dir}")
    print(f"SPY file: {spy_path}")
    print(f"SMH file: {smh_path}")
    print(f"Saved: {output_path}")
    print(f"Rows: {len(combined):,}")
    print(f"Columns: {len(combined.columns):,}")
    print(f"One-hot ticker columns: {ticker_columns}")
    print("Group encoding: 0=semiconductor, 1=context")
    print("Rows per ticker:")
    print(rows_per_ticker.to_string())


if __name__ == "__main__":
    main()


'''
feature_columns =[
  "LogReturn1",
        
         "LogReturn3",
       
         "LogReturn10",
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
            'CloseZ', 'VolumeZ', 'ReturnZ',
 
 'spy_LogReturn1',
 'spy_Direction',
 'spy_LogReturn3',
  'spy_LogReturn10',
  
 'spy_BodyPct',
 'spy_RangePct',
 'spy_UpperShadowPct',
 'spy_LowerShadowPct',
 'spy_BodyToRange',
 'spy_Bullish',
  

 'spy_GapPct',
 'spy_GapDirection',

 'spy_EMA20',
 
 'spy_EMA_Ratio',
 'spy_PriceEMA20',
 'spy_Trend5',
 

 'spy_VolumeEMA20',
 'spy_VolumeChange',
 'spy_VolumeZ',
 'spy_LogDollarVolume',
  
 'spy_RollingMean',
 'spy_RollingStd',
 'spy_RollingRange',
 'spy_RollingSkew',
 'spy_RollingKurtosis',

 'spy_Volatility5',
 'spy_Volatility10',
 'spy_Volatility20',
 'spy_TrueRangePct',
  
 'spy_CloseZ',
 'spy_ReturnZ',
 
 
 'smh_volume',
 'smh_LogReturn1',
 'smh_Direction',
 
 'smh_LogReturn3',
 'smh_LogReturn10',
  
 'smh_BodyPct',

 'smh_RangePct',

 'smh_UpperShadowPct',
 'smh_LowerShadowPct',
 'smh_BodyToRange',
 'smh_Bullish',
  
 
 'smh_GapPct',
 'smh_GapDirection',
 
 'smh_EMA20',
 'smh_EMA_Ratio',
 'smh_PriceEMA20',

 'smh_Trend5',


 'smh_VolumeEMA20',

 'smh_VolumeChange',
 'smh_VolumeZ',
 'smh_VolumeEMARatio',

 'smh_LogDollarVolume',

  

 'smh_RollingMean',

 'smh_RollingStd',

 'smh_RollingRange',
 'smh_RollingSkew',
 'smh_RollingKurtosis',

 'smh_Volatility5',
 'smh_Volatility10',


 'smh_TrueRangePct',
 'smh_CloseZ',
 'smh_ReturnZ',
 ]
'''