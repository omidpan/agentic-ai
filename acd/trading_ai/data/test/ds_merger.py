import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

dataset_names = {
    "semiconductors": ["nvda", "amd"]
}

dataset_context = {
    "semiconductors": ["smh"],
    "broad_market": ["spy"]
}

REQUIRED_COLUMNS = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume"
}


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

def add_features(
    df: pd.DataFrame,
    dataset_name: str
) -> pd.DataFrame:
    """Create instrument-specific features."""

    df = df.copy()

    df[f"{dataset_name}_range_pct"] = (
        (df["high"] - df["low"]) / df["close"]
    )

    return df


# --------------------------------------------------
# Load and validate one CSV
# --------------------------------------------------

def load_dataset(
    base_dir: str,
    name: str
) -> tuple[str, pd.DataFrame]:

    file_path = Path(base_dir) / f"{name}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    dataset_name = file_path.stem.lower()

    df = pd.read_csv(
        file_path,
        parse_dates=["date"]
    )

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name}.csv is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if df["date"].duplicated().any():
        duplicate_dates = df.loc[
            df["date"].duplicated(),
            "date"
        ].tolist()

        raise ValueError(
            f"{dataset_name}.csv contains duplicate dates: "
            f"{duplicate_dates}"
        )

    if (df["close"] == 0).any():
        raise ValueError(
            f"{dataset_name}.csv contains a zero close price."
        )

    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    df = add_features(df, dataset_name)

    return dataset_name, df


# --------------------------------------------------
# 1. Load stock datasets
# --------------------------------------------------

stock_datasets = {}

for group, ticker_names in dataset_names.items():

    for ticker_name in ticker_names:

        dataset_name, df = load_dataset(
            BASE_DIR,
            ticker_name
        )

        df["ticker"] = dataset_name.upper()
        df["group"] = group

        stock_datasets[dataset_name] = df


# --------------------------------------------------
# 2. Combine stocks vertically
# --------------------------------------------------

stocks = pd.concat(
    stock_datasets.values(),
    ignore_index=True
)


# --------------------------------------------------
# 3. Rename stock OHLCV columns
# --------------------------------------------------
# All stock rows use the same stock_* columns because
# NVDA and AMD are stacked vertically.

stocks = stocks.rename(columns={
    "open": "stock_open",
    "high": "stock_high",
    "low": "stock_low",
    "close": "stock_close",
    "volume": "stock_volume"
})


# Rename each ticker-specific range column into one
# shared stock_range_pct column.

range_columns = {
    f"{ticker_name}_range_pct": "stock_range_pct"
    for ticker_names in dataset_names.values()
    for ticker_name in ticker_names
}

stocks = stocks.rename(columns=range_columns)

# After concatenation, multiple columns may have been
# renamed to stock_range_pct. Combine them into one.
duplicate_range_columns = stocks.loc[
    :,
    stocks.columns == "stock_range_pct"
]

if duplicate_range_columns.shape[1] > 1:
    stocks = stocks.drop(
        columns="stock_range_pct"
    )

    stocks["stock_range_pct"] = (
        duplicate_range_columns.bfill(axis=1).iloc[:, 0]
    )


# --------------------------------------------------
# 4. Create ticker IDs for an embedding layer
# --------------------------------------------------

ticker_names = sorted(stocks["ticker"].unique())

ticker_to_id = {
    ticker: ticker_id
    for ticker_id, ticker in enumerate(ticker_names)
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
    dtype="int8"
)


# --------------------------------------------------
# 6. Load context datasets
# --------------------------------------------------

context_datasets = {}

for context_type, context_names in dataset_context.items():

    for context_name in context_names:

        dataset_name, df = load_dataset(
            BASE_DIR,
            context_name
        )

        # Prefix raw context columns so they do not
        # conflict with stock or other context columns.
        df = df.rename(columns={
            "open": f"{dataset_name}_open",
            "high": f"{dataset_name}_high",
            "low": f"{dataset_name}_low",
            "close": f"{dataset_name}_close",
            "volume": f"{dataset_name}_volume"
        })

        context_datasets[dataset_name] = df


# --------------------------------------------------
# 7. Merge contexts horizontally by date
# --------------------------------------------------

combined = stocks.copy()

for context_name, context_df in context_datasets.items():

    combined = combined.merge(
        context_df,
        on="date",
        how="left",
        validate="many_to_one"
    )


# --------------------------------------------------
# 8. Sort each stock time series chronologically
# --------------------------------------------------

combined = (
    combined
    .sort_values(
        ["ticker", "date"],
        ascending=[True, True]
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
        f"{context_name}_range_pct"
    ]
]

missing_context = combined[
    context_feature_columns
].isna().any(axis=1)

if missing_context.any():
    problem_rows = combined.loc[
        missing_context,
        ["date", "ticker"]
    ]

    raise ValueError(
        "Some stock dates have no matching context data:\n"
        f"{problem_rows}"
    )
print(combined.columns)
print(combined.head())
