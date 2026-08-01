from pathlib import Path

import pandas as pd


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
    "semiconductors": ["smh", "spy"]
}

bar_size = "1day"
session = "extended"

REQUIRED_COLUMNS = {
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

def add_features(
    df: pd.DataFrame,
    feature_prefix: str,
) -> pd.DataFrame:
    """Add calculated features using a predictable column prefix."""

    df = df.copy()

    df[f"{feature_prefix}_range_pct"] = (
        (df["high"] - df["low"]) / df["close"]
    )

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
    df = add_features(df, prefix)

    return instrument_name, df


# --------------------------------------------------
# 1. Load stock datasets
# --------------------------------------------------

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

stocks = pd.concat(
    stock_datasets.values(),
    ignore_index=True,
)


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
        ["ticker", "datetime"],
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
        f"{context_name}_range_pct",
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

    raise ValueError(
        "Some stock dates have no matching context data:\n"
        f"{problem_rows.to_string(index=False)}"
    )


# --------------------------------------------------
# 10. Inspect and save the completed dataset
# --------------------------------------------------

print("Combined columns:")
print(combined.columns.tolist())

print("First two combined rows:")
print(combined.head(2).to_string(index=False))

output_path = BASE_DIR / "combined_dataset.csv"
combined.to_csv(output_path, index=False)

print(f"Combined dataset saved to: {output_path}")