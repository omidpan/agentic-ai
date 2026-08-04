"""Create missing-date and duplicate reports for merged 1-hour stock data.

Expected input columns: datetime, ticker, group.
Expected timestamps are inferred from the union of timestamps found across all
tickers, avoiding false gaps for nights, weekends, and market holidays.
"""

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("./combined_semiconductor_1hour.csv")
OUTPUT_DIR = Path(".")

MISSING_TIMESTAMPS_FILE = "semiconductor_1hour_missing_timestamps.csv"
TICKER_SUMMARY_FILE = "semiconductor_1hour_missing_summary_by_ticker.csv"
DAILY_SUMMARY_FILE = "semiconductor_1hour_missing_summary_by_date.csv"
DUPLICATES_FILE = "semiconductor_1hour_duplicate_rows.csv"


def load_and_validate_data(file_path: Path) -> tuple[pd.DataFrame, int]:
    """Load the merged CSV and validate its required identifying columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    df = pd.read_csv(file_path)
    required_columns = {"datetime", "ticker", "group"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    invalid_datetime_count = int(df["datetime"].isna().sum())

    df = df.dropna(subset=["datetime", "ticker"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["group"] = df["group"].astype(str).str.lower().str.strip()

    # Normalize timezone-aware datetimes to timezone-naive values.
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)

    return df, invalid_datetime_count


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return every row involved in a duplicate ticker-datetime pair."""
    return (
        df[df.duplicated(["ticker", "datetime"], keep=False)]
        .sort_values(["datetime", "ticker", "group"])
        .reset_index(drop=True)
    )


def find_missing_timestamps(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Index]:
    """Find ticker-hour combinations absent from the merged data."""
    expected_datetimes = pd.Index(
        df["datetime"].drop_duplicates().sort_values(),
        name="datetime",
    )
    tickers = sorted(df["ticker"].unique())

    expected_index = pd.MultiIndex.from_product(
        [expected_datetimes, tickers],
        names=["datetime", "ticker"],
    )
    actual_index = pd.MultiIndex.from_frame(
        df[["datetime", "ticker"]].drop_duplicates()
    )

    missing_report = expected_index.difference(actual_index).to_frame(index=False)

    group_by_ticker = (
        df[["ticker", "group"]]
        .drop_duplicates("ticker")
        .set_index("ticker")["group"]
    )
    missing_report["group"] = missing_report["ticker"].map(group_by_ticker)

    missing_report = (
        missing_report[["datetime", "ticker", "group"]]
        .sort_values(["datetime", "ticker", "group"])
        .reset_index(drop=True)
    )
    return missing_report, expected_datetimes


def create_ticker_summary(
    df: pd.DataFrame,
    missing_report: pd.DataFrame,
    expected_datetimes: pd.Index,
) -> pd.DataFrame:
    """Summarize expected, actual, and missing observations by ticker."""
    ticker_summary = (
        df.groupby(["ticker", "group"], as_index=False)
        .agg(
            first_datetime=("datetime", "min"),
            last_datetime=("datetime", "max"),
            actual_rows=("datetime", "nunique"),
        )
    )

    missing_counts = (
        missing_report.groupby("ticker")
        .size()
        .rename("missing_hours")
        .reset_index()
    )
    ticker_summary = ticker_summary.merge(missing_counts, on="ticker", how="left")
    ticker_summary["missing_hours"] = (
        ticker_summary["missing_hours"].fillna(0).astype(int)
    )
    ticker_summary["expected_hours"] = len(expected_datetimes)
    ticker_summary["coverage_percent"] = (
        ticker_summary["actual_rows"]
        .div(ticker_summary["expected_hours"])
        .mul(100)
        .round(2)
    )

    return ticker_summary[
        [
            "ticker",
            "group",
            "first_datetime",
            "last_datetime",
            "expected_hours",
            "actual_rows",
            "missing_hours",
            "coverage_percent",
        ]
    ].sort_values(["missing_hours", "ticker"], ascending=[False, True])


def create_daily_summary(missing_report: pd.DataFrame) -> pd.DataFrame:
    """Group missing hourly candles by date, ticker, and group."""
    columns = ["date", "ticker", "group", "missing_candles", "missing_times"]
    if missing_report.empty:
        return pd.DataFrame(columns=columns)

    daily_source = missing_report.copy()
    daily_source["date"] = daily_source["datetime"].dt.date

    return (
        daily_source.groupby(["date", "ticker", "group"], as_index=False)
        .agg(
            missing_candles=("datetime", "count"),
            missing_times=(
                "datetime",
                lambda values: ", ".join(values.dt.strftime("%H:%M")),
            ),
        )
        .sort_values(["date", "ticker", "group"])
        .reset_index(drop=True)
    )


def save_reports(
    missing_report: pd.DataFrame,
    ticker_summary: pd.DataFrame,
    daily_summary: pd.DataFrame,
    duplicates: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save all reports as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_report.to_csv(output_dir / MISSING_TIMESTAMPS_FILE, index=False)
    ticker_summary.to_csv(output_dir / TICKER_SUMMARY_FILE, index=False)
    daily_summary.to_csv(output_dir / DAILY_SUMMARY_FILE, index=False)

    # Always create the duplicate report, even when it contains zero rows.
    duplicates.to_csv(output_dir / DUPLICATES_FILE, index=False)


def main() -> None:
    df, invalid_datetime_count = load_and_validate_data(INPUT_FILE)
    duplicates = find_duplicates(df)
    missing_report, expected_datetimes = find_missing_timestamps(df)
    ticker_summary = create_ticker_summary(
        df,
        missing_report,
        expected_datetimes,
    )
    daily_summary = create_daily_summary(missing_report)

    save_reports(
        missing_report,
        ticker_summary,
        daily_summary,
        duplicates,
        OUTPUT_DIR,
    )

    print(f"Input rows analyzed: {len(df):,}")
    print(f"Tickers analyzed: {df['ticker'].nunique():,}")
    print(f"Expected market timestamps: {len(expected_datetimes):,}")
    print(f"Invalid datetime rows excluded: {invalid_datetime_count:,}")
    print(f"Duplicate ticker-datetime rows: {len(duplicates):,}")
    print(f"Missing ticker-hour observations: {len(missing_report):,}")
    print("\nMissing-data summary by ticker:")
    print(ticker_summary.to_string(index=False))
    print(f"\nReports saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
