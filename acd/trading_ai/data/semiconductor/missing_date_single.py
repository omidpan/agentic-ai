"""Analyze missing and duplicate candles in one stock CSV file.

The CSV file belongs to one stock, so only a datetime column is required.
Ticker and group columns are not required.

Expected candle times:

1 hour:
    04:00, 05:00, ..., 19:00
    16 candles per observed trading date

4 hours:
    04:00, 08:00, 12:00, 16:00
    4 candles per observed trading date

When --context is provided, the input file is read from ../context/.
Otherwise, it is read from the current directory.

This script audits the data but does not insert missing candles.
"""

import argparse
from pathlib import Path

import pandas as pd


SESSION_START_HOUR = 4
SESSION_END_EXCLUSIVE_HOUR = 20

BAR_SIZE_ALIASES = {
    "1h": "1hour",
    "1hr": "1hour",
    "1hour": "1hour",
    "1hours": "1hour",
    "1 hour": "1hour",
    "1 hours": "1hour",

    "4h": "4hours",
    "4hr": "4hours",
    "4hour": "4hours",
    "4hours": "4hours",
    "4 hour": "4hours",
    "4 hours": "4hours",
}

BAR_SIZE_HOURS = {
    "1hour": 1,
    "4hours": 4,
}


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze missing and duplicate timestamps "
            "for one stock CSV file."
        )
    )

    parser.add_argument(
        "-s",
        "--symbol",
        type=str,
        default="NVDA",
        help="Stock symbol. Default: NVDA.",
    )

    parser.add_argument(
        "-bs",
        "--bar-size",
        dest="bar_size",
        type=str,
        default="1hour",
        help=(
            "Candle size. Supported values include "
            "1h, 1hour, 4h, and 4hours. Default: 1hour."
        ),
    )

    parser.add_argument(
        "-c",
        "--context",
        action="store_true",
        help="Read the input CSV from the ../context directory.",
    )

    return parser.parse_args()


def normalize_arguments(
    symbol: str,
    bar_size: str,
) -> tuple[str, str]:
    """Normalize and validate the symbol and bar-size arguments."""

    normalized_symbol = symbol.strip().lower()
    normalized_bar_size = bar_size.strip().lower()

    if not normalized_symbol:
        raise ValueError("Symbol cannot be empty.")

    if normalized_bar_size not in BAR_SIZE_ALIASES:
        supported_values = ", ".join(
            sorted(BAR_SIZE_ALIASES.keys())
        )

        raise ValueError(
            f"Unsupported bar size: {bar_size!r}. "
            f"Supported values: {supported_values}"
        )

    normalized_bar_size = BAR_SIZE_ALIASES[
        normalized_bar_size
    ]

    return normalized_symbol, normalized_bar_size


def resolve_input_file(
    symbol: str,
    bar_size: str,
    context: bool,
) -> Path:
    """Build and validate the input CSV path."""

    file_name = f"{symbol}_{bar_size}_extended.csv"

    if context:
        input_file = Path("../context") / file_name
    else:
        input_file = Path(file_name)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file.resolve()}"
        )

    return input_file


def load_and_validate_data(
    file_path: Path,
) -> tuple[pd.DataFrame, int]:
    """Load the CSV and validate its datetime column."""

    df = pd.read_csv(file_path)

    # Normalize all column names.
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    if "datetime" not in df.columns:
        raise ValueError(
            f"{file_path.name} must contain a "
            "'datetime' column."
        )

    # Invalid datetime values become NaT.
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    invalid_datetime_count = int(
        df["datetime"].isna().sum()
    )

    # Remove rows whose datetime could not be parsed.
    df = (
        df.dropna(subset=["datetime"])
        .copy()
    )

    if df.empty:
        raise ValueError(
            f"{file_path.name} contains no valid datetimes."
        )

    # Convert timezone-aware datetimes to timezone-naive values
    # while preserving the displayed market time.
    if df["datetime"].dt.tz is not None:
        df["datetime"] = (
            df["datetime"].dt.tz_localize(None)
        )

    df = (
        df.sort_values("datetime")
        .reset_index(drop=True)
    )

    return df, invalid_datetime_count


def find_duplicates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return all rows involved in duplicate datetimes."""

    duplicates = df[
        df.duplicated(
            subset=["datetime"],
            keep=False,
        )
    ]

    return (
        duplicates
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def build_expected_datetimes(
    df: pd.DataFrame,
    bar_size: str,
) -> pd.DatetimeIndex:
    """Create expected candle timestamps for observed dates.

    Any date containing at least one candle is considered an
    observed trading date and is audited against the complete
    expected session.
    """

    observed_dates = pd.DatetimeIndex(
        df["datetime"]
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )

    step_hours = BAR_SIZE_HOURS[bar_size]

    expected_hours = range(
        SESSION_START_HOUR,
        SESSION_END_EXCLUSIVE_HOUR,
        step_hours,
    )

    expected_datetimes = [
        date + pd.Timedelta(hours=hour)
        for date in observed_dates
        for hour in expected_hours
    ]

    return pd.DatetimeIndex(
        expected_datetimes,
        name="datetime",
    )


def find_missing_timestamps(
    df: pd.DataFrame,
    expected_datetimes: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Return expected timestamps missing from the CSV."""

    actual_datetimes = pd.DatetimeIndex(
        df["datetime"].drop_duplicates(),
        name="datetime",
    )

    missing_datetimes = expected_datetimes.difference(
        actual_datetimes
    )

    return (
        pd.DataFrame(
            {"datetime": missing_datetimes}
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def create_overall_summary(
    df: pd.DataFrame,
    expected_datetimes: pd.DatetimeIndex,
    missing_report: pd.DataFrame,
) -> pd.DataFrame:
    """Create the overall missing-candle summary."""

    unique_actual_datetimes = pd.DatetimeIndex(
        df["datetime"].drop_duplicates()
    )

    actual_session_candles = int(
        unique_actual_datetimes
        .isin(expected_datetimes)
        .sum()
    )

    expected_candles = len(expected_datetimes)
    missing_candles = len(missing_report)

    if expected_candles == 0:
        coverage_percent = 0.0
    else:
        coverage_percent = round(
            (
                actual_session_candles
                / expected_candles
            )
            * 100,
            2,
        )

    return pd.DataFrame(
        [
            {
                "first_datetime": df["datetime"].min(),
                "last_datetime": df["datetime"].max(),
                "observed_trading_dates": (
                    df["datetime"]
                    .dt.normalize()
                    .nunique()
                ),
                "expected_candles": expected_candles,
                "actual_unique_session_candles": (
                    actual_session_candles
                ),
                "missing_candles": missing_candles,
                "coverage_percent": coverage_percent,
            }
        ]
    )


def create_daily_summary(
    missing_report: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize missing candles for each observed date."""

    columns = [
        "date",
        "missing_candles",
        "missing_times",
    ]

    if missing_report.empty:
        return pd.DataFrame(columns=columns)

    daily_source = missing_report.copy()

    daily_source["date"] = (
        daily_source["datetime"].dt.date
    )

    daily_summary = (
        daily_source
        .groupby(
            "date",
            as_index=False,
        )
        .agg(
            missing_candles=(
                "datetime",
                "count",
            ),
            missing_times=(
                "datetime",
                lambda values: ", ".join(
                    values
                    .sort_values()
                    .dt.strftime("%H:%M")
                ),
            ),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily_summary


def save_reports(
    missing_report: pd.DataFrame,
    overall_summary: pd.DataFrame,
    daily_summary: pd.DataFrame,
    duplicates: pd.DataFrame,
    symbol: str,
    bar_size: str,
    context:bool
) -> list[Path]:
    """Save all audit reports in the current directory."""
    if not context:
        output_prefix = f"{symbol}_{bar_size}"
    else:
       output_prefix= Path("../context") / f"{symbol}_{bar_size}"

    missing_file = Path(
        f"{output_prefix}_missing_timestamps.csv"
    )

    overall_summary_file = Path(
        f"{output_prefix}_missing_summary.csv"
    )

    daily_summary_file = Path(
        f"{output_prefix}_missing_summary_by_date.csv"
    )

    duplicates_file = Path(
        f"{output_prefix}_duplicate_rows.csv"
    )

    # Files are always created, including empty reports.
    missing_report.to_csv(
        missing_file,
        index=False,
    )

    overall_summary.to_csv(
        overall_summary_file,
        index=False,
    )

    daily_summary.to_csv(
        daily_summary_file,
        index=False,
    )

    duplicates.to_csv(
        duplicates_file,
        index=False,
    )

    return [
        missing_file,
        overall_summary_file,
        daily_summary_file,
        duplicates_file,
    ]


def main() -> None:
    """Run the missing-date analysis."""

    args = parse_arguments()

    symbol, bar_size  = normalize_arguments(
        args.symbol,
        args.bar_size
    )

    input_file = resolve_input_file(
        symbol=symbol,
        bar_size=bar_size,
        context=args.context,
    )

    df, invalid_datetime_count = (
        load_and_validate_data(input_file)
    )

    duplicates = find_duplicates(df)

    expected_datetimes = build_expected_datetimes(
        df,
        bar_size,
    )

    missing_report = find_missing_timestamps(
        df,
        expected_datetimes,
    )

    overall_summary = create_overall_summary(
        df,
        expected_datetimes,
        missing_report,
    )

    daily_summary = create_daily_summary(
        missing_report
    )

    report_files = save_reports(
        missing_report=missing_report,
        overall_summary=overall_summary,
        daily_summary=daily_summary,
        duplicates=duplicates,
        symbol=symbol,
        bar_size=bar_size,
        context=args.context
    )

    print(f"Input file: {input_file.resolve()}")
    print(f"Input rows analyzed: {len(df):,}")

    print(
        "Observed trading dates: "
        f"{df['datetime'].dt.normalize().nunique():,}"
    )

    print(f"Bar size: {bar_size}")

    print(
        "Expected session candles: "
        f"{len(expected_datetimes):,}"
    )

    print(
        "Invalid datetime rows excluded: "
        f"{invalid_datetime_count:,}"
    )

    print(
        "Duplicate datetime rows: "
        f"{len(duplicates):,}"
    )

    print(
        f"Missing candles: {len(missing_report):,}"
    )

    print("\nSummary:")
    print(overall_summary.to_string(index=False))

    print("\nReports saved:")

    for report_file in report_files:
        print(f"  {report_file.resolve()}")


if __name__ == "__main__":
    main()