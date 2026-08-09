"""Repair missing OHLCV candles with causal forward filling.

The script audits one stock CSV at a time. For every date already present in
the CSV, it creates the expected intraday timestamps and inserts any absent
rows. Newly inserted rows are repaired as follows:

    open, high, low, close = the previous available value in each column
    volume                 = 0

No future candle is used. If the dataset begins with missing candles, those
leading rows remain unfilled because no previous observation exists.

The repaired CSV keeps the same columns as the input. Separate audit and run
history files make each preprocessing experiment reproducible without adding
an ``is_imputed`` feature to the model input.
"""

from argparse import ArgumentParser
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SESSION = "extended"

BAR_FREQUENCIES = {
    "1hour": "1h",
    "4hours": "4h",
    "30min": "30min",
    "15min": "15min",
    "1min": "1min",
}

BAR_SIZE_ALIASES = {
    "1hour": "1hour",
    "1 hour": "1hour",
    "1h": "1hour",
    "4hours": "4hours",
    "4 hours": "4hours",
    "4hour": "4hours",
    "4 hour": "4hours",
    "4h": "4hours",
    "30min": "30min",
    "30 min": "30min",
    "30m": "30min",
    "15min": "15min",
    "15 min": "15min",
    "15m": "15min",
    "1min": "1min",
    "1 min": "1min",
    "1m": "1min",
}

PRICE_VOLUME_COLUMNS = ["open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
REQUIRED_COLUMNS = {"datetime", *PRICE_VOLUME_COLUMNS}
IMPUTATION_METHOD = "column_forward_fill_volume_zero"


def parse_arguments():
    parser = ArgumentParser(
        description=(
            "Insert missing candles, forward-fill OHLC values, and set the "
            "inserted candles' volume to zero."
        )
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
        help=(
            "Supported values: 1hour, 4hours, 30min, 15min, 1min; "
            "default: 1hour."
        ),
    )
    parser.add_argument(
        "-c",
        "--context",
        action="store_true",
        help="Read and write CSV files in the ../context directory.",
    )
    parser.add_argument(
        "--experiment-label",
        default="ffill_volume_zero",
        help=(
            "Label saved with the run metrics; default: ffill_volume_zero."
        ),
    )
    return parser.parse_args()


def normalize_bar_size(value: str) -> str:
    normalized = " ".join(value.lower().strip().split())
    if normalized not in BAR_SIZE_ALIASES:
        raise ValueError(
            f"Unsupported bar size {value!r}. Use 1hour, 4hours, "
            "30min, 15min, or 1min."
        )
    return BAR_SIZE_ALIASES[normalized]


def file_sha256(file_path: Path) -> str:
    """Return a stable fingerprint so a model run can identify its dataset."""

    digest = sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stock(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path.resolve()}")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing columns: {sorted(missing_columns)}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)

    duplicate_mask = df["datetime"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_times = (
            df.loc[duplicate_mask, "datetime"].drop_duplicates().tolist()
        )
        raise ValueError(
            f"{file_path.name} contains duplicate datetimes: "
            f"{duplicate_times[:10]}"
        )

    for column in PRICE_VOLUME_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    partial_ohlcv_mask = df[PRICE_VOLUME_COLUMNS].isna().any(axis=1)
    if partial_ohlcv_mask.any():
        bad_times = df.loc[partial_ohlcv_mask, "datetime"].tolist()
        raise ValueError(
            "Existing rows with partially missing OHLCV values were found: "
            f"{bad_times[:10]}. This script inserts absent timestamps; it "
            "does not silently alter incomplete source rows."
        )

    return df.sort_values("datetime").reset_index(drop=True)


def build_expected_datetimes(
    df: pd.DataFrame,
    bar_size: str,
) -> pd.DatetimeIndex:
    """Build expected slots for every trading date observed in this CSV.

    This preserves the behavior of the previous script: it does not invent
    weekends, holidays, or a date that is completely absent from the file.
    For the extended session, expected bar start times run from 04:00 through
    19:00. With four-hour bars this produces 04:00, 08:00, 12:00, and 16:00.
    """

    observed_dates = (
        df["datetime"].dt.normalize().drop_duplicates().sort_values()
    )
    freq = BAR_FREQUENCIES[bar_size]

    datetimes = []
    for date in observed_dates:
        start_dt = date + pd.Timedelta(hours=4)
        end_dt = date + pd.Timedelta(hours=19)
        session_range = pd.date_range(start=start_dt, end=end_dt, freq=freq)
        datetimes.extend(session_range)

    return pd.DatetimeIndex(datetimes, name="datetime")


def find_consecutive_windows(positions: list[int]) -> list[list[int]]:
    if not positions:
        return []

    windows = [[positions[0]]]
    for position in positions[1:]:
        if position == windows[-1][-1] + 1:
            windows[-1].append(position)
        else:
            windows.append([position])
    return windows


def repair_stock(
    df: pd.DataFrame,
    expected_datetimes: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Insert missing rows and repair them without using future values."""

    original_columns = list(df.columns)
    observed_datetimes = pd.Index(df["datetime"])

    repaired = (
        df.set_index("datetime")
        .reindex(expected_datetimes)
        .reset_index()
    )

    # Membership in the original timestamp index identifies truly inserted
    # rows. This is safer than identifying them from values after filling.
    inserted_mask = ~repaired["datetime"].isin(observed_datetimes)
    inserted_positions = repaired.index[inserted_mask].tolist()
    windows = find_consecutive_windows(inserted_positions)

    gap_size_by_position: dict[int, int] = {}
    gap_position_by_position: dict[int, int] = {}
    for window in windows:
        for gap_position, row_position in enumerate(window, start=1):
            gap_size_by_position[row_position] = len(window)
            gap_position_by_position[row_position] = gap_position

    # Record the real source candle before filling. Every inserted bar in the
    # same gap points to the last originally observed candle, not to another
    # synthetic row.
    source_datetimes = repaired["datetime"].where(~inserted_mask).ffill()

    # This is the requested pandas forward-fill behavior: each OHLC column is
    # copied independently from the preceding row. Volume is intentionally not
    # forward-filled.
    repaired[PRICE_COLUMNS] = repaired[PRICE_COLUMNS].ffill()
    repaired.loc[inserted_mask, "volume"] = 0

    filled_mask = (
        inserted_mask
        & repaired[PRICE_COLUMNS].notna().all(axis=1)
        & repaired["volume"].notna()
    )
    unfilled_mask = inserted_mask & ~filled_mask

    # Preserve constant metadata columns, such as a symbol column, if present.
    # Variable feature/target columns are deliberately not imputed.
    extra_columns = [
        column
        for column in original_columns
        if column not in {"datetime", *PRICE_VOLUME_COLUMNS}
    ]
    for column in extra_columns:
        unique_values = df[column].dropna().unique()
        if len(unique_values) == 1:
            repaired.loc[inserted_mask, column] = unique_values[0]

    audit_rows = []
    for row_position in repaired.index[filled_mask]:
        audit_rows.append(
            {
                "datetime": repaired.loc[row_position, "datetime"],
                "method": IMPUTATION_METHOD,
                "source_datetime": source_datetimes.loc[row_position],
                "gap_size": gap_size_by_position[row_position],
                "position_in_gap": gap_position_by_position[row_position],
                "volume_assigned": 0,
            }
        )

    audit = pd.DataFrame(
        audit_rows,
        columns=[
            "datetime",
            "method",
            "source_datetime",
            "gap_size",
            "position_in_gap",
            "volume_assigned",
        ],
    )

    unfilled = pd.DataFrame(
        [
            {
                "datetime": repaired.loc[row_position, "datetime"],
                "reason": "no_previous_candle_for_forward_fill",
            }
            for row_position in repaired.index[unfilled_mask]
        ],
        columns=["datetime", "reason"],
    )

    # Remove only the unfillable leading timestamps. No observed source row is
    # removed because load_stock() has already validated its OHLCV values.
    repaired = repaired.dropna(subset=PRICE_VOLUME_COLUMNS).copy()

    rounded_volume = repaired["volume"].round()
    if repaired["volume"].eq(rounded_volume).all():
        repaired["volume"] = rounded_volume.astype("int64")

    repaired = repaired[
        [column for column in original_columns if column in repaired.columns]
    ]
    repaired = repaired.sort_values("datetime").reset_index(drop=True)

    gap_sizes = [len(window) for window in windows]
    repair_metrics = {
        "missing_rows_detected": int(inserted_mask.sum()),
        "rows_forward_filled": int(filled_mask.sum()),
        "rows_unfilled": int(unfilled_mask.sum()),
        "missing_gap_count": len(windows),
        "largest_missing_gap": max(gap_sizes, default=0),
    }
    return repaired, audit, unfilled, repair_metrics


def append_run_metrics(
    metrics_path: Path,
    run_metrics: dict,
) -> None:
    """Append one immutable summary row for every preprocessing run."""

    new_row = pd.DataFrame([run_metrics])
    if metrics_path.exists():
        previous_runs = pd.read_csv(metrics_path)
        history = pd.concat([previous_runs, new_row], ignore_index=True)
    else:
        history = new_row
    history.to_csv(metrics_path, index=False)


def main() -> None:
    args = parse_arguments()
    symbol = args.symbol.strip().lower()
    bar_size = normalize_bar_size(args.bar_size)
    experiment_label = args.experiment_label.strip()
    if not experiment_label:
        raise ValueError("--experiment-label cannot be empty.")

    data_directory = (
        BASE_DIR if not args.context else (BASE_DIR.parent / "context")
    )
    input_path = data_directory / f"{symbol}_{bar_size}_{SESSION}.csv"
    output_path = (
        data_directory / f"{symbol}_{bar_size}_{SESSION}_revisit.csv"
    )
    audit_path = (
        data_directory / f"{symbol}_{bar_size}_synthetic_audit.csv"
    )
    unfilled_path = (
        data_directory / f"{symbol}_{bar_size}_unfilled_audit.csv"
    )
    metrics_path = (
        data_directory / f"{symbol}_{bar_size}_experiment_metrics.csv"
    )

    run_started_at = datetime.now(UTC)
    run_id = run_started_at.strftime("%Y%m%dT%H%M%S%fZ")

    df = load_stock(input_path)
    input_hash = file_sha256(input_path)
    expected_datetimes = build_expected_datetimes(df, bar_size)
    repaired, audit, unfilled, repair_metrics = repair_stock(
        df,
        expected_datetimes,
    )

    # The run ID connects the detailed audit rows to the summary history row.
    audit.insert(0, "run_id", run_id)
    unfilled.insert(0, "run_id", run_id)

    repaired.to_csv(output_path, index=False)
    audit.to_csv(audit_path, index=False)
    unfilled.to_csv(unfilled_path, index=False)

    output_hash = file_sha256(output_path)
    inserted_percentage = (
        repair_metrics["rows_forward_filled"] / len(repaired) * 100
        if len(repaired)
        else 0.0
    )

    run_metrics = {
        "run_id": run_id,
        "run_started_at_utc": run_started_at.isoformat(),
        "experiment_label": experiment_label,
        "symbol": symbol.upper(),
        "bar_size": bar_size,
        "session": SESSION,
        "imputation_method": IMPUTATION_METHOD,
        "input_rows": len(df),
        "observed_trading_dates": df["datetime"].dt.date.nunique(),
        "expected_rows_on_observed_dates": len(expected_datetimes),
        "output_rows": len(repaired),
        **repair_metrics,
        "inserted_percentage_of_output": round(inserted_percentage, 6),
        "zero_volume_rows_in_output": int(repaired["volume"].eq(0).sum()),
        "input_sha256": input_hash,
        "output_sha256": output_hash,
        "input_file": str(input_path.resolve()),
        "output_file": str(output_path.resolve()),
    }
    append_run_metrics(metrics_path, run_metrics)

    print(f"Run ID: {run_id}")
    print(f"Experiment label: {experiment_label}")
    print(f"Symbol: {symbol.upper()}")
    print(f"Bar size: {bar_size}")
    print(f"Input rows: {len(df):,}")
    print(f"Observed trading dates: {df['datetime'].dt.date.nunique():,}")
    print(f"Expected rows on observed dates: {len(expected_datetimes):,}")
    print(
        "Candles forward-filled with volume 0: "
        f"{repair_metrics['rows_forward_filled']:,}"
    )
    print(f"Unfilled leading candles: {repair_metrics['rows_unfilled']:,}")
    print(f"Largest missing gap: {repair_metrics['largest_missing_gap']:,}")
    print(f"Completed dataset: {output_path}")
    print(f"Synthetic audit: {audit_path}")
    print(f"Unfilled audit: {unfilled_path}")
    print(f"Experiment metrics history: {metrics_path}")
    print(f"Output SHA-256: {output_hash}")


if __name__ == "__main__":
    main()