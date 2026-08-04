"""Repair missing OHLCV candles in one stock CSV.

The input file is selected with --symbol and --bar-size. Only dates already
present in that stock file are audited; the script does not compare or merge
different tickers.
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SESSION = "extended"
BAR_SCHEDULES = {
    "1hour": [*range(4, 20)],
    "4hours": [4, 8, 12, 16],
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


def average_candle_values(
    left_candle: pd.Series,
    right_candle: pd.Series,
) -> pd.Series:
    """Average OHLCV values and round each result before it is reused."""

    averaged = (
        left_candle[PRICE_VOLUME_COLUMNS].astype(float)
        + right_candle[PRICE_VOLUME_COLUMNS].astype(float)
    ) / 2.0
    averaged[PRICE_COLUMNS] = averaged[PRICE_COLUMNS].round(2)
    averaged["volume"] = round(float(averaged["volume"]))
    return averaged


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

    return df.sort_values("datetime").reset_index(drop=True)


def build_expected_datetimes(
    df: pd.DataFrame,
    bar_size: str,
) -> pd.DatetimeIndex:
    """Build the full session for every date observed in this one CSV."""

    observed_dates = (
        df["datetime"].dt.normalize().drop_duplicates().sort_values()
    )
    hours = BAR_SCHEDULES[bar_size]

    return pd.DatetimeIndex(
        [
            date + pd.Timedelta(hours=hour)
            for date in observed_dates
            for hour in hours
        ],
        name="datetime",
    )


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


def fill_window_middle_out(
    repaired: pd.DataFrame,
    left_position: int,
    right_position: int,
    synthetic_positions: list[int],
) -> None:
    """Fill an interior gap: middle, right, left, right, left, ..."""

    first_missing = left_position + 1
    last_missing = right_position - 1
    middle = (first_missing + last_missing) // 2

    repaired.loc[middle, PRICE_VOLUME_COLUMNS] = average_candle_values(
        repaired.loc[left_position],
        repaired.loc[right_position],
    )
    synthetic_positions.append(middle)

    next_right = middle + 1
    next_left = middle - 1
    nearest_right = middle
    nearest_left = middle

    while next_right <= last_missing or next_left >= first_missing:
        if next_right <= last_missing:
            repaired.loc[next_right, PRICE_VOLUME_COLUMNS] = (
                average_candle_values(
                    repaired.loc[nearest_right],
                    repaired.loc[right_position],
                )
            )
            synthetic_positions.append(next_right)
            nearest_right = next_right
            next_right += 1

        if next_left >= first_missing:
            repaired.loc[next_left, PRICE_VOLUME_COLUMNS] = (
                average_candle_values(
                    repaired.loc[left_position],
                    repaired.loc[nearest_left],
                )
            )
            synthetic_positions.append(next_left)
            nearest_left = next_left
            next_left -= 1


def fill_leading_window(
    repaired: pd.DataFrame,
    window: list[int],
    synthetic_positions: list[int],
) -> bool:
    """Fill a dataset-head gap backward using two future candles."""

    first_future = window[-1] + 1
    second_future = first_future + 1
    if (
        second_future >= len(repaired)
        or not repaired.loc[
            [first_future, second_future], PRICE_VOLUME_COLUMNS
        ].notna().all().all()
    ):
        return False

    nearer_future = first_future
    farther_future = second_future
    for position in reversed(window):
        repaired.loc[position, PRICE_VOLUME_COLUMNS] = average_candle_values(
            repaired.loc[nearer_future], repaired.loc[farther_future]
        )
        synthetic_positions.append(position)
        farther_future = nearer_future
        nearer_future = position
    return True


def fill_trailing_window(
    repaired: pd.DataFrame,
    window: list[int],
    synthetic_positions: list[int],
) -> bool:
    """Fill a dataset-tail gap forward using two previous candles."""

    nearest_previous = window[0] - 1
    second_previous = nearest_previous - 1
    if (
        second_previous < 0
        or not repaired.loc[
            [second_previous, nearest_previous], PRICE_VOLUME_COLUMNS
        ].notna().all().all()
    ):
        return False

    farther_previous = second_previous
    nearer_previous = nearest_previous
    for position in window:
        repaired.loc[position, PRICE_VOLUME_COLUMNS] = average_candle_values(
            repaired.loc[farther_previous], repaired.loc[nearer_previous]
        )
        synthetic_positions.append(position)
        farther_previous = nearer_previous
        nearer_previous = position
    return True


def repair_stock(
    df: pd.DataFrame,
    expected_datetimes: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repair all fillable missing windows in this one stock dataset."""

    original_columns = list(df.columns)
    repaired = df.set_index("datetime").reindex(expected_datetimes).reset_index()

    missing_mask = repaired[PRICE_VOLUME_COLUMNS].isna().all(axis=1)
    partial_mask = (
        repaired[PRICE_VOLUME_COLUMNS].isna().any(axis=1) & ~missing_mask
    )
    if partial_mask.any():
        bad_times = repaired.loc[partial_mask, "datetime"].tolist()
        raise ValueError(
            f"Partially missing OHLCV rows found: {bad_times[:10]}"
        )

    missing_positions = repaired.index[missing_mask].tolist()
    windows = find_consecutive_windows(missing_positions)
    synthetic_positions: list[int] = []
    unfilled_rows: list[dict] = []

    for window in windows:
        left_position = window[0] - 1
        right_position = window[-1] + 1

        if left_position < 0:
            filled = fill_leading_window(
                repaired, window, synthetic_positions
            )
            reason = "missing_two_future_candles"
        elif right_position >= len(repaired):
            filled = fill_trailing_window(
                repaired, window, synthetic_positions
            )
            reason = "missing_two_previous_candles"
        else:
            safe_boundaries = (
                repaired.loc[left_position, PRICE_VOLUME_COLUMNS]
                .notna()
                .all()
                and repaired.loc[right_position, PRICE_VOLUME_COLUMNS]
                .notna()
                .all()
            )
            if safe_boundaries:
                fill_window_middle_out(
                    repaired,
                    left_position,
                    right_position,
                    synthetic_positions,
                )
                filled = True
            else:
                filled = False
                reason = "missing_boundary_candle"

        if not filled:
            unfilled_rows.extend(
                {"datetime": repaired.loc[position, "datetime"], "reason": reason}
                for position in window
            )

    synthetic_times = sorted(
        repaired.loc[synthetic_positions, "datetime"].drop_duplicates()
    )
    audit = pd.DataFrame(
        {
            "datetime": synthetic_times,
            "method": "recursive_boundary_average",
        }
    )
    unfilled = pd.DataFrame(
        unfilled_rows,
        columns=["datetime", "reason"],
    )

    repaired = repaired.dropna(subset=PRICE_VOLUME_COLUMNS).copy()
    repaired[PRICE_COLUMNS] = repaired[PRICE_COLUMNS].round(2)
    repaired["volume"] = repaired["volume"].round().astype("int64")

    # Preserve every original non-OHLCV column without creating ticker/group.
    repaired = repaired[[column for column in original_columns if column in repaired]]
    repaired = repaired.sort_values("datetime").reset_index(drop=True)
    return repaired, audit, unfilled


def main() -> None:
    args = parse_arguments()
    symbol = args.symbol.strip().lower()
    bar_size = normalize_bar_size(args.bar_size)
    
    input_path = BASE_DIR / f"{symbol}_{bar_size}_{SESSION}.csv" if not  args.context else Path("../context") / f"{symbol}_{bar_size}_{SESSION}.csv"
    output_path = BASE_DIR / f"{symbol}_{bar_size}_{SESSION}.csv" if not  args.context else Path("../context")/f"{symbol}_{bar_size}_{SESSION}.csv"
    audit_path = BASE_DIR / f"{symbol}_{bar_size}_synthetic_audit.csv" if not  args.context else Path("../context")/f"{symbol}_{bar_size}_synthetic_audit.csv"
    unfilled_path = BASE_DIR / f"{symbol}_{bar_size}_unfilled_audit.csv" if not  args.context else Path("../context")/f"{symbol}_{bar_size}_unfilled_audit.csv"

    df = load_stock(input_path)
    expected_datetimes = build_expected_datetimes(df, bar_size)
    repaired, audit, unfilled = repair_stock(df, expected_datetimes)

    repaired.to_csv(output_path, index=False)
    audit.to_csv(audit_path, index=False)
    unfilled.to_csv(unfilled_path, index=False)

    print(f"Symbol: {symbol.upper()}")
    print(f"Bar size: {bar_size}")
    print(f"Input rows: {len(df):,}")
    print(f"Observed trading dates: {df['datetime'].dt.date.nunique():,}")
    print(f"Expected rows on observed dates: {len(expected_datetimes):,}")
    print(f"Synthetic candles inserted: {len(audit):,}")
    print(f"Unfilled candles: {len(unfilled):,}")
    print(f"Completed dataset: {output_path}")
    print(f"Synthetic audit: {audit_path}")
    print(f"Unfilled audit: {unfilled_path}")


if __name__ == "__main__":
    main()