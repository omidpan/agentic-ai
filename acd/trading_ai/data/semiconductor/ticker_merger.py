"""Repair missing extended-session 1-hour candles, then merge stock CSVs.

Each observed trading date has exactly 16 hourly candle labels from 04:00
through 19:00. No 20:00 or overnight timestamps are expected. Missing windows
are repaired only
when possible from the nearest available candles for the same ticker.
The 04:00 and 19:00 boundaries may use the preceding session's 19:00 candle or
the following session's 04:00 candle, respectively.

For a missing window, the middle missing candle is filled first using the two
known boundaries. The algorithm then moves outward, alternating right and
left. Every new candle is averaged from the nearest available candle toward
the middle and the original boundary on that side.

There are two dataset-edge exceptions. A leading gap is filled backward using
the first two future candles; a trailing gap is filled forward using the last
two previous candles. Each newly generated candle is rounded before it is
reused in the next step.
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BAR_SIZE = "1hour"
SESSION = "extended"
GROUP_NAME = "semiconductor"
SESSION_START_HOUR = 4
SESSION_END_HOUR = 19
SESSION_CANDLE_COUNT = SESSION_END_HOUR - SESSION_START_HOUR + 1

# TICKERS = ["nvda", "amd", "avgo", "intc", "mrvl", "mu", "tsm", "amat"]
TICKERS = ["nvda"]
PRICE_VOLUME_COLUMNS = ["open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
REQUIRED_COLUMNS = {"datetime", *PRICE_VOLUME_COLUMNS}


def average_candle_values(
    left_candle: pd.Series,
    right_candle: pd.Series,
) -> pd.Series:
    """Average every OHLCV value using the required rounding rules.

    Prices are rounded to two decimal places and volume to an integer at every
    insertion. This matters because a generated candle can become a boundary
    for the next recursive middle-out calculation.
    """

    averaged = (
        left_candle[PRICE_VOLUME_COLUMNS].astype(float)
        + right_candle[PRICE_VOLUME_COLUMNS].astype(float)
    ) / 2.0
    averaged[PRICE_COLUMNS] = averaged[PRICE_COLUMNS].round(2)
    averaged["volume"] = round(float(averaged["volume"]))
    return averaged


def load_stock(ticker: str) -> pd.DataFrame:
    """Load and validate one stock CSV without changing its observations."""

    ticker = ticker.strip().lower()
    file_path = BASE_DIR / f"{ticker}_{BAR_SIZE}_{SESSION}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing columns: {sorted(missing_columns)}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
    df = df.sort_values("datetime").reset_index(drop=True)

    duplicate_mask = df["datetime"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_datetimes = (
            df.loc[duplicate_mask, "datetime"].drop_duplicates().tolist()
        )
        raise ValueError(
            f"{file_path.name} contains duplicate datetimes: "
            f"{duplicate_datetimes}"
        )

    for column in PRICE_VOLUME_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    return df


def find_consecutive_windows(
    repaired: pd.DataFrame,
    positions: list[int],
) -> list[list[int]]:
    """Split missing positions into consecutive expected-session windows.

    Position adjacency is used intentionally. In the expected calendar,
    19:00 on one trading date is followed by 04:00 on the next trading date;
    there are no overnight candle positions between them.
    """

    if not positions:
        return []

    windows = [[positions[0]]]
    for position in positions[1:]:
        previous_position = windows[-1][-1]
        if position == previous_position + 1:
            windows[-1].append(position)
        else:
            windows.append([position])
    return windows


def fill_window_recursively(
    repaired: pd.DataFrame,
    left_position: int,
    right_position: int,
    synthetic_positions: list[int],
) -> None:
    """Fill a gap middle-out, alternating right then left.

    For an odd gap there is one middle position. For an even gap the
    left-middle position is selected first, making the result deterministic.
    """

    if right_position - left_position <= 1:
        return

    first_missing = left_position + 1
    last_missing = right_position - 1
    middle_position = (first_missing + last_missing) // 2
    repaired.loc[middle_position, PRICE_VOLUME_COLUMNS] = (
        average_candle_values(
            repaired.loc[left_position],
            repaired.loc[right_position],
        )
    )
    synthetic_positions.append(middle_position)

    next_right = middle_position + 1
    next_left = middle_position - 1
    nearest_right = middle_position
    nearest_left = middle_position

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
    """Fill a dataset-head gap backward using two future candles.

    Example: if 04:00--08:00 are missing and 09:00 plus 10:00 exist, create
    08:00 from the average of 09:00 and 10:00. Then create 07:00 from 08:00
    and 09:00, continuing backward until 04:00.
    """

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
    """Fill a dataset-tail gap forward using two previous candles.

    Example: if 18:00--19:00 are missing and 16:00 plus 17:00 exist, create
    18:00 from the average of 16:00 and 17:00. Then create 19:00 from 17:00
    and the newly generated 18:00 candle.
    """

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
    ticker: str,
    expected_datetimes: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[pd.Timestamp, str]]]:
    """Insert synthetic candles for all bounded missing windows of one stock."""

    ticker = ticker.upper()
    # Every configured ticker receives the same expected trading calendar.
    # This intentionally repairs a whole absent day for one ticker whenever
    # that date is present for at least one of the other configured tickers.
    repaired = df.set_index("datetime").reindex(expected_datetimes)
    repaired.index.name = "datetime"
    repaired = repaired.reset_index()

    missing_mask = repaired[PRICE_VOLUME_COLUMNS].isna().all(axis=1)
    partial_mask = repaired[PRICE_VOLUME_COLUMNS].isna().any(axis=1) & ~missing_mask
    if partial_mask.any():
        bad_times = repaired.loc[partial_mask, "datetime"].tolist()
        raise ValueError(
            f"{ticker} has partially missing OHLCV rows: {bad_times[:10]}"
        )

    missing_dates = repaired.loc[missing_mask, "datetime"].dt.normalize()
    missing_count_by_date = missing_dates.value_counts()
    fully_missing_dates = set(
        missing_count_by_date[
            missing_count_by_date == SESSION_CANDLE_COUNT
        ].index
    )

    synthetic_positions: list[int] = []
    unfilled_datetimes: list[tuple[pd.Timestamp, str]] = []

    # A completely absent ticker-day is still valid when other tickers traded.
    # Fill each of its 16 hourly labels using the same hour from the closest
    # available trading day before and after it.
    for missing_date in sorted(fully_missing_dates):
        day_positions = repaired.index[
            repaired["datetime"].dt.normalize().eq(missing_date)
        ].tolist()
        first_position = day_positions[0]
        last_position = day_positions[-1]
        previous_day_start = first_position - SESSION_CANDLE_COUNT
        next_day_start = last_position + 1

        has_adjacent_sessions = (
            previous_day_start >= 0
            and next_day_start + SESSION_CANDLE_COUNT <= len(repaired)
            and repaired.loc[
                previous_day_start:first_position - 1,
                PRICE_VOLUME_COLUMNS,
            ].notna().all().all()
            and repaired.loc[
                next_day_start:next_day_start + SESSION_CANDLE_COUNT - 1,
                PRICE_VOLUME_COLUMNS,
            ].notna().all().all()
        )

        if not has_adjacent_sessions:
            unfilled_datetimes.extend(
                (repaired.loc[position, "datetime"], "missing_adjacent_day")
                for position in day_positions
            )
            continue

        for offset, position in enumerate(day_positions):
            previous_position = previous_day_start + offset
            next_position = next_day_start + offset
            repaired.loc[position, PRICE_VOLUME_COLUMNS] = (
                average_candle_values(
                    repaired.loc[previous_position],
                    repaired.loc[next_position],
                )
            )
            synthetic_positions.append(position)

    # Recalculate missing positions after full-day repair, then interpolate all
    # remaining partial gaps using the closest existing candles on both sides.
    missing_mask = repaired[PRICE_VOLUME_COLUMNS].isna().all(axis=1)
    missing_positions = repaired.index[missing_mask].tolist()
    windows = find_consecutive_windows(repaired, missing_positions)

    for window in windows:
        left_position = window[0] - 1
        right_position = window[-1] + 1

        # Exception 1: the missing window is at the beginning of this ticker's
        # ordered expected dataset. Work backward from two future candles.
        if left_position < 0:
            if not fill_leading_window(
                repaired, window, synthetic_positions
            ):
                unfilled_datetimes.extend(
                    (value, "missing_two_future_candles")
                    for value in repaired.loc[window, "datetime"].tolist()
                )
            continue

        # Exception 2: the missing window is at the end. Work forward from two
        # previous candles, reusing every newly generated candle.
        if right_position >= len(repaired):
            if not fill_trailing_window(
                repaired, window, synthetic_positions
            ):
                unfilled_datetimes.extend(
                    (value, "missing_two_previous_candles")
                    for value in repaired.loc[window, "datetime"].tolist()
                )
            continue

        has_safe_boundaries = (
            repaired.loc[
                left_position, PRICE_VOLUME_COLUMNS
            ].notna().all()
            and repaired.loc[
                right_position, PRICE_VOLUME_COLUMNS
            ].notna().all()
        )

        # Cross-session boundaries are valid: 19:00 -> next session's 04:00.
        if not has_safe_boundaries:
            unfilled_datetimes.extend(
                (value, "missing_boundary_candle")
                for value in repaired.loc[window, "datetime"].tolist()
            )
            continue

        fill_window_recursively(
            repaired,
            left_position,
            right_position,
            synthetic_positions,
        )

    # Only dataset-edge gaps without two safe boundaries remain unfilled.
    repaired = repaired.dropna(subset=PRICE_VOLUME_COLUMNS).copy()
    repaired["ticker"] = ticker
    repaired["group"] = GROUP_NAME

    synthetic_datetime_set = set(
        repaired.loc[synthetic_positions, "datetime"].tolist()
    )
    audit = pd.DataFrame(
        [
            {
                "datetime": value,
                "ticker": ticker,
                "group": GROUP_NAME,
                "method": "recursive_boundary_average",
            }
            for value in sorted(synthetic_datetime_set)
        ],
        columns=["datetime", "ticker", "group", "method"],
    )

    return repaired, audit, unfilled_datetimes


def merge_semiconductor_stocks() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, repair, vertically merge, and chronologically sort all stocks."""

    raw_stocks = {ticker: load_stock(ticker) for ticker in TICKERS}

    # A date is a trading date when at least one configured ticker has data.
    # For each such date, expect only the extended-session candle labels
    # 04:00, 05:00, ..., 19:00. Never generate 20:00 or overnight rows.
    trading_dates = pd.DatetimeIndex(
        sorted(
            set().union(
                *(
                    set(df["datetime"].dt.normalize())
                    for df in raw_stocks.values()
                )
            )
        )
    )
    expected_datetimes = pd.DatetimeIndex(
        [
            trading_date + pd.Timedelta(hours=hour)
            for trading_date in trading_dates
            for hour in range(SESSION_START_HOUR, SESSION_END_HOUR + 1)
        ]
    )

    repaired_stocks = []
    audits = []
    unfilled_rows = []

    for ticker, df in raw_stocks.items():
        repaired, audit, unfilled = repair_stock(
            df, ticker, expected_datetimes
        )
        repaired_stocks.append(repaired)
        audits.append(audit)
        unfilled_rows.extend(
            {
                "datetime": value,
                "ticker": ticker.upper(),
                "group": GROUP_NAME,
                "reason": reason,
            }
            for value, reason in unfilled
        )

    combined = pd.concat(repaired_stocks, ignore_index=True)
    combined = combined.sort_values(
        ["datetime", "ticker", "group"]
    ).reset_index(drop=True)

    identifier_columns = ["datetime", "ticker", "group"]
    remaining_columns = [
        column for column in combined.columns if column not in identifier_columns
    ]
    combined = combined[identifier_columns + remaining_columns]

    audit_report = pd.concat(audits, ignore_index=True)
    if not audit_report.empty:
        audit_report = audit_report.sort_values(
            ["datetime", "ticker"]
        ).reset_index(drop=True)

    unfilled_report = pd.DataFrame(
        unfilled_rows,
        columns=["datetime", "ticker", "group", "reason"],
    )

    return combined, audit_report, unfilled_report


if __name__ == "__main__":
    combined_dataset, synthetic_audit, unfilled_audit = (
        merge_semiconductor_stocks()
    )

    output_path = BASE_DIR / f"nvda_no_missing_{BAR_SIZE}.csv"
    audit_path = BASE_DIR / "synthetic_candles_audit.csv"
    unfilled_path = BASE_DIR / "unfilled_missing_candles.csv"

    combined_dataset.to_csv(output_path, index=False)
    synthetic_audit.to_csv(audit_path, index=False)
    unfilled_audit.to_csv(unfilled_path, index=False)

    print(f"Merged tickers: {', '.join(t.upper() for t in TICKERS)}")
    print(f"Final rows: {len(combined_dataset):,}")
    print(f"Synthetic candles inserted: {len(synthetic_audit):,}")
    print(f"Unsafe gaps left unfilled: {len(unfilled_audit):,}")
    print(f"Merged dataset: {output_path}")
    print(f"Synthetic audit: {audit_path}")
    print(f"Unfilled audit: {unfilled_path}")