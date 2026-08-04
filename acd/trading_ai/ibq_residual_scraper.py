"""Download only missing 1-hour IBKR candles listed in a daily CSV report.

Input CSV columns:
    date,ticker,group,missing_candles,missing_times

Example missing_times value:
    04:00, 05:00, 06:00

Each exact missing datetime is reconstructed from ``date + missing_times``.

The missing timestamps are grouped by ticker into request windows of at most
30 calendar days. Only the exact timestamps present in the input CSV are
retained. Output files use the requested spelling:
    resedualAMAT.csv, resedualAVGO.csv, ...

Resume support:
    Every successful ticker/window request is first saved as a temporary part
    CSV and recorded in a JSON checkpoint. If execution stops, rerun the same
    command and completed requests will be skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from ibapi.client import EClient
from ibapi.common import BarData
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

from utils.appenv import APPENV
from config import BASE_DIR

DEFAULT_INPUT = BASE_DIR / 'data' / 'semiconductor' / "semiconductor_1hour_missing_summary_by_date.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / 'data' / 'semiconductor' / Path("residual_data")


@dataclass(frozen=True)
class RequestJob:
    ticker: str
    group: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    missing_datetimes: frozenset[pd.Timestamp]

    @property
    def key(self) -> str:
        return (
            f"{self.ticker}|{self.start_date:%Y-%m-%d}|"
            f"{self.end_date:%Y-%m-%d}"
        )


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("._")


def write_json_atomic(data: dict[str, Any], path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def write_csv_atomic(data: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temp_path, index=False)
    os.replace(temp_path, path)


def load_missing_jobs(csv_path: Path) -> tuple[list[RequestJob], pd.DataFrame]:
    data = pd.read_csv(csv_path)
    required = {"date", "ticker", "group", "missing_candles", "missing_times"}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Input CSV is missing columns: {', '.join(sorted(missing_columns))}"
        )

    data = data.loc[:, ["date", "ticker", "group", "missing_candles", "missing_times"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    if data["date"].isna().any():
        bad_count = int(data["date"].isna().sum())
        raise ValueError(f"Input CSV contains {bad_count} invalid date value(s).")

    data["missing_candles"] = pd.to_numeric(
        data["missing_candles"], errors="coerce"
    )
    if data["missing_candles"].isna().any():
        raise ValueError("Input CSV contains invalid missing_candles value(s).")
    data["ticker"] = data["ticker"].astype(str).str.strip().str.upper()
    data["group"] = data["group"].astype(str).str.strip().str.lower()

    expanded_rows: list[dict[str, Any]] = []
    for row_number, row in data.iterrows():
        times = [
            value.strip()
            for value in str(row["missing_times"]).split(",")
            if value.strip()
        ]
        expected_count = int(row["missing_candles"])
        if len(times) != expected_count:
            raise ValueError(
                f"CSV row {row_number + 2} says {expected_count} missing candle(s), "
                f"but missing_times contains {len(times)} time(s)."
            )

        for missing_time in times:
            datetime_value = pd.to_datetime(
                f"{row['date']:%Y-%m-%d} {missing_time}",
                format="%Y-%m-%d %H:%M",
                errors="coerce",
            )
            if pd.isna(datetime_value):
                raise ValueError(
                    f"CSV row {row_number + 2} contains invalid time: {missing_time!r}."
                )
            expanded_rows.append(
                {
                    "datetime": datetime_value,
                    "ticker": row["ticker"],
                    "group": row["group"],
                }
            )

    expanded = pd.DataFrame(
        expanded_rows,
        columns=["datetime", "ticker", "group"],
    )
    if expanded.empty:
        raise ValueError("Input CSV contains no missing timestamps.")

    expanded = (
        expanded.drop_duplicates(["datetime", "ticker"])
        .sort_values(["ticker", "datetime"])
        .reset_index(drop=True)
    )
    expanded["request_date"] = expanded["datetime"].dt.normalize()

    jobs: list[RequestJob] = []
    for ticker, ticker_rows in expanded.groupby("ticker", sort=True):
        groups = ticker_rows["group"].unique()
        if len(groups) != 1:
            raise ValueError(f"{ticker} has multiple group values.")

        dates = sorted(ticker_rows["request_date"].unique())
        window_start = pd.Timestamp(dates[0])
        window_dates: list[pd.Timestamp] = []
        for raw_date in dates:
            request_date = pd.Timestamp(raw_date)
            if window_dates and request_date > window_start + timedelta(days=29):
                selected = ticker_rows[
                    ticker_rows["request_date"].isin(window_dates)
                ]
                jobs.append(
                    RequestJob(
                        ticker=ticker,
                        group=str(groups[0]),
                        start_date=window_dates[0],
                        end_date=window_dates[-1],
                        missing_datetimes=frozenset(selected["datetime"].tolist()),
                    )
                )
                window_start = request_date
                window_dates = []
            window_dates.append(request_date)

        if window_dates:
            selected = ticker_rows[ticker_rows["request_date"].isin(window_dates)]
            jobs.append(
                RequestJob(
                    ticker=ticker,
                    group=str(groups[0]),
                    start_date=window_dates[0],
                    end_date=window_dates[-1],
                    missing_datetimes=frozenset(selected["datetime"].tolist()),
                )
            )

    return jobs, expanded.drop(columns="request_date")


class IBResidualClient(EWrapper, EClient, APPENV):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        APPENV.__init__(self)
        self.connected_event = threading.Event()
        self.request_event = threading.Event()
        self.rows: list[dict[str, Any]] = []
        self.request_error: str | None = None
        self.active_request_id: int | None = None
        self.request_id = int(time.time()) % 1_000_000

    def nextValidId(self, orderId: int) -> None:  # noqa: N802
        print(f"IB connected. Next valid order ID: {orderId}")
        self.connected_event.set()

    def error(
        self,
        reqId: int,
        errorCode: int,
        errorString: str,
        advancedOrderRejectJson: str = "",
    ) -> None:
        if errorCode in {2104, 2106, 2107, 2108, 2158}:
            print(f"IB INFO {errorCode}: {errorString}")
            return
        print(f"IB ERROR request={reqId} code={errorCode}: {errorString}")
        if reqId == self.active_request_id:
            self.request_error = f"{errorCode}: {errorString}"
            self.request_event.set()

    def historicalData(self, reqId: int, bar: BarData) -> None:  # noqa: N802
        if reqId != self.active_request_id:
            return
        self.rows.append(
            {
                "datetime": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": float(bar.volume),
            }
        )

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:  # noqa: N802
        if reqId == self.active_request_id:
            self.request_event.set()

    def connect_ib(self) -> None:
        print(f"Connecting IB {self.host}:{self.port}")
        self.connect(self.host, self.port, self.client_id)
        threading.Thread(target=self.run, daemon=True, name="ib-api-loop").start()
        if not self.connected_event.wait(timeout=10):
            self.disconnect()
            raise ConnectionError("IB API handshake failed")
        print("IB API ready")

    @staticmethod
    def stock_contract(ticker: str, exchange: str, currency: str) -> Contract:
        contract = Contract()
        contract.symbol = ticker
        contract.secType = "STK"
        contract.exchange = exchange
        contract.currency = currency
        return contract

    def download_date(
        self,
        job: RequestJob,
        *,
        timezone: str,
        exchange: str,
        currency: str,
        what_to_show: str,
        timeout: int,
    ) -> pd.DataFrame:
        self.request_id += 1
        self.active_request_id = self.request_id
        self.rows = []
        self.request_error = None
        self.request_event.clear()

        # Asking through midnight after the last date covers all extended-hour
        # bars in this request window.
        next_date = job.end_date + timedelta(days=1)
        duration_days = (job.end_date - job.start_date).days + 1
        end_datetime = f"{next_date:%Y%m%d} 00:00:00 {timezone}"
        print(
            f"[{job.ticker}] {job.start_date:%Y-%m-%d} to "
            f"{job.end_date:%Y-%m-%d}: requesting "
            f"{len(job.missing_datetimes)} missing candle(s)"
        )

        self.reqHistoricalData(
            reqId=self.active_request_id,
            contract=self.stock_contract(job.ticker, exchange, currency),
            endDateTime=end_datetime,
            durationStr=f"{duration_days} D",
            barSizeSetting="1 hour",
            whatToShow=what_to_show,
            useRTH=0,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[],
        )

        if not self.request_event.wait(timeout=timeout):
            self.cancelHistoricalData(self.active_request_id)
            raise TimeoutError(
                f"{job.key}: request timed out after {timeout}s"
            )
        if self.request_error:
            raise RuntimeError(
                f"{job.key}: {self.request_error}"
            )

        columns = ["datetime", "ticker", "group", "open", "high", "low", "close", "volume"]
        if not self.rows:
            return pd.DataFrame(columns=columns)

        result = pd.DataFrame(self.rows)
        result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
        result = result[result["datetime"].isin(job.missing_datetimes)].copy()
        result["ticker"] = job.ticker
        result["group"] = job.group
        return (
            result.loc[:, columns]
            .drop_duplicates(["datetime", "ticker"])
            .sort_values("datetime")
            .reset_index(drop=True)
        )


def load_checkpoint(path: Path, input_path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "input_csv": str(input_path.resolve()),
            "completed_requests": [],
        }
    with path.open("r", encoding="utf-8") as stream:
        checkpoint = json.load(stream)
    if checkpoint.get("input_csv") != str(input_path.resolve()):
        raise ValueError(
            f"Checkpoint belongs to another input file. Remove {path} to restart."
        )
    return checkpoint


def finalize_ticker(
    ticker: str,
    jobs: list[RequestJob],
    parts_dir: Path,
    output_dir: Path,
) -> Path:
    frames: list[pd.DataFrame] = []
    for job in jobs:
        part_path = parts_dir / (
            f"{safe_name(job.ticker)}_{job.start_date:%Y%m%d}_"
            f"{job.end_date:%Y%m%d}.csv"
        )
        if part_path.exists():
            frames.append(pd.read_csv(part_path, parse_dates=["datetime"]))

    columns = ["datetime", "ticker", "group", "open", "high", "low", "close", "volume"]
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    if not result.empty:
        result = (
            result.drop_duplicates(["datetime", "ticker"])
            .sort_values(["datetime", "ticker", "group"])
            .reset_index(drop=True)
        )

    output_path = output_dir / f"resedual{ticker}.csv"
    write_csv_atomic(result.loc[:, columns], output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download missing 1-hour candles from IBKR using a CSV report."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timezone", default="US/Eastern")
    parser.add_argument("--exchange", default="SMART")
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--what-to-show", default="TRADES")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--pause",
        type=float,
        default=1.5,
        help="Seconds between IBKR historical requests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / ".residual_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "residual_checkpoint.json"

    jobs, missing_rows = load_missing_jobs(input_path)
    checkpoint = load_checkpoint(checkpoint_path, input_path)
    completed = set(checkpoint.get("completed_requests", []))
    print(
        f"Loaded {len(missing_rows):,} missing timestamps for "
        f"{missing_rows['ticker'].nunique()} ticker(s), grouped into {len(jobs)} requests."
    )
    print(f"Resume state: {len(completed)} request(s) already completed.")

    app = IBResidualClient()
    failures: list[str] = []
    try:
        pending_jobs = [job for job in jobs if job.key not in completed]
        if pending_jobs:
            app.connect_ib()
        for index, job in enumerate(pending_jobs):
            try:
                result = app.download_date(
                    job,
                    timezone=args.timezone,
                    exchange=args.exchange,
                    currency=args.currency,
                    what_to_show=args.what_to_show,
                    timeout=args.timeout,
                )
                part_path = parts_dir / (
                    f"{safe_name(job.ticker)}_{job.start_date:%Y%m%d}_"
                    f"{job.end_date:%Y%m%d}.csv"
                )
                write_csv_atomic(result, part_path)
                completed.add(job.key)
                checkpoint["completed_requests"] = sorted(completed)
                write_json_atomic(checkpoint, checkpoint_path)
                print(f"[{job.key}] saved {len(result)} matching candle(s)")
            except Exception as exc:
                failures.append(job.key)
                print(f"[{job.key}] FAILED: {exc}")
            if index < len(pending_jobs) - 1:
                time.sleep(max(0.0, args.pause))
    finally:
        if app.isConnected():
            app.disconnect()

    jobs_by_ticker: dict[str, list[RequestJob]] = {}
    for job in jobs:
        jobs_by_ticker.setdefault(job.ticker, []).append(job)
    for ticker, ticker_jobs in jobs_by_ticker.items():
        path = finalize_ticker(ticker, ticker_jobs, parts_dir, output_dir)
        print(f"[{ticker}] finalized -> {path}")

    requested = set(zip(missing_rows["ticker"], missing_rows["datetime"]))
    recovered: set[tuple[str, pd.Timestamp]] = set()
    for ticker in jobs_by_ticker:
        path = output_dir / f"resedual{ticker}.csv"
        data = pd.read_csv(path, parse_dates=["datetime"])
        recovered.update(zip(data["ticker"], data["datetime"]))
    still_missing = requested.difference(recovered)
    print(
        f"Finished: recovered {len(recovered):,} of {len(requested):,} requested candles; "
        f"{len(still_missing):,} were not returned by IBKR."
    )
    if failures:
        print(f"Failed requests: {len(failures)}. Rerun the script to retry them.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
