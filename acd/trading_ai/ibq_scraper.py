"""Download IBKR historical bars for one symbol or a YAML-configured batch.
Single-symbol compatibility:
    python ibq_scraper.py -s NVDA -bs "1 day" -d "5 Y" -o false
Batch mode:
    python ibq_scraper.py --config scraper-config-<groupName>.yml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from ibapi.client import EClient
from ibapi.common import BarData
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

import config as app_config
from utils.appenv import APPENV


# Prefer BASE_DIR. Fall back to the parent of the older DATA_DIR setting.
if hasattr(app_config, "BASE_DIR"):
    BASE_DIR = Path(app_config.BASE_DIR)
elif hasattr(app_config, "DATA_DIR"):
    BASE_DIR = Path(app_config.DATA_DIR).parent
else:
    BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DownloadJob:
    symbol: str
    group: str
    duration: str
    bar_size: str
    session: str = "rth"
    what_to_show: str = "TRADES"
    exchange: str | None = None
    primary_exchange: str | None = None
    security_type: str = "STK"
    currency: str = "USD"


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    row_count: int
    resumed: bool = False


def parse_bool(value: Any) -> bool:
    """Parse booleans safely; argparse's type=bool treats 'False' as True."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def parse_sessions(settings: dict[str, Any]) -> list[str]:
    """Return canonical session names from YAML or the legacy overnight flag."""
    raw_sessions = settings.get("sessions")
    if raw_sessions is None:
        # Backward compatibility with the original configuration/CLI model.
        return ["overnight" if parse_bool(settings.get("overnight", False)) else "rth"]
    if isinstance(raw_sessions, str):
        raw_sessions = [raw_sessions]
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError("'sessions' must be a non-empty string or list.")

    aliases = {
        "rth": "rth",
        "regular": "rth",
        "regular_hours": "rth",
        "extended": "extended",
        "extended_hours": "extended",
        "pre_post": "extended",
        "pre-market": "extended",
        "post-market": "extended",
        "overnight": "overnight",
    }
    sessions: list[str] = []
    for value in raw_sessions:
        normalized = str(value).strip().lower().replace(" ", "_")
        if normalized not in aliases:
            raise ValueError(
                f"Unsupported session {value!r}; use rth, extended, or overnight."
            )
        session = aliases[normalized]
        if session not in sessions:
            sessions.append(session)
    return sessions


def load_jobs(config_path: Path) -> tuple[list[DownloadJob], int, Path]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    defaults = config.get("defaults", {})
    groups = config.get("groups", {})
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Config must contain a non-empty 'groups' mapping.")

    jobs: list[DownloadJob] = []
    seen: set[tuple[str, str, str]] = set()
    for group, group_config in groups.items():
        if isinstance(group_config, list):
            group_config = {"symbols": group_config}
        if not isinstance(group_config, dict):
            raise ValueError(f"Group {group!r} must be a list or mapping.")

        merged = {**defaults, **group_config}
        symbols = merged.get("symbols", [])
        if not symbols:
            raise ValueError(f"Group {group!r} has no symbols.")

        for symbol_entry in symbols:
            overrides = symbol_entry if isinstance(symbol_entry, dict) else {}
            symbol = overrides.get("symbol") if overrides else symbol_entry
            if not symbol:
                raise ValueError(f"Invalid symbol entry in group {group!r}.")
            settings = {**merged, **overrides}
            for session in parse_sessions(settings):
                key = (str(group).lower(), str(symbol).upper(), session)
                if key in seen:
                    raise ValueError(
                        f"Duplicate job: group={group}, symbol={symbol}, session={session}"
                    )
                seen.add(key)
                jobs.append(
                    DownloadJob(
                        symbol=str(symbol).upper(),
                        group=str(group).lower(),
                        duration=str(settings.get("duration", "365 D")),
                        bar_size=str(settings.get("bar_size", "1 day")),
                        session=session,
                        what_to_show=str(settings.get("what_to_show", "TRADES")),
                        exchange=settings.get("exchange"),
                        primary_exchange=settings.get("primary_exchange"),
                        security_type=str(settings.get("security_type", "STK")),
                        currency=str(settings.get("currency", "USD")),
                    )
                )

    max_workers = int(config.get("max_workers", 3))
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    output_root = Path(config.get("output_root", Path(BASE_DIR) / "data"))
    if not output_root.is_absolute():
        output_root = Path(BASE_DIR) / output_root
    return jobs, max_workers, output_root


class IBClient(EWrapper, EClient, APPENV):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        APPENV.__init__(self)
        self.connected_event = threading.Event()
        self.historical_data: dict[int, list[dict[str, Any]]] = {}
        self.request_events: dict[int, threading.Event] = {}
        self.request_errors: dict[int, str] = {}
        self.earliest_date: dict[int, str | None] = {}
        self._state_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._request_id = int(time.time()) % 1_000_000

    def nextValidId(self, orderId: int) -> None:  # noqa: N802 (IB callback)
        print(f"IB connected. Next valid order ID: {orderId}")
        self.connected_event.set()

    def error(
        self,
        reqId: int,
        errorCode: int,
        errorString: str,
        advancedOrderRejectJson: str = "",
    ) -> None:
        # Common informational farm/connectivity messages do not fail requests.
        if errorCode in {2104, 2106, 2107, 2108, 2158}:
            print(f"IB INFO {errorCode}: {errorString}")
            return
        print(f"IB ERROR request={reqId} code={errorCode}: {errorString}")
        with self._state_lock:
            event = self.request_events.get(reqId)
            if reqId >= 0 and event is not None:
                self.request_errors[reqId] = f"{errorCode}: {errorString}"
                event.set()

    def connect_ib(self) -> None:
        print(f"Connecting IB {self.host}:{self.port}")
        self.connect(self.host, self.port, self.client_id)
        threading.Thread(target=self.run, daemon=True, name="ib-api-loop").start()
        if not self.connected_event.wait(timeout=10):
            self.disconnect()
            raise ConnectionError("IB API handshake failed")
        print("IB API ready")

    def next_request_id(self) -> int:
        with self._state_lock:
            request_id = self._request_id
            self._request_id += 1
            return request_id

    @staticmethod
    def stock_contract(job: DownloadJob) -> Contract:
        contract = Contract()
        contract.symbol = job.symbol
        contract.secType = job.security_type
        contract.currency = job.currency

        if job.exchange:
            contract.exchange = job.exchange
        elif job.symbol in {"BIT", "BITCOIN"}:
            contract.exchange = "PAXOS"
        elif job.session == "overnight":
            contract.exchange = "OVERNIGHT"
        else:
            contract.exchange = "SMART"

        if job.primary_exchange:
            contract.primaryExchange = job.primary_exchange
        elif job.symbol == "VIX":
            contract.primaryExchange = "CBOE"
        return contract

    def historicalData(self, reqId: int, bar: BarData) -> None:  # noqa: N802
        with self._state_lock:
            if reqId not in self.historical_data:
                return
            self.historical_data[reqId].append(
                {
                    "datetime": bar.date,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": float(bar.volume),
                }
            )
            if self.earliest_date[reqId] is None:
                self.earliest_date[reqId] = bar.date

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:  # noqa: N802
        with self._state_lock:
            event = self.request_events.get(reqId)
        if event:
            event.set()
    @staticmethod
    def _normalize_bar_size(bar_size: str) -> str:
        """Return an IBKR-compatible bar-size string.

        YAML/CLI input is intentionally forgiving, but IBKR receives only its
        canonical spelling (for example, ``1 min`` rather than ``1 mins``).
        """
        value = " ".join(str(bar_size).strip().lower().split())

        aliases = {
            "1 sec": "1 secs",
            "1 second": "1 secs",
            "1 seconds": "1 secs",
            "5 sec": "5 secs",
            "5 second": "5 secs",
            "5 seconds": "5 secs",
            "10 sec": "10 secs",
            "10 second": "10 secs",
            "10 seconds": "10 secs",
            "15 sec": "15 secs",
            "15 second": "15 secs",
            "15 seconds": "15 secs",
            "30 sec": "30 secs",
            "30 second": "30 secs",
            "30 seconds": "30 secs",
            "1 mins": "1 min",
            "1 minute": "1 min",
            "1 minutes": "1 min",
            "2 min": "2 mins",
            "2 minute": "2 mins",
            "2 minutes": "2 mins",
            "3 min": "3 mins",
            "3 minute": "3 mins",
            "3 minutes": "3 mins",
            "5 min": "5 mins",
            "5 minute": "5 mins",
            "5 minutes": "5 mins",
            "10 min": "10 mins",
            "10 minute": "10 mins",
            "10 minutes": "10 mins",
            "15 min": "15 mins",
            "15 minute": "15 mins",
            "15 minutes": "15 mins",
            "20 min": "20 mins",
            "20 minute": "20 mins",
            "20 minutes": "20 mins",
            "30 min": "30 mins",
            "30 minute": "30 mins",
            "30 minutes": "30 mins",
            "1 hours": "1 hour",
            "1 day": "1 day",
            "1 days": "1 day",
            "1 week": "1 week",
            "1 weeks": "1 week",
            "1 month": "1 month",
            "1 months": "1 month",
        }

        return aliases.get(value, value)

    @classmethod
    def _max_chunk_days(cls, bar_size: str) -> int:
        normalized = cls._normalize_bar_size(bar_size)

        chunk_sizes = {
            "1 secs": 1,
            "5 secs": 1,
            "10 secs": 1,
            "15 secs": 1,
            "30 secs": 1,
            "1 min": 1,
            "2 mins": 2,
            "3 mins": 7,
            "5 mins": 7,
            "10 mins": 7,
            "15 mins": 7,
            "20 mins": 7,
            "30 mins": 30,
            "1 hour": 30,
            "2 hours": 30,
            "3 hours": 30,
            "4 hours": 30,
            "8 hours": 30,
            "1 day": 365,
            "1 week": 365,
            "1 month": 365,
        }

        if normalized not in chunk_sizes:
            raise ValueError(
                f"Unsupported bar size: {bar_size!r} "
                f"(normalized as {normalized!r})"
            )

        return chunk_sizes[normalized]
    @staticmethod
    def _parse_duration_to_days(duration: str) -> int:
        match = re.fullmatch(r"\s*(\d+)\s*([DWMY])\s*", duration.upper())
        if not match:
            raise ValueError(f"Unsupported duration {duration!r}; use forms such as '30 D' or '5 Y'.")
        value, unit = int(match.group(1)), match.group(2)
        return value * {"D": 1, "W": 7, "M": 30, "Y": 365}[unit]

    def get_historical_data(
        self,
        job: DownloadJob,
        output_root: Path,
        timeout: int = 120,
    ) -> DownloadResult:
        paths = download_paths(job, output_root)
        paths["group_dir"].mkdir(parents=True, exist_ok=True)
        paths["parts_dir"].mkdir(parents=True, exist_ok=True)

        requested_days = self._parse_duration_to_days(job.duration)
        normalized_bar_size = self._normalize_bar_size(job.bar_size)
        max_chunk_days = self._max_chunk_days(normalized_bar_size)

        expected_job = checkpoint_job(job, normalized_bar_size)
        checkpoint = load_checkpoint(paths["checkpoint"])
        resumed = checkpoint is not None

        if checkpoint:
            if checkpoint.get("job") != expected_job:
                raise ValueError(
                    f"{job.symbol}: existing checkpoint does not match this job. "
                    f"Remove {paths['checkpoint']} and {paths['parts_dir']} to restart."
                )
            if checkpoint.get("completed") and paths["output"].exists():
                row_count = int(checkpoint.get("row_count", 0))
                print(f"[{job.symbol}] already complete -> {paths['output']}")
                return DownloadResult(paths["output"], row_count, resumed=True)
            days_remaining = int(checkpoint["days_remaining"])
            end_date_anchor = str(checkpoint["next_end_datetime"])
            chunk_number = int(checkpoint["next_chunk_number"])
            print(
                f"[{job.symbol}] resuming at chunk {chunk_number}; "
                f"{days_remaining} requested days remain"
            )
        else:
            days_remaining = requested_days
            end_date_anchor = ""
            chunk_number = 0
            save_checkpoint(
                paths["checkpoint"],
                {
                    "version": 1,
                    "job": expected_job,
                    "days_remaining": days_remaining,
                    "next_end_datetime": end_date_anchor,
                    "next_chunk_number": chunk_number,
                    "completed": False,
                },
            )

        while days_remaining > 0:
            chunk_days = min(days_remaining, max_chunk_days)
            req_id = self.next_request_id()
            event = threading.Event()
            with self._state_lock:
                self.historical_data[req_id] = []
                self.earliest_date[req_id] = None
                self.request_events[req_id] = event

            print(f"[{job.symbol}] request {req_id}: {chunk_days} D from '{end_date_anchor}'")
            # Serialize socket writes while keeping outstanding requests concurrent.
            with self._send_lock:
                self.reqHistoricalData(
                    reqId=req_id,
                    contract=self.stock_contract(job),
                    endDateTime=end_date_anchor,
                    durationStr=f"{chunk_days} D",
                    barSizeSetting=normalized_bar_size,
                    whatToShow=job.what_to_show,
                    # extended and overnight both include data outside RTH;
                    # overnight is distinguished by the OVERNIGHT exchange.
                    useRTH=1 if job.session == "rth" else 0,
                    formatDate=1,
                    keepUpToDate=False,
                    chartOptions=[],
                )

            if not event.wait(timeout=timeout):
                self.cancelHistoricalData(req_id)
                self._cleanup_request(req_id)
                raise TimeoutError(f"{job.symbol}: request {req_id} timed out after {timeout}s")

            with self._state_lock:
                rows = list(self.historical_data.get(req_id, []))
                oldest = self.earliest_date.get(req_id)
                request_error = self.request_errors.get(req_id)
            self._cleanup_request(req_id)
            if request_error:
                raise RuntimeError(f"{job.symbol}: IB request failed ({request_error})")
            if not rows:
                break

            chunk = prepare_chunk(pd.DataFrame(rows))
            if chunk.empty:
                break

            part_path = paths["parts_dir"] / f"chunk_{chunk_number:06d}.csv"
            write_csv_atomic(chunk, part_path)

            next_days_remaining = max(0, days_remaining - chunk_days)
            if next_days_remaining and oldest:
                date_text = str(oldest).strip().split()[0].replace("-", "")
                next_end_date_anchor = f"{date_text}-00:00:00"
            else:
                next_end_date_anchor = ""

            # Advance progress only after the chunk is safely on disk.
            chunk_number += 1
            days_remaining = next_days_remaining
            end_date_anchor = next_end_date_anchor
            save_checkpoint(
                paths["checkpoint"],
                {
                    "version": 1,
                    "job": expected_job,
                    "days_remaining": days_remaining,
                    "next_end_datetime": end_date_anchor,
                    "next_chunk_number": chunk_number,
                    "completed": False,
                },
            )
            print(
                f"[{job.symbol}] checkpointed chunk {chunk_number}: "
                f"{len(chunk)} rows; {days_remaining} days remain"
            )

            if days_remaining and oldest:
                time.sleep(1.5)  # modest pacing between chunks for one symbol
            else:
                break

        part_files = sorted(paths["parts_dir"].glob("chunk_*.csv"))
        if not part_files:
            raise RuntimeError(f"{job.symbol}: no historical data received")

        row_count = finalize_parts(part_files, paths["output"])
        save_checkpoint(
            paths["checkpoint"],
            {
                "version": 1,
                "job": expected_job,
                "days_remaining": days_remaining,
                "next_end_datetime": end_date_anchor,
                "next_chunk_number": chunk_number,
                "completed": True,
                "row_count": row_count,
            },
        )
        shutil.rmtree(paths["parts_dir"])
        print(f"[{job.symbol}] saved {row_count} rows -> {paths['output']}")
        return DownloadResult(paths["output"], row_count, resumed=resumed)

    def _cleanup_request(self, req_id: int) -> None:
        with self._state_lock:
            self.historical_data.pop(req_id, None)
            self.request_events.pop(req_id, None)
            self.request_errors.pop(req_id, None)
            self.earliest_date.pop(req_id, None)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("._").lower()


def download_paths(job: DownloadJob, output_root: Path) -> dict[str, Path]:
    group_dir = output_root / safe_name(job.group)
    bar_name = safe_name(IBClient._normalize_bar_size(job.bar_size).replace(" ", ""))
    session = job.session
    stem = f"{job.symbol.lower()}_{bar_name}_{session}"
    return {
        "group_dir": group_dir,
        "output": group_dir / f"{stem}.csv",
        "checkpoint": group_dir / f"{stem}.checkpoint.json",
        "parts_dir": group_dir / f".{stem}.parts",
    }


def checkpoint_job(job: DownloadJob, normalized_bar_size: str) -> dict[str, Any]:
    return {
        "symbol": job.symbol,
        "group": job.group,
        "duration": job.duration,
        "bar_size": normalized_bar_size,
        "session": job.session,
        "what_to_show": job.what_to_show,
        "exchange": job.exchange,
        "primary_exchange": job.primary_exchange,
        "security_type": job.security_type,
        "currency": job.currency,
    }


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def prepare_chunk(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["datetime", "open", "high", "low", "close", "volume"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    result = df.loc[:, columns].copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    return (
        result.dropna(subset=["datetime"])
        .drop_duplicates(subset=["datetime"], keep="first")
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, path)


def finalize_parts(part_files: list[Path], output_path: Path) -> int:
    """Stream backward-downloaded parts into one chronological, unique CSV."""
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    wrote_header = False
    row_count = 0
    last_datetime: pd.Timestamp | None = None

    # Newer chunks were downloaded first, so reverse the files for chronology.
    for part_path in reversed(part_files):
        chunk = prepare_chunk(pd.read_csv(part_path))
        if last_datetime is not None:
            chunk = chunk[chunk["datetime"] > last_datetime]
        if chunk.empty:
            continue
        chunk.to_csv(temp_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        row_count += len(chunk)
        last_datetime = chunk["datetime"].iloc[-1]

    if not wrote_header:
        raise RuntimeError("Downloaded chunks contained no valid rows")
    os.replace(temp_path, output_path)
    return row_count


def run_batch(app: IBClient, jobs: list[DownloadJob], workers: int, output_root: Path) -> int:
    failures = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs)), thread_name_prefix="ib-download") as pool:
        futures = {
            pool.submit(app.get_historical_data, job, output_root): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # one failed ticker does not stop the batch
                failures += 1
                print(f"[{job.symbol}] FAILED: {exc}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical data from a local IB Gateway.")
    parser.add_argument("--config", type=Path, help="YAML batch configuration file",default=BASE_DIR / "config" / "scraper-config-semicond.yml")
    parser.add_argument("-s", "--symbol", help="Single symbol (keeps the original CLI mode)")
    parser.add_argument("-o", "--overnight", type=parse_bool, default=False)
    parser.add_argument("-d", "--duration", default="365 D")
    parser.add_argument("-bs", "--bar_size", default="1 hour")
    parser.add_argument("--group", default="ungrouped", help="Output group for single-symbol mode")
    parser.add_argument("--workers", type=int, default=3, help="Workers in single-symbol mode")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.config:
        jobs, workers, output_root = load_jobs(args.config)
    else:
        jobs = [
            DownloadJob(
                symbol=(args.symbol or "NVDA").upper(),
                group=args.group,
                duration=args.duration,
                bar_size=args.bar_size,
                session="overnight" if args.overnight else "rth",
            )
        ]
        workers = args.workers
        output_root = Path(BASE_DIR) / "data"

    app = IBClient()
    try:
        app.connect_ib()
        failures = run_batch(app, jobs, workers, output_root)
        print(f"Completed: {len(jobs) - failures} succeeded, {failures} failed")
        return 1 if failures else 0
    finally:
        if app.isConnected():
            app.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())