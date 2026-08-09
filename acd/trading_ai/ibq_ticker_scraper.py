"""Download IBKR historical Time & Sales ticks from a local IB Gateway.

This is intentionally separate from ``ibq_scraper.py``.  It does not change
the existing historical-bar workflow or its checkpoints.

Run a YAML-configured batch:

    python ibq_tick_scraper.py --config config/scraper-config-semicond-ticks.yml

IBKR calls this data "Historical Time & Sales".  The API method used here is
``reqHistoricalTicks``; it is different from the live-streaming
``reqTickByTickData`` method.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

import config as app_config
from utils.appenv import APPENV


# Match the base-directory behavior of the existing bar scraper.
if hasattr(app_config, "BASE_DIR"):
    BASE_DIR = Path(app_config.BASE_DIR)
elif hasattr(app_config, "DATA_DIR"):
    BASE_DIR = Path(app_config.DATA_DIR).parent
else:
    BASE_DIR = Path(__file__).resolve().parent


MAX_HISTORICAL_TICK_SECONDS = 3 * 365 * 24 * 60 * 60
SUPPORTED_TICK_TYPES = {"TRADES", "BID_ASK", "MIDPOINT"}


@dataclass(frozen=True)
class TickDownloadJob:
    symbol: str
    group: str
    duration: str
    session: str = "rth"
    tick_type: str = "TRADES"
    ticks_per_request: int = 1000
    ignore_size: bool = False
    end_datetime: str | None = None
    end_delay_seconds: int = 5
    request_timeout_seconds: int = 120
    max_retries: int = 2
    retry_delay_seconds: float = 16.0
    empty_step_seconds: int = 86_400
    exchange: str | None = None
    primary_exchange: str | None = None
    security_type: str = "STK"
    currency: str = "USD"


@dataclass(frozen=True)
class TickDownloadResult:
    path: Path
    row_count: int
    resumed: bool = False


@dataclass(frozen=True)
class PacingSettings:
    # IBKR documents 60 historical request units per ten minutes.  Use 55 to
    # retain headroom for another process using the same account/session.
    max_request_units: int = 55
    window_seconds: float = 600.0
    # IBKR documents fewer than six requests for the same
    # contract/exchange/tick type in two seconds.
    same_contract_max_requests: int = 5
    same_contract_window_seconds: float = 2.0
    minimum_interval_seconds: float = 0.25


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def normalize_tick_type(value: Any) -> str:
    normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "TRADE": "TRADES",
        "LAST": "TRADES",
        "BIDASK": "BID_ASK",
        "BID__ASK": "BID_ASK",
        "MID": "MIDPOINT",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_TICK_TYPES:
        choices = ", ".join(sorted(SUPPORTED_TICK_TYPES))
        raise ValueError(f"Unsupported tick type {value!r}; use {choices}.")
    return normalized


def parse_sessions(settings: dict[str, Any]) -> list[str]:
    raw_sessions = settings.get("sessions")
    if raw_sessions is None:
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


def parse_duration_seconds(duration: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([HDWMY])\s*", str(duration).upper())
    if not match:
        raise ValueError(
            f"Unsupported duration {duration!r}; use forms such as '6 H', "
            "'5 D', '4 W', '6 M', or '1 Y'."
        )
    value, unit = int(match.group(1)), match.group(2)
    multipliers = {
        "H": 60 * 60,
        "D": 24 * 60 * 60,
        "W": 7 * 24 * 60 * 60,
        "M": 30 * 24 * 60 * 60,
        "Y": 365 * 24 * 60 * 60,
    }
    seconds = value * multipliers[unit]
    if seconds <= 0:
        raise ValueError("duration must be greater than zero.")
    if seconds > MAX_HISTORICAL_TICK_SECONDS:
        raise ValueError(
            f"duration {duration!r} exceeds IBKR's three-year Historical "
            "Time & Sales availability limit."
        )
    return seconds


def parse_end_datetime(value: str) -> int:
    text = value.strip()
    if not text:
        raise ValueError("end_datetime cannot be empty.")

    # First accept normal ISO-8601 values, preferably with Z or an offset.
    iso_text = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError:
        parsed = None

    if parsed is None:
        formats = (
            "%Y%m%d %H:%M:%S UTC",
            "%Y%m%d-%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        )
        for date_format in formats:
            try:
                parsed = datetime.strptime(text, date_format).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue

    if parsed is None:
        raise ValueError(
            f"Invalid end_datetime {value!r}; use ISO-8601, for example "
            "'2026-08-07T20:00:00Z'."
        )
    if parsed.tzinfo is None:
        # A timezone-free YAML value is interpreted as UTC deliberately.  This
        # avoids depending on the timezone selected in the Gateway login UI.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp())


def resolve_end_epoch(job: TickDownloadJob) -> int:
    if job.end_datetime:
        return parse_end_datetime(job.end_datetime)
    now_utc = datetime.now(timezone.utc)
    return int((now_utc - timedelta(seconds=job.end_delay_seconds)).timestamp())


def _positive_int(settings: dict[str, Any], name: str, default: int) -> int:
    value = int(settings.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


def _non_negative_int(settings: dict[str, Any], name: str, default: int) -> int:
    value = int(settings.get(name, default))
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")
    return value


def load_jobs(
    config_path: Path,
) -> tuple[list[TickDownloadJob], int, Path, PacingSettings]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    defaults = config.get("defaults", {})
    groups = config.get("groups", {})
    if not isinstance(defaults, dict):
        raise ValueError("'defaults' must be a mapping.")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Config must contain a non-empty 'groups' mapping.")

    jobs: list[TickDownloadJob] = []
    seen: set[tuple[str, str, str, str]] = set()

    for group, group_config in groups.items():
        if isinstance(group_config, list):
            group_config = {"symbols": group_config}
        if not isinstance(group_config, dict):
            raise ValueError(f"Group {group!r} must be a list or mapping.")

        merged_group = {**defaults, **group_config}
        symbols = merged_group.get("symbols", [])
        if not symbols:
            raise ValueError(f"Group {group!r} has no symbols.")

        for symbol_entry in symbols:
            overrides = symbol_entry if isinstance(symbol_entry, dict) else {}
            symbol = overrides.get("symbol") if overrides else symbol_entry
            if not symbol:
                raise ValueError(f"Invalid symbol entry in group {group!r}.")
            settings = {**merged_group, **overrides}

            duration = str(settings.get("duration", "1 D"))
            parse_duration_seconds(duration)  # Validate before connecting to IBKR.

            tick_type = normalize_tick_type(
                settings.get("tick_type", settings.get("what_to_show", "TRADES"))
            )
            ticks_per_request = _positive_int(settings, "ticks_per_request", 1000)
            if ticks_per_request > 1000:
                raise ValueError(
                    f"{symbol}: ticks_per_request cannot exceed IBKR's limit of 1000."
                )

            for session in parse_sessions(settings):
                key = (
                    str(group).lower(),
                    str(symbol).upper(),
                    session,
                    tick_type,
                )
                if key in seen:
                    raise ValueError(
                        "Duplicate tick job: "
                        f"group={group}, symbol={symbol}, session={session}, "
                        f"tick_type={tick_type}"
                    )
                seen.add(key)

                job = TickDownloadJob(
                    symbol=str(symbol).upper(),
                    group=str(group).lower(),
                    duration=duration,
                    session=session,
                    tick_type=tick_type,
                    ticks_per_request=ticks_per_request,
                    ignore_size=parse_bool(settings.get("ignore_size", False)),
                    end_datetime=(
                        str(settings["end_datetime"])
                        if settings.get("end_datetime") is not None
                        else None
                    ),
                    end_delay_seconds=_non_negative_int(
                        settings, "end_delay_seconds", 5
                    ),
                    request_timeout_seconds=_positive_int(
                        settings, "request_timeout_seconds", 120
                    ),
                    max_retries=_non_negative_int(settings, "max_retries", 2),
                    retry_delay_seconds=float(
                        settings.get("retry_delay_seconds", 16.0)
                    ),
                    empty_step_seconds=_positive_int(
                        settings, "empty_step_seconds", 86_400
                    ),
                    exchange=settings.get("exchange"),
                    primary_exchange=settings.get("primary_exchange"),
                    security_type=str(settings.get("security_type", "STK")),
                    currency=str(settings.get("currency", "USD")),
                )
                if job.retry_delay_seconds < 0:
                    raise ValueError("retry_delay_seconds cannot be negative.")
                if job.end_datetime:
                    parse_end_datetime(job.end_datetime)
                jobs.append(job)

    max_workers = int(config.get("max_workers", 3))
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")

    output_root = Path(config.get("output_root", Path(BASE_DIR) / "data" / "ticks"))
    if not output_root.is_absolute():
        output_root = Path(BASE_DIR) / output_root

    pacing_config = config.get("pacing", {}) or {}
    if not isinstance(pacing_config, dict):
        raise ValueError("'pacing' must be a mapping.")
    pacing = PacingSettings(
        max_request_units=int(pacing_config.get("max_request_units", 55)),
        window_seconds=float(pacing_config.get("window_seconds", 600.0)),
        same_contract_max_requests=int(
            pacing_config.get("same_contract_max_requests", 5)
        ),
        same_contract_window_seconds=float(
            pacing_config.get("same_contract_window_seconds", 2.0)
        ),
        minimum_interval_seconds=float(
            pacing_config.get("minimum_interval_seconds", 0.25)
        ),
    )
    if pacing.max_request_units < 1:
        raise ValueError("pacing.max_request_units must be at least 1.")
    if pacing.window_seconds <= 0:
        raise ValueError("pacing.window_seconds must be greater than zero.")
    if pacing.same_contract_max_requests < 1:
        raise ValueError("pacing.same_contract_max_requests must be at least 1.")
    if pacing.same_contract_window_seconds <= 0:
        raise ValueError(
            "pacing.same_contract_window_seconds must be greater than zero."
        )
    if pacing.minimum_interval_seconds < 0:
        raise ValueError("pacing.minimum_interval_seconds cannot be negative.")

    return jobs, max_workers, output_root, pacing


class HistoricalRequestPacer:
    """Thread-safe rolling-window limiter shared by every worker.

    BID_ASK requests cost two historical request units according to IBKR's
    documented historical-data pacing rules.
    """

    def __init__(self, settings: PacingSettings) -> None:
        self.settings = settings
        self._condition = threading.Condition()
        self._global_events: deque[tuple[float, int]] = deque()
        self._global_units = 0
        self._contract_events: dict[tuple[str, str, str], deque[float]] = defaultdict(deque)
        self._next_send_time = 0.0

    def acquire(
        self,
        key: tuple[str, str, str],
        units: int,
        label: str,
    ) -> None:
        if units < 1 or units > self.settings.max_request_units:
            raise ValueError("Invalid historical request pacing cost.")

        announced_wait = False
        while True:
            with self._condition:
                now = time.monotonic()
                self._purge(now, key)

                waits = [max(0.0, self._next_send_time - now)]

                excess_units = (
                    self._global_units + units - self.settings.max_request_units
                )
                if excess_units > 0:
                    released_units = 0
                    for event_time, event_units in self._global_events:
                        released_units += event_units
                        if released_units >= excess_units:
                            waits.append(
                                max(
                                    0.0,
                                    event_time + self.settings.window_seconds - now,
                                )
                            )
                            break

                contract_events = self._contract_events[key]
                if len(contract_events) >= self.settings.same_contract_max_requests:
                    waits.append(
                        max(
                            0.0,
                            contract_events[0]
                            + self.settings.same_contract_window_seconds
                            - now,
                        )
                    )

                wait_seconds = max(waits)
                if wait_seconds > 0:
                    if wait_seconds >= 1.0 and not announced_wait:
                        print(
                            f"[{label}] pacing wait: approximately "
                            f"{wait_seconds:.1f} seconds"
                        )
                        announced_wait = True
                    self._condition.wait(timeout=wait_seconds)
                    continue

                sent_at = time.monotonic()
                self._global_events.append((sent_at, units))
                self._global_units += units
                contract_events.append(sent_at)
                self._next_send_time = (
                    sent_at + self.settings.minimum_interval_seconds
                )
                return

    def _purge(self, now: float, key: tuple[str, str, str]) -> None:
        while (
            self._global_events
            and now - self._global_events[0][0] >= self.settings.window_seconds
        ):
            _, units = self._global_events.popleft()
            self._global_units -= units

        contract_events = self._contract_events[key]
        while (
            contract_events
            and now - contract_events[0]
            >= self.settings.same_contract_window_seconds
        ):
            contract_events.popleft()


class IBRequestError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"IB error {code}: {message}")
        self.code = code
        self.message = message

    @property
    def retryable(self) -> bool:
        text = self.message.lower()
        return "pacing" in text or "temporarily" in text or "try again" in text


class IBTickClient(EWrapper, EClient, APPENV):
    INFORMATIONAL_CODES = {2104, 2106, 2107, 2108, 2158}
    CONNECTION_LOST_CODES = {1100, 1300}
    CONNECTION_RESTORED_CODES = {1101, 1102}

    def __init__(self, pacer: HistoricalRequestPacer) -> None:
        EClient.__init__(self, self)
        APPENV.__init__(self)
        self.pacer = pacer
        self.connected_event = threading.Event()
        self.tick_rows: dict[int, list[dict[str, Any]]] = {}
        self.request_events: dict[int, threading.Event] = {}
        self.request_errors: dict[int, IBRequestError] = {}
        self.no_data_requests: set[int] = set()
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
        if errorCode in self.INFORMATIONAL_CODES:
            print(f"IB INFO {errorCode}: {errorString}")
            return

        if errorCode in self.CONNECTION_RESTORED_CODES:
            print(f"IB INFO {errorCode}: {errorString}")
            return

        print(f"IB ERROR request={reqId} code={errorCode}: {errorString}")
        error_text = errorString.lower()
        is_no_data = errorCode in {162, 165} and (
            "no data" in error_text or "no such data" in error_text
        )

        with self._state_lock:
            if errorCode in self.CONNECTION_LOST_CODES:
                for active_req_id, event in self.request_events.items():
                    self.request_errors[active_req_id] = IBRequestError(
                        errorCode, errorString
                    )
                    event.set()
                return

            event = self.request_events.get(reqId)
            if reqId >= 0 and event is not None:
                if is_no_data:
                    self.no_data_requests.add(reqId)
                else:
                    self.request_errors[reqId] = IBRequestError(
                        errorCode, errorString
                    )
                event.set()

    def connectionClosed(self) -> None:  # noqa: N802 (IB callback)
        with self._state_lock:
            for req_id, event in self.request_events.items():
                self.request_errors[req_id] = IBRequestError(
                    0, "IB API connection closed"
                )
                event.set()

    def connect_ib(self) -> None:
        print(f"Connecting IB {self.host}:{self.port}")
        self.connect(self.host, self.port, self.client_id)
        threading.Thread(
            target=self.run,
            daemon=True,
            name="ib-api-loop",
        ).start()
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
    def stock_contract(job: TickDownloadJob) -> Contract:
        contract = Contract()
        contract.symbol = job.symbol
        contract.secType = job.security_type
        contract.currency = job.currency

        if job.exchange:
            contract.exchange = job.exchange
        elif job.symbol in {"BIT", "BITCOIN"}:
            contract.exchange = "PAXOS"
        elif job.session == "overnight":
            # IBKR treats the US overnight market as a distinct venue.
            contract.exchange = "OVERNIGHT"
        else:
            contract.exchange = "SMART"

        if job.primary_exchange:
            contract.primaryExchange = job.primary_exchange
        elif job.symbol == "VIX":
            contract.primaryExchange = "CBOE"
        return contract

    def historicalTicks(self, reqId: int, ticks: Any, done: bool) -> None:  # noqa: N802
        rows = []
        for tick in ticks:
            epoch = int(tick.time)
            rows.append(
                {
                    "timestamp_utc": epoch_to_iso_utc(epoch),
                    "epoch_seconds": epoch,
                    "midpoint": to_float(tick.price),
                    "size": to_float(getattr(tick, "size", 0)),
                }
            )
        self._append_tick_rows(reqId, rows, done)

    def historicalTicksBidAsk(  # noqa: N802
        self,
        reqId: int,
        ticks: Any,
        done: bool,
    ) -> None:
        rows = []
        for tick in ticks:
            epoch = int(tick.time)
            attributes = getattr(tick, "tickAttribBidAsk", None)
            rows.append(
                {
                    "timestamp_utc": epoch_to_iso_utc(epoch),
                    "epoch_seconds": epoch,
                    "bid_price": to_float(tick.priceBid),
                    "ask_price": to_float(tick.priceAsk),
                    "bid_size": to_float(tick.sizeBid),
                    "ask_size": to_float(tick.sizeAsk),
                    "bid_past_low": bool(
                        getattr(attributes, "bidPastLow", False)
                    ),
                    "ask_past_high": bool(
                        getattr(attributes, "askPastHigh", False)
                    ),
                }
            )
        self._append_tick_rows(reqId, rows, done)

    def historicalTicksLast(  # noqa: N802
        self,
        reqId: int,
        ticks: Any,
        done: bool,
    ) -> None:
        rows = []
        for tick in ticks:
            epoch = int(tick.time)
            attributes = getattr(tick, "tickAttribLast", None)
            rows.append(
                {
                    "timestamp_utc": epoch_to_iso_utc(epoch),
                    "epoch_seconds": epoch,
                    "price": to_float(tick.price),
                    "size": to_float(tick.size),
                    "exchange": str(getattr(tick, "exchange", "")),
                    "special_conditions": str(
                        getattr(tick, "specialConditions", "")
                    ),
                    "past_limit": bool(
                        getattr(attributes, "pastLimit", False)
                    ),
                    "unreported": bool(
                        getattr(attributes, "unreported", False)
                    ),
                }
            )
        self._append_tick_rows(reqId, rows, done)

    def _append_tick_rows(
        self,
        req_id: int,
        rows: list[dict[str, Any]],
        done: bool,
    ) -> None:
        with self._state_lock:
            target = self.tick_rows.get(req_id)
            event = self.request_events.get(req_id)
            if target is None:
                return
            target.extend(rows)
        if done and event is not None:
            event.set()

    def request_tick_page(
        self,
        job: TickDownloadJob,
        cursor_end_epoch: int,
    ) -> list[dict[str, Any]]:
        for attempt in range(job.max_retries + 1):
            if attempt:
                # Repeating the same historical query too quickly can itself
                # be a pacing violation.  Keep retries more than 15 seconds apart.
                retry_delay = max(
                    16.0,
                    job.retry_delay_seconds * (2 ** (attempt - 1)),
                )
                print(
                    f"[{job.symbol}/{job.session}] retry {attempt}/"
                    f"{job.max_retries} in {retry_delay:.1f} seconds"
                )
                time.sleep(retry_delay)

            try:
                return self._request_tick_page_once(job, cursor_end_epoch)
            except (TimeoutError, IBRequestError) as exc:
                retryable = isinstance(exc, TimeoutError) or exc.retryable
                if attempt >= job.max_retries or not retryable:
                    raise
                print(f"[{job.symbol}/{job.session}] transient request failure: {exc}")

        raise AssertionError("unreachable")

    def _request_tick_page_once(
        self,
        job: TickDownloadJob,
        cursor_end_epoch: int,
    ) -> list[dict[str, Any]]:
        contract = self.stock_contract(job)
        pacing_key = (job.symbol, contract.exchange, job.tick_type)
        request_units = 2 if job.tick_type == "BID_ASK" else 1
        label = f"{job.symbol}/{job.session}/{job.tick_type}"
        self.pacer.acquire(pacing_key, request_units, label)

        if not self.isConnected():
            raise ConnectionError("IB API socket is not connected")

        req_id = self.next_request_id()
        event = threading.Event()
        with self._state_lock:
            self.tick_rows[req_id] = []
            self.request_events[req_id] = event

        end_datetime = epoch_to_ib_utc(cursor_end_epoch)
        print(
            f"[{job.symbol}/{job.session}] request {req_id}: "
            f"{job.tick_type}, ending {end_datetime}"
        )

        try:
            # EClient socket writes are serialized.  The requests themselves
            # remain outstanding concurrently and callbacks are correlated by ID.
            with self._send_lock:
                self.reqHistoricalTicks(
                    reqId=req_id,
                    contract=contract,
                    startDateTime="",
                    endDateTime=end_datetime,
                    numberOfTicks=job.ticks_per_request,
                    whatToShow=job.tick_type,
                    useRth=1 if job.session == "rth" else 0,
                    ignoreSize=job.ignore_size,
                    miscOptions=[],
                )
        except Exception:
            self._cleanup_request(req_id)
            raise

        if not event.wait(timeout=job.request_timeout_seconds):
            # reqHistoricalTicks has no dedicated cancel method in the socket
            # API.  The unique request ID lets us safely ignore a late callback.
            self._cleanup_request(req_id)
            raise TimeoutError(
                f"{job.symbol}: request {req_id} timed out after "
                f"{job.request_timeout_seconds}s"
            )

        with self._state_lock:
            rows = list(self.tick_rows.get(req_id, []))
            request_error = self.request_errors.get(req_id)
            no_data = req_id in self.no_data_requests
        self._cleanup_request(req_id)

        if request_error:
            raise request_error
        if no_data:
            return []
        return rows

    def _cleanup_request(self, req_id: int) -> None:
        with self._state_lock:
            self.tick_rows.pop(req_id, None)
            self.request_events.pop(req_id, None)
            self.request_errors.pop(req_id, None)
            self.no_data_requests.discard(req_id)


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(str(value))


def epoch_to_iso_utc(epoch: int) -> str:
    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def epoch_to_ib_utc(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
        "%Y%m%d %H:%M:%S UTC"
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("._").lower()


def download_paths(job: TickDownloadJob, output_root: Path) -> dict[str, Path]:
    group_dir = output_root / safe_name(job.group)
    tick_name = safe_name(job.tick_type)
    stem = f"{job.symbol.lower()}_{tick_name}_ticks_{job.session}"
    return {
        "group_dir": group_dir,
        "output": group_dir / f"{stem}.csv",
        "checkpoint": group_dir / f"{stem}.checkpoint.json",
        "parts_dir": group_dir / f".{stem}.parts",
    }


def checkpoint_job(job: TickDownloadJob) -> dict[str, Any]:
    return {
        "symbol": job.symbol,
        "group": job.group,
        "duration": job.duration,
        "session": job.session,
        "tick_type": job.tick_type,
        "ticks_per_request": job.ticks_per_request,
        "ignore_size": job.ignore_size,
        "end_datetime": job.end_datetime,
        "end_delay_seconds": job.end_delay_seconds,
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
        json.dump(state, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def raw_tick_columns(tick_type: str) -> list[str]:
    common = ["timestamp_utc", "epoch_seconds"]
    if tick_type == "TRADES":
        return common + [
            "price",
            "size",
            "exchange",
            "special_conditions",
            "past_limit",
            "unreported",
        ]
    if tick_type == "BID_ASK":
        return common + [
            "bid_price",
            "ask_price",
            "bid_size",
            "ask_size",
            "bid_past_low",
            "ask_past_high",
        ]
    return common + ["midpoint", "size"]


def prepare_tick_page(df: pd.DataFrame, tick_type: str) -> pd.DataFrame:
    columns = raw_tick_columns(tick_type)
    if df.empty:
        return pd.DataFrame(columns=columns)

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Tick page is missing columns: {missing}")

    result = df.loc[:, columns].copy()
    result["epoch_seconds"] = pd.to_numeric(
        result["epoch_seconds"], errors="coerce"
    )
    result = result.dropna(subset=["epoch_seconds"])
    result["epoch_seconds"] = result["epoch_seconds"].astype("int64")
    result["timestamp_utc"] = pd.to_datetime(
        result["epoch_seconds"], unit="s", utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Preserve legitimate duplicate trades within the same second.  The
    # callback order is used as a stable tiebreaker and is not written to CSV.
    result["_callback_order"] = range(len(result))
    result = result.sort_values(
        ["epoch_seconds", "_callback_order"],
        kind="mergesort",
    )
    return result.drop(columns=["_callback_order"]).reset_index(drop=True)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, path)


def finalize_parts(
    part_files: list[Path],
    output_path: Path,
    tick_type: str,
) -> int:
    """Stream newest-first pages into one chronological CSV.

    IBKR can return more than the requested 1,000 ticks to finish a complete
    second.  Each next request therefore moves one second earlier.  This keeps
    page boundaries non-overlapping without deleting legitimate identical
    trades that occurred during the same second.
    """

    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    wrote_header = False
    row_count = 0
    last_epoch: int | None = None

    for part_path in reversed(part_files):
        page = prepare_tick_page(pd.read_csv(part_path), tick_type)
        if last_epoch is not None:
            # Defensive boundary protection.  A complete second should never
            # be split by IBKR, so older/equal rows here are page overlap.
            page = page[page["epoch_seconds"] > last_epoch]
        if page.empty:
            continue

        page.insert(0, "sequence", range(row_count, row_count + len(page)))
        page.to_csv(
            temp_path,
            mode="a",
            header=not wrote_header,
            index=False,
        )
        wrote_header = True
        row_count += len(page)
        last_epoch = int(page["epoch_seconds"].iloc[-1])

    if not wrote_header:
        raise RuntimeError("Downloaded tick pages contained no valid rows")
    os.replace(temp_path, output_path)
    return row_count


def checkpoint_state(
    job: TickDownloadJob,
    start_epoch: int,
    end_epoch: int,
    cursor_end_epoch: int,
    next_page_number: int,
    rows_checkpointed: int,
    completed: bool,
    row_count: int | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "version": 1,
        "job": checkpoint_job(job),
        "start_epoch": start_epoch,
        "start_utc": epoch_to_iso_utc(start_epoch),
        "end_epoch": end_epoch,
        "end_utc": epoch_to_iso_utc(end_epoch),
        "cursor_end_epoch": cursor_end_epoch,
        "cursor_end_utc": epoch_to_iso_utc(cursor_end_epoch),
        "next_page_number": next_page_number,
        "rows_checkpointed": rows_checkpointed,
        "completed": completed,
    }
    if row_count is not None:
        state["row_count"] = row_count
    return state


def download_historical_ticks(
    app: IBTickClient,
    job: TickDownloadJob,
    output_root: Path,
) -> TickDownloadResult:
    paths = download_paths(job, output_root)
    paths["group_dir"].mkdir(parents=True, exist_ok=True)
    paths["parts_dir"].mkdir(parents=True, exist_ok=True)

    expected_job = checkpoint_job(job)
    checkpoint = load_checkpoint(paths["checkpoint"])
    resumed = checkpoint is not None

    if checkpoint:
        if checkpoint.get("job") != expected_job:
            raise ValueError(
                f"{job.symbol}/{job.session}: existing checkpoint does not "
                "match this job. Remove "
                f"{paths['checkpoint']} and {paths['parts_dir']} to restart."
            )
        if checkpoint.get("completed"):
            if not paths["output"].exists():
                raise RuntimeError(
                    f"Completed checkpoint exists but output is missing: "
                    f"{paths['output']}"
                )
            row_count = int(checkpoint.get("row_count", 0))
            print(
                f"[{job.symbol}/{job.session}] already complete -> "
                f"{paths['output']}"
            )
            return TickDownloadResult(paths["output"], row_count, resumed=True)

        start_epoch = int(checkpoint["start_epoch"])
        end_epoch = int(checkpoint["end_epoch"])
        cursor_end_epoch = int(checkpoint["cursor_end_epoch"])
        page_number = int(checkpoint["next_page_number"])
        rows_checkpointed = int(checkpoint.get("rows_checkpointed", 0))
        print(
            f"[{job.symbol}/{job.session}] resuming at page {page_number}; "
            f"cursor={epoch_to_iso_utc(cursor_end_epoch)}"
        )
    else:
        duration_seconds = parse_duration_seconds(job.duration)
        end_epoch = resolve_end_epoch(job)
        start_epoch = end_epoch - duration_seconds
        cursor_end_epoch = end_epoch
        page_number = 0
        rows_checkpointed = 0
        save_checkpoint(
            paths["checkpoint"],
            checkpoint_state(
                job,
                start_epoch,
                end_epoch,
                cursor_end_epoch,
                page_number,
                rows_checkpointed,
                completed=False,
            ),
        )

    while cursor_end_epoch >= start_epoch:
        rows = app.request_tick_page(job, cursor_end_epoch)
        if not rows:
            cursor_end_epoch -= job.empty_step_seconds
            save_checkpoint(
                paths["checkpoint"],
                checkpoint_state(
                    job,
                    start_epoch,
                    end_epoch,
                    cursor_end_epoch,
                    page_number,
                    rows_checkpointed,
                    completed=False,
                ),
            )
            print(
                f"[{job.symbol}/{job.session}] no ticks; moved cursor to "
                f"{epoch_to_iso_utc(cursor_end_epoch)}"
            )
            continue

        page = prepare_tick_page(pd.DataFrame(rows), job.tick_type)
        if page.empty:
            raise RuntimeError(
                f"{job.symbol}/{job.session}: IBKR returned an invalid empty page"
            )

        earliest_epoch = int(page["epoch_seconds"].iloc[0])
        if earliest_epoch > cursor_end_epoch:
            raise RuntimeError(
                f"{job.symbol}/{job.session}: historical tick cursor did not "
                "move backward; check Gateway timezone/API logs."
            )

        in_range = page[
            (page["epoch_seconds"] >= start_epoch)
            & (page["epoch_seconds"] <= end_epoch)
        ].reset_index(drop=True)

        if not in_range.empty:
            part_path = paths["parts_dir"] / f"page_{page_number:09d}.csv"
            write_csv_atomic(in_range, part_path)
            page_number += 1
            rows_checkpointed += len(in_range)

        # IBKR promises that it may exceed 1,000 ticks to complete the whole
        # second.  Moving back one second therefore cannot split a busy second.
        cursor_end_epoch = earliest_epoch - 1
        save_checkpoint(
            paths["checkpoint"],
            checkpoint_state(
                job,
                start_epoch,
                end_epoch,
                cursor_end_epoch,
                page_number,
                rows_checkpointed,
                completed=False,
            ),
        )
        print(
            f"[{job.symbol}/{job.session}] checkpointed page {page_number}: "
            f"{len(in_range)} rows; cursor={epoch_to_iso_utc(cursor_end_epoch)}"
        )

    part_files = sorted(paths["parts_dir"].glob("page_*.csv"))
    if not part_files:
        raise RuntimeError(
            f"{job.symbol}/{job.session}: no historical {job.tick_type} ticks "
            "were received for the configured interval"
        )

    row_count = finalize_parts(part_files, paths["output"], job.tick_type)
    save_checkpoint(
        paths["checkpoint"],
        checkpoint_state(
            job,
            start_epoch,
            end_epoch,
            cursor_end_epoch,
            page_number,
            rows_checkpointed,
            completed=True,
            row_count=row_count,
        ),
    )
    shutil.rmtree(paths["parts_dir"])
    print(
        f"[{job.symbol}/{job.session}] saved {row_count} rows -> "
        f"{paths['output']}"
    )
    return TickDownloadResult(paths["output"], row_count, resumed=resumed)


def run_batch(
    app: IBTickClient,
    jobs: list[TickDownloadJob],
    workers: int,
    output_root: Path,
) -> int:
    failures = 0
    with ThreadPoolExecutor(
        max_workers=min(workers, len(jobs)),
        thread_name_prefix="ib-ticks",
    ) as pool:
        futures = {
            pool.submit(download_historical_ticks, app, job, output_root): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # One failed ticker does not stop the batch.
                failures += 1
                print(f"[{job.symbol}/{job.session}] FAILED: {exc}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download IBKR historical Time & Sales ticks from a local Gateway."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=BASE_DIR / "config" / "scraper-config-semicond-ticks.yml",
        help="YAML tick batch configuration file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs, workers, output_root, pacing_settings = load_jobs(args.config)
    pacer = HistoricalRequestPacer(pacing_settings)
    app = IBTickClient(pacer)

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