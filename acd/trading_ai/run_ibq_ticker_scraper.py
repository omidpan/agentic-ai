#!/usr/bin/env python3
"""Run ibq_ticker_scraper.py and restart it after an IBKR pacing wait."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


SCRIPT = Path(__file__).with_name("ibq_ticker_scraper.py")
DEFAULT_CONFIG = Path(
    "/Users/leo/champlain/agentic-ai/acd/trading_ai/config/"
    "scraper-config-semicond-ticks.yml"
)
PACING_TEXT = "pacing wait: approximately"
SUCCESS_TEXT = "Completed: 1 succeeded, 0 failed"
RESTART_DELAY_SECONDS = 5
STOP_TIMEOUT_SECONDS = 10


def stop_process(process: subprocess.Popen[str]) -> None:
    """Stop the scraper process group and wait until it is really gone."""
    if process.poll() is not None:
        return

    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_once(arguments: list[str]) -> str:
    process = subprocess.Popen(
        [sys.executable, "-u", str(SCRIPT), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    print(f"Started {SCRIPT.name} (PID {process.pid})", flush=True)

    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            normalized_line = line.casefold()

            if SUCCESS_TEXT.casefold() in normalized_line:
                # The scraper reported success. Wait for its normal shutdown.
                return_code = process.wait()
                return "success" if return_code == 0 else "failed"

            if PACING_TEXT.casefold() in normalized_line:
                print("Pacing wait detected; stopping scraper...", flush=True)
                stop_process(process)
                return "restart"

        return_code = process.wait()
        print(
            f"Scraper exited with code {return_code} without the success message.",
            file=sys.stderr,
            flush=True,
        )
        return "failed"
    finally:
        stop_process(process)


def main() -> int:
    if not SCRIPT.is_file():
        print(f"Cannot find {SCRIPT}", file=sys.stderr)
        return 2

    scraper_arguments = sys.argv[1:]
    if not scraper_arguments:
        scraper_arguments = ["--config", str(DEFAULT_CONFIG)]

    try:
        while True:
            result = run_once(scraper_arguments)

            if result == "success":
                print("Scraper completed successfully.", flush=True)
                return 0

            if result == "failed":
                return 1

            print(
                f"Scraper is stopped. Restarting in {RESTART_DELAY_SECONDS} seconds...",
                flush=True,
            )
            time.sleep(RESTART_DELAY_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
