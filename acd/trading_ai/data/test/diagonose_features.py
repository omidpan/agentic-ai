"""Trace NaN creation through the uploaded stock feature pipeline.

Example
-------
python diagnose_feature_nans.py \
    --functions "upload/Pasted code(4).py" \
    --input upload/fake_data.csv \
    --reference "upload/feature_added_fake_data(1).csv" \
    --report-dir nan_reports
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


PIPELINE: list[tuple[str, dict[str, Any]]] = [
    ("add_log_returns", {}),
    ("add_candle_features", {}),
    ("add_gap_features", {}),
    ("add_trend_features", {}),
    ("add_volume_features", {"window": 20}),
    ("add_rolling_statistics", {"window": 20}),
    ("add_volatility_features", {}),
    ("add_zscore_features", {"window": 20}),
]


def load_functions(path: Path) -> dict[str, Any]:
    """Execute the original function file with pandas/numpy made available."""
    namespace: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "__file__": str(path),
        "__name__": "uploaded_feature_functions",
    }
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace


def prepare_input(path: Path) -> pd.DataFrame:
    """Load the source CSV and enforce chronological ascending order."""
    df = pd.read_csv(path)

    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
    return df.sort_values("datetime", ascending=True).reset_index(drop=True)


def trace_pipeline(
    df: pd.DataFrame,
    namespace: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run each original function and record NaNs created by that function."""
    result = df.copy()
    known_columns = set(result.columns)
    records: list[dict[str, Any]] = []

    for function_name, kwargs in PIPELINE:
        function: Callable[..., pd.DataFrame] = namespace[function_name]
        before_nan = result.isna().sum()
        result = function(result, **kwargs)

        new_columns = [column for column in result.columns if column not in known_columns]
        known_columns.update(new_columns)

        # A later function may overwrite an existing feature (VolumeZ does this).
        all_columns = list(result.columns)
        before_aligned = before_nan.reindex(all_columns, fill_value=0)
        after_nan = result.isna().sum()
        changed_columns = [
            column
            for column in all_columns
            if int(after_nan[column]) != int(before_aligned[column])
        ]

        records.append(
            {
                "function": function_name,
                "new_column_count": len(new_columns),
                "new_columns": ", ".join(new_columns),
                "new_column_nan_cells": int(result[new_columns].isna().sum().sum()),
                "changed_nan_columns": ", ".join(changed_columns),
                "total_nan_cells_after_function": int(result.isna().sum().sum()),
                "rows_with_nan_after_function": int(result.isna().any(axis=1).sum()),
            }
        )

    return result, pd.DataFrame(records)


def make_feature_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row for every feature containing at least one NaN."""
    records: list[dict[str, Any]] = []

    for column in df.columns:
        nan_mask = df[column].isna()
        if not nan_mask.any():
            continue

        affected_positions = np.flatnonzero(nan_mask.to_numpy())
        valid_positions = np.flatnonzero((~nan_mask).to_numpy())
        first_valid_position = int(valid_positions[0]) if len(valid_positions) else None

        records.append(
            {
                "column": column,
                "nan_count": int(nan_mask.sum()),
                "nan_percentage": round(float(nan_mask.mean() * 100), 4),
                "first_nan_datetime": df.loc[nan_mask, "datetime"].min(),
                "last_nan_datetime": df.loc[nan_mask, "datetime"].max(),
                "first_valid_row_position": first_valid_position,
                "nan_positions": affected_positions.tolist(),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(["nan_count", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )


def make_row_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return datetime plus only the names of NaN columns for affected rows."""
    nan_mask = df.isna().any(axis=1)
    affected = df.loc[nan_mask]

    return pd.DataFrame(
        {
            "datetime": affected["datetime"].to_numpy(),
            "nan_count": affected.isna().sum(axis=1).to_numpy(),
            "nan_columns": affected.apply(
                lambda row: row.index[row.isna()].tolist(),
                axis=1,
            ).to_numpy(),
        }
    )


def compare_with_reference(generated: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Compare the reproduced DataFrame to the provided feature CSV."""
    reference = pd.read_csv(path)
    reference["datetime"] = pd.to_datetime(reference["datetime"], errors="raise")
    reference = reference.sort_values("datetime").reset_index(drop=True)

    if generated.shape != reference.shape:
        raise AssertionError(
            f"Shape mismatch: generated={generated.shape}, reference={reference.shape}"
        )

    if generated.columns.tolist() != reference.columns.tolist():
        raise AssertionError("Generated and reference column order do not match.")

    records: list[dict[str, Any]] = []
    for column in generated.columns:
        left = generated[column]
        right = reference[column]

        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            equal = np.isclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-12,
                atol=1e-12,
            )
        else:
            equal = left.eq(right).to_numpy()

        records.append(
            {
                "column": column,
                "matching_rows": int(equal.sum()),
                "different_rows": int((~equal).sum()),
            }
        )

    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the original feature functions and trace every NaN."
    )
    parser.add_argument("--functions", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--report-dir", type=Path, default=Path("nan_reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    namespace = load_functions(args.functions)
    source = prepare_input(args.input)
    generated, function_report = trace_pipeline(source, namespace)
    feature_report = make_feature_report(generated)
    row_report = make_row_report(generated)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    generated.to_csv(args.report_dir / "reproduced_feature_data.csv", index=False)
    function_report.to_csv(args.report_dir / "nan_by_function.csv", index=False)
    feature_report.to_csv(args.report_dir / "nan_by_feature.csv", index=False)
    row_report.to_csv(args.report_dir / "nan_by_row.csv", index=False)

    print(f"Rows: {len(generated)}")
    print(f"Columns: {generated.shape[1]}")
    print(f"Rows containing NaN: {len(row_report)} ({len(row_report) / len(generated):.2%})")
    print(
        "Missing cells: "
        f"{int(generated.isna().sum().sum())}/{generated.size} "
        f"({generated.isna().sum().sum() / generated.size:.2%})"
    )
    print(f"NaN cells after the first 20 rows: {int(generated.iloc[20:].isna().sum().sum())}")

    if args.reference:
        comparison = compare_with_reference(generated, args.reference)
        comparison.to_csv(args.report_dir / "reference_comparison.csv", index=False)
        print(f"Different cells versus reference: {int(comparison['different_rows'].sum())}")

    print(f"Reports written to: {args.report_dir.resolve()}")


if __name__ == "__main__":
    main()