"""
Automated unit tests for data inspection module.
Validates metric extraction, address masking, duplicate counting, and report generation.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.utils import mask_address, dataframe_to_markdown
from src.data_inspection import inspect_dataset, generate_inspection_markdown_report


def test_mask_address():
    """Verifies that Ethereum addresses are masked properly."""
    addr = "0x00009277775ac7d0d59eaad8fee3d10ac6c805e8"
    masked = mask_address(addr)
    assert masked == "0x0000...05e8"
    assert len(masked) == 13

    # Short string edge case
    assert mask_address("0x1234") == "0x1234"


def test_dataframe_to_markdown():
    """Verifies that markdown table generator outputs proper Markdown structure."""
    records = [
        {"col_a": "val1", "col_b": 10},
        {"col_a": "val2", "col_b": 20},
    ]
    md = dataframe_to_markdown(records)
    assert "| col_a | col_b |" in md
    assert "| val1  | 10    |" in md


def test_inspect_dataset_on_synthetic_data(tmp_path: Path):
    """Verifies inspection metrics on a controlled synthetic dataset."""
    synthetic_csv = tmp_path / "test_raw.csv"
    df = pd.DataFrame({
        "Unnamed: 0": [0, 1],
        "Index": [1, 2],
        "Address": [
            "0x0000000000000000000000000000000000000001",
            "0x0000000000000000000000000000000000000002",
        ],
        "FLAG": [0, 1],
        "Sent tnx": [10, 5],
        "Received Tnx": [20, 10],
        "Number of Created Contracts": [0, 0],
        "total transactions (including tnx to create contract": [30, 15],
        "total ether balance": [5.0, -10.0],  # Negative balance
        "constant_col": [0.0, 0.0],
    })
    df.to_csv(synthetic_csv, index=False)

    metrics = inspect_dataset(synthetic_csv)
    assert metrics["n_rows"] == 2
    assert metrics["n_cols"] == 10
    assert metrics["target_counts"] == {0: 1, 1: 1}
    assert metrics["negative_counts"]["total ether balance"]["count"] == 1
    assert any(c["column"] == "constant_col" for c in metrics["constant_cols"])

    # Test report generation
    out_report = tmp_path / "report.md"
    generate_inspection_markdown_report(metrics, out_report)
    assert out_report.exists()
    assert "# Dataset Inspection & Data Quality Audit Report" in out_report.read_text(encoding="utf-8")
