"""
Utility functions for Ethereum Wallet Fraud Detection (Phase 1).
Provides dataset discovery, wallet address masking, JSON serialization, and column normalization.
"""

from pathlib import Path
import json
from typing import Any, Optional
import config


def find_raw_dataset(specified_path: Optional[Path] = None) -> Path:
    """
    Locates the raw Ethereum fraud detection dataset CSV.
    If the file is missing, raises an informative FileNotFoundError with
    exact placement instructions rather than silently fabricating synthetic data.
    """
    if specified_path is not None:
        if specified_path.is_file():
            return specified_path
        # If caller explicitly gave a path that doesn't exist, raise error immediately
        missing_target = specified_path
    else:
        if config.RAW_DATASET_PATH.is_file():
            return config.RAW_DATASET_PATH
        # Check if any .csv file exists in data/raw/
        raw_csvs = list(config.RAW_DATA_DIR.glob("*.csv"))
        if raw_csvs:
            return raw_csvs[0]
        missing_target = config.RAW_DATASET_PATH

    # File missing: Stop the data-dependent pipeline and report instructions
    error_message = (
        f"\n{'='*70}\n"
        f"DATASET MISSING ERROR:\n"
        f"The raw Ethereum fraud detection dataset CSV was not found at:\n"
        f"  {missing_target.resolve()}\n\n"
        f"INSTRUCTIONS TO RESUME:\n"
        f"1. Download the 'Ethereum Fraud Detection Dataset' from Kaggle:\n"
        f"   https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset\n"
        f"2. Place the downloaded CSV file into:\n"
        f"   {config.RAW_DATA_DIR.resolve()}\n"
        f"3. Ensure the file is named '{config.DEFAULT_RAW_CSV_NAME}' (or any .csv in that folder).\n"
        f"4. Re-run the inspection or preprocessing script.\n"
        f"{'='*70}"
    )
    raise FileNotFoundError(error_message)


def mask_address(address: str, prefix_len: int = 6, suffix_len: int = 4) -> str:
    """
    Safely masks an Ethereum wallet address (e.g., '0x0000927777...05e8')
    to prevent exposing raw identifiers in reports while retaining inspectability.
    """
    if not isinstance(address, str):
        return str(address)
    address = address.strip()
    if len(address) <= (prefix_len + suffix_len):
        return address
    return f"{address[:prefix_len]}...{address[-suffix_len:]}"


def clean_column_name(col_name: str) -> str:
    """
    Normalizes a column name by trimming whitespace, lowercasing,
    replacing spaces/special characters with underscores, and removing unclosed brackets.
    """
    cleaned = col_name.strip().lower()
    cleaned = cleaned.replace("(", "").replace(")", "").replace(".", "_")
    cleaned = "_".join(cleaned.split())
    return cleaned


def save_json(data: Any, file_path: Path, indent: int = 2) -> None:
    """Saves a Python dictionary or list to a JSON file with formatting."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(file_path: Path) -> Any:
    """Loads a JSON file from disk."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def dataframe_to_markdown(df_records: list[dict[str, Any]], columns: Optional[list[str]] = None) -> str:
    """
    Renders a list of dictionaries as a GitHub-flavored Markdown table
    without requiring third-party libraries like tabulate.
    """
    if not df_records:
        return "_No records available._"

    cols = columns if columns is not None else list(df_records[0].keys())
    headers = [str(c) for c in cols]
    
    rows = []
    for r in df_records:
        rows.append([str(r.get(c, "")) for c in cols])

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            widths[idx] = max(widths[idx], len(val))

    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]

    return "\n".join([header_line, separator_line] + data_lines)

