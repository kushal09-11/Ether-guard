"""
Automated unit tests for the Etherscan Feature Compatibility Audit and Common Feature Schema.
Validates report structure, key completeness, schema formatting, and dataset compatibility.
"""

from pathlib import Path
import pytest
import pandas as pd
import json

import config
from src.feature_audit import generate_feature_audit_reports, FEATURE_AUDIT_DATA
from src.utils import load_json


def test_feature_audit_report_generation(tmp_path: Path):
    """Verifies that all 3 audit artifacts are generated and valid."""
    csv_file = tmp_path / "audit.csv"
    md_file = tmp_path / "audit.md"
    schema_file = tmp_path / "schema.json"

    generate_feature_audit_reports(
        csv_out_path=csv_file,
        md_out_path=md_file,
        schema_out_path=schema_file,
    )

    assert csv_file.exists() and csv_file.stat().st_size > 0
    assert md_file.exists() and md_file.stat().st_size > 0
    assert schema_file.exists() and schema_file.stat().st_size > 0


def test_audit_csv_contains_all_51_raw_features(tmp_path: Path):
    """Verifies that the audit table covers exactly 51 columns matching raw dataset schema."""
    csv_file = tmp_path / "audit.csv"
    generate_feature_audit_reports(csv_out_path=csv_file, md_out_path=tmp_path / "a.md", schema_out_path=tmp_path / "s.json")

    audit_df = pd.read_csv(csv_file)
    assert len(audit_df) == 51
    assert "csv_column" in audit_df.columns
    assert "reproducibility" in audit_df.columns
    assert "additional_endpoint" in audit_df.columns


def test_common_feature_schema_structure(tmp_path: Path):
    """
    Validates that every feature in common_feature_schema.json strictly conforms
    to the 9 required specification attributes.
    """
    schema_file = tmp_path / "test_schema.json"
    generate_feature_audit_reports(
        csv_out_path=tmp_path / "a.csv",
        md_out_path=tmp_path / "a.md",
        schema_out_path=schema_file,
    )

    schema_data = load_json(schema_file)
    assert "features" in schema_data
    assert schema_data["total_recommended_features"] == len(schema_data["features"])

    required_keys = [
        "canonical_feature_name",
        "data_type",
        "unit",
        "calculation_formula",
        "required_source_fields",
        "missing_value_policy",
        "observation_period",
        "training_dataset_mapping",
        "etherscan_data_mapping",
    ]

    for feat in schema_data["features"]:
        for k in required_keys:
            assert k in feat, f"Missing required key '{k}' in feature {feat.get('canonical_feature_name')}"
        assert feat["data_type"] in ["float64", "int64"]
        assert isinstance(feat["required_source_fields"], list)
        assert len(feat["required_source_fields"]) > 0
        assert "endpoint" in feat["etherscan_data_mapping"]


def test_schema_features_present_in_cleaned_dataset():
    """
    Verifies that all recommended common features exist within the cleaned dataset.
    Ensures zero train-inference feature disparity.
    """
    if not config.COMMON_FEATURE_SCHEMA_PATH.exists() or not config.PROCESSED_DATASET_PATH.exists():
        pytest.skip("Artifacts not yet generated on disk.")

    schema = load_json(config.COMMON_FEATURE_SCHEMA_PATH)
    cleaned_df = pd.read_csv(config.PROCESSED_DATASET_PATH, nrows=5)

    for feat in schema["features"]:
        feat_name = feat["canonical_feature_name"]
        assert feat_name in cleaned_df.columns, (
            f"Schema feature '{feat_name}' is missing from cleaned dataset columns!"
        )
