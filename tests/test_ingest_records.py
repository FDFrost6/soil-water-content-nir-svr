"""
Unit tests for src/ingest_records.py
=====================================
Tests file parsing, label extraction, and ingestion quality checks.

Run:
  venv/bin/python -m pytest tests/test_ingest_records.py -v
"""

import sys
import os
import re
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingest_records import (
    parse_directory_label,
    parse_replicate_id,
    parse_nirquest_file,
    ingest_all,
    FILE_PATTERN,
    DIR_PATTERN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_FILE_CONTENT = textwrap.dedent("""\
    Data from soil_rep_Reflection__0__0001.txt Node

    Date: Wed Feb 18 15:33:16 CET 2026
    User: root
    Spectrometer: NQ5500316
    Trigger mode: 0
    Integration Time (sec): 1.000000E-1
    Scans to average: 25
    Nonlinearity correction enabled: true
    Boxcar width: 1
    Storing dark spectrum: false
    XAxis mode: Wavelengths
    Number of Pixels in Spectrum: 3
    >>>>>Begin Spectral Data<<<<<
    900.0\t100.0
    950.0\t110.0
    1000.0\t120.0
""")


@pytest.fixture
def tmp_record_file(tmp_path):
    """Write a minimal NIRQuest file and return its path."""
    f = tmp_path / "soil_rep_Reflection__0__0001.txt"
    f.write_text(MINIMAL_FILE_CONTENT)
    return f


@pytest.fixture
def tmp_records_dir(tmp_path):
    """
    Build a minimal records directory:
      W15N0/   — 2 standard files  (soil_rep_*  naming)
      W5N0/    — 2 alternate files (soil_01_rep_* naming)
      W0N0/    — 2 standard files
    """
    conditions = [
        ("W15N0",  "soil_rep_Reflection__{r}__00{r:02d}.txt"),
        ("W5N0",   "soil_01_rep_Reflection__{r}__00{r:02d}.txt"),
        ("W0N0",   "soil_rep_Reflection__{r}__01{r:02d}.txt"),
    ]
    for dirname, tmpl in conditions:
        d = tmp_path / dirname
        d.mkdir()
        for rep in range(2):
            fname = tmpl.format(r=rep)
            (d / fname).write_text(
                MINIMAL_FILE_CONTENT.replace(
                    "soil_rep_Reflection__0__0001.txt Node",
                    f"{fname} Node"
                )
            )
    return tmp_path


# ---------------------------------------------------------------------------
# parse_directory_label
# ---------------------------------------------------------------------------

class TestParseDirectoryLabel:
    def test_standard(self):
        assert parse_directory_label("W15N0") == (15, 0)

    def test_zero_water(self):
        assert parse_directory_label("W0N0") == (0, 0)

    def test_nitrogen_300(self):
        assert parse_directory_label("W35N300") == (35, 300)

    def test_w5n75(self):
        assert parse_directory_label("W5N75") == (5, 75)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_directory_label("bad_directory")

    def test_no_match_raises(self):
        with pytest.raises(ValueError):
            parse_directory_label("W15")


# ---------------------------------------------------------------------------
# parse_replicate_id (now returns file_id)
# ---------------------------------------------------------------------------

class TestParseReplicateId:
    def test_standard_filename(self):
        # soil_rep_Reflection__0__0088.txt → file_id = 88
        assert parse_replicate_id("soil_rep_Reflection__0__0088.txt") == 88

    def test_alternate_01_prefix(self):
        # soil_01_rep_Reflection__0__0016.txt → file_id = 16
        assert parse_replicate_id("soil_01_rep_Reflection__0__0016.txt") == 16

    def test_alternate_02_prefix(self):
        # soil_02_rep_Reflection__5__0033.txt → file_id = 33
        assert parse_replicate_id("soil_02_rep_Reflection__5__0033.txt") == 33

    def test_invalid_filename_raises(self):
        with pytest.raises(ValueError):
            parse_replicate_id("unrecognised_file.txt")

    def test_file_pattern_accepts_both_prefixes(self):
        assert FILE_PATTERN.search("soil_rep_Reflection__0__0001.txt") is not None
        assert FILE_PATTERN.search("soil_01_rep_Reflection__0__0001.txt") is not None
        assert FILE_PATTERN.search("soil_02_rep_Reflection__0__0001.txt") is not None

    def test_file_pattern_rejects_garbage(self):
        assert FILE_PATTERN.search("garbage.txt") is None


# ---------------------------------------------------------------------------
# parse_nirquest_file
# ---------------------------------------------------------------------------

class TestParseNirquestFile:
    def test_parses_header_and_spectra(self, tmp_record_file):
        header, wl, ref = parse_nirquest_file(tmp_record_file)
        assert "Spectrometer" in header
        assert len(wl) == 3
        assert len(ref) == 3

    def test_wavelength_values(self, tmp_record_file):
        _, wl, _ = parse_nirquest_file(tmp_record_file)
        np.testing.assert_allclose(wl, [900.0, 950.0, 1000.0])

    def test_reflectance_values(self, tmp_record_file):
        _, _, ref = parse_nirquest_file(tmp_record_file)
        np.testing.assert_allclose(ref, [100.0, 110.0, 120.0])

    def test_missing_marker_raises(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("No marker here\n900.0\t100.0\n")
        with pytest.raises(ValueError, match="not found"):
            parse_nirquest_file(f)

    def test_empty_spectral_section_raises(self, tmp_path):
        f = tmp_path / "empty_spec.txt"
        f.write_text(">>>>>Begin Spectral Data<<<<<\n\n")
        with pytest.raises(ValueError, match="No spectral data"):
            parse_nirquest_file(f)


# ---------------------------------------------------------------------------
# ingest_all (integration)
# ---------------------------------------------------------------------------

class TestIngestAll:
    def test_returns_dataframe(self, tmp_records_dir):
        df = ingest_all(tmp_records_dir, verbose=False)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, tmp_records_dir):
        df = ingest_all(tmp_records_dir, verbose=False)
        assert set(df.columns) == {
            "water_content_percent", "nitrogen_mg_kg", "replicate",
            "wavelength_nm", "reflectance",
        }

    def test_labels_correct(self, tmp_records_dir):
        df = ingest_all(tmp_records_dir, verbose=False)
        assert set(df["water_content_percent"].unique()) == {0, 5, 15}
        assert set(df["nitrogen_mg_kg"].unique()) == {0}

    def test_no_nans(self, tmp_records_dir):
        df = ingest_all(tmp_records_dir, verbose=False)
        assert df.isnull().sum().sum() == 0

    def test_wavelength_count(self, tmp_records_dir):
        df = ingest_all(tmp_records_dir, verbose=False)
        # 3 conditions × 2 reps × 3 wavelengths = 18 rows
        assert len(df) == 18

    def test_alternate_prefix_included(self, tmp_records_dir):
        """Files with soil_01_rep_* naming (in W5N0) must be parsed."""
        df = ingest_all(tmp_records_dir, verbose=False)
        w5n0 = df[(df["water_content_percent"] == 5) & (df["nitrogen_mg_kg"] == 0)]
        assert len(w5n0) > 0, "W5N0 (soil_01_rep_* files) should be included"

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest_all(tmp_path / "nonexistent", verbose=False)

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError):
            ingest_all(tmp_path, verbose=False)


# ---------------------------------------------------------------------------
# Regression: real dataset (only if CSV already exists)
# ---------------------------------------------------------------------------

REAL_CSV = Path(__file__).parent.parent / "data" / "soil_spectral_data_records.csv"


@pytest.mark.skipif(not REAL_CSV.exists(), reason="Real dataset not yet generated")
class TestRealDataset:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(REAL_CSV)

    def test_no_nans(self, df):
        assert df.isnull().sum().sum() == 0

    def test_water_levels(self, df):
        assert set(df["water_content_percent"].unique()) == {0, 5, 15, 25, 35}

    def test_nitrogen_levels(self, df):
        assert set(df["nitrogen_mg_kg"].unique()).issubset({0, 75, 150, 300})

    def test_wavelength_points(self, df):
        assert df["wavelength_nm"].nunique() == 512

    def test_total_samples(self, df):
        n_meas = df.groupby(["water_content_percent", "nitrogen_mg_kg", "replicate"]).ngroups
        assert n_meas >= 300  # at least 300 unique measurements

    def test_min_replicates_per_condition(self, df):
        counts = df.groupby(["water_content_percent", "nitrogen_mg_kg"])["replicate"].nunique()
        assert (counts >= 20).all()
