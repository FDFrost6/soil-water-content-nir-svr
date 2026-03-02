"""
Phase 1: NIRQuest Record Ingestion
===================================
Parses all spectral text files from data/records/W{W}N{N}/ subdirectories
and builds a long-format CSV dataset.

Output columns:
  water_content_percent  — integer water level (0/5/15/25/35)
  nitrogen_mg_kg         — integer nitrogen level (0/75/150/300)
  replicate              — 0-based replicate index parsed from filename
  wavelength_nm          — wavelength in nm
  reflectance            — raw reflectance count from spectrometer

Usage:
  python src/ingest_records.py
  (from project root)
"""

import os
import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RECORDS_DIR = Path("data/records")
OUTPUT_CSV = Path("data/soil_spectral_data_records.csv")
SPECTRAL_MARKER = ">>>>>Begin Spectral Data<<<<<"
EXPECTED_PIXELS = 512

# Expected directory name pattern: W{number}N{number}
DIR_PATTERN = re.compile(r"^W(\d+)N(\d+)$")

# Expected filename pattern: soil[_NN]_rep_Reflection__{replicate}__{file_id}.txt
# Accepts both: soil_rep_Reflection__0__0088.txt
#           and: soil_01_rep_Reflection__0__0016.txt
FILE_PATTERN = re.compile(r"soil(?:_\d+)?_rep_Reflection__\d+__(\d+)\.txt$")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_directory_label(dirname: str) -> tuple[int, int]:
    """Extract (water_content_percent, nitrogen_mg_kg) from directory name."""
    m = DIR_PATTERN.match(dirname)
    if m is None:
        raise ValueError(f"Directory name '{dirname}' does not match W{{W}}N{{N}} pattern")
    return int(m.group(1)), int(m.group(2))


def parse_replicate_id(filename: str) -> int:
    """Extract the file_id (global sequential ID) from the filename.
    Using the file_id rather than the within-condition replicate number
    ensures uniqueness even when the same replicate number appears in
    multiple files (e.g. duplicate measurements in W5N0).
    """
    m = FILE_PATTERN.search(filename)
    if m is None:
        raise ValueError(f"Filename '{filename}' does not match expected pattern")
    return int(m.group(1))


def parse_nirquest_file(filepath: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Parse a NIRQuest spectral text file.

    Returns
    -------
    header : dict
        Key-value pairs from the header block.
    wavelengths : np.ndarray of shape (N,)
    reflectances : np.ndarray of shape (N,)
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    header = {}
    spectral_start = None

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if SPECTRAL_MARKER in line_stripped:
            spectral_start = i + 1
            break
        # Parse header key: value lines
        if ":" in line_stripped:
            parts = line_stripped.split(":", 1)
            header[parts[0].strip()] = parts[1].strip()

    if spectral_start is None:
        raise ValueError(f"'{SPECTRAL_MARKER}' not found in {filepath}")

    wavelengths, reflectances = [], []
    for line in lines[spectral_start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            wavelengths.append(float(parts[0]))
            reflectances.append(float(parts[1]))
        except ValueError:
            continue

    if not wavelengths:
        raise ValueError(f"No spectral data found in {filepath}")

    return header, np.array(wavelengths), np.array(reflectances)


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_all(records_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Walk every W{W}N{N} subdirectory under *records_dir*, parse all .txt files,
    and return a long-format DataFrame.
    """
    if not records_dir.exists():
        raise FileNotFoundError(f"Records directory not found: {records_dir}")

    subdirs = sorted(
        [d for d in records_dir.iterdir() if d.is_dir() and DIR_PATTERN.match(d.name)]
    )
    if not subdirs:
        raise ValueError(f"No W{{W}}N{{N}} subdirectories found in {records_dir}")

    if verbose:
        print("=" * 70)
        print("NIRQuest RECORD INGESTION")
        print("=" * 70)
        print(f"Records directory : {records_dir.resolve()}")
        print(f"Subdirectories    : {len(subdirs)}")

    rows = []
    reference_wavelengths = None
    warnings_issued = []
    total_files = 0
    skipped_files = 0

    for subdir in subdirs:
        water_pct, nitrogen_mgkg = parse_directory_label(subdir.name)
        txt_files = sorted(subdir.glob("*.txt"))

        if verbose:
            print(f"\n  {subdir.name:<12} | water={water_pct:>2}%  nitrogen={nitrogen_mgkg:>3} mg/kg | {len(txt_files)} files")

        for filepath in txt_files:
            total_files += 1
            try:
                replicate_id = parse_replicate_id(filepath.name)
                header, wavelengths, reflectances = parse_nirquest_file(filepath)

                # Validate wavelength grid consistency
                if reference_wavelengths is None:
                    reference_wavelengths = wavelengths
                else:
                    if len(wavelengths) != len(reference_wavelengths):
                        msg = (
                            f"  ⚠ Pixel count mismatch in {filepath}: "
                            f"expected {len(reference_wavelengths)}, got {len(wavelengths)}"
                        )
                        warnings_issued.append(msg)
                        if verbose:
                            print(msg)
                        skipped_files += 1
                        continue
                    if not np.allclose(wavelengths, reference_wavelengths, atol=0.01):
                        msg = f"  ⚠ Wavelength grid mismatch in {filepath} (skipped)"
                        warnings_issued.append(msg)
                        if verbose:
                            print(msg)
                        skipped_files += 1
                        continue

                # Validate reflectance values
                if np.any(np.isnan(reflectances)):
                    msg = f"  ⚠ NaN reflectances in {filepath} — NaNs removed"
                    warnings_issued.append(msg)
                    if verbose:
                        print(msg)

                for wl, ref in zip(wavelengths, reflectances):
                    rows.append({
                        "water_content_percent": water_pct,
                        "nitrogen_mg_kg": nitrogen_mgkg,
                        "replicate": replicate_id,
                        "wavelength_nm": wl,
                        "reflectance": ref,
                    })

            except Exception as exc:
                msg = f"  ✗ Error parsing {filepath.name}: {exc}"
                warnings_issued.append(msg)
                if verbose:
                    print(msg)
                skipped_files += 1

    df = pd.DataFrame(rows)

    if verbose:
        print("\n" + "=" * 70)
        print("INGESTION SUMMARY")
        print("=" * 70)
        print(f"  Total files found    : {total_files}")
        print(f"  Files parsed OK      : {total_files - skipped_files}")
        print(f"  Files skipped/errors : {skipped_files}")
        print(f"  Total rows           : {len(df):,}")
        print(f"  Unique measurements  : {df.groupby(['water_content_percent', 'nitrogen_mg_kg', 'replicate']).ngroups}")
        print(f"  Wavelength range     : {df['wavelength_nm'].min():.1f} – {df['wavelength_nm'].max():.1f} nm")
        print(f"  Wavelength points    : {df['wavelength_nm'].nunique()}")
        print(f"\n  Samples per (water%, nitrogen mg/kg):")
        counts = (
            df.groupby(["water_content_percent", "nitrogen_mg_kg"])["replicate"]
            .nunique()
            .reset_index()
            .rename(columns={"replicate": "n_replicates"})
        )
        for _, row in counts.iterrows():
            print(f"    W{int(row.water_content_percent):>2}% N{int(row.nitrogen_mg_kg):>3}: {int(row.n_replicates)} replicates")

        if warnings_issued:
            print(f"\n  Warnings ({len(warnings_issued)}):")
            for w in warnings_issued:
                print(f"    {w}")

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Run quality checks on the ingested DataFrame and print results."""
    print("\n" + "=" * 70)
    print("DATA QUALITY VALIDATION")
    print("=" * 70)

    checks_passed = 0
    checks_failed = 0

    def check(condition: bool, msg_pass: str, msg_fail: str):
        nonlocal checks_passed, checks_failed
        if condition:
            print(f"  ✓ {msg_pass}")
            checks_passed += 1
        else:
            print(f"  ✗ {msg_fail}")
            checks_failed += 1

    # No NaN values
    nan_count = df.isnull().sum().sum()
    check(nan_count == 0, "No NaN values", f"{nan_count} NaN values found")

    # Expected water levels
    expected_water = {0, 5, 15, 25, 35}
    actual_water = set(df["water_content_percent"].unique())
    check(actual_water == expected_water,
          f"Water levels OK: {sorted(actual_water)}",
          f"Unexpected water levels: got {sorted(actual_water)}, expected {sorted(expected_water)}")

    # Expected nitrogen levels
    expected_nitrogen = {0, 75, 150, 300}
    actual_nitrogen = set(df["nitrogen_mg_kg"].unique())
    # W0N0 has only nitrogen=0; all other water levels have 0/75/150/300
    # So we expect at least {0} ⊆ actual_nitrogen
    check(actual_nitrogen.issubset({0, 75, 150, 300}),
          f"Nitrogen levels OK: {sorted(actual_nitrogen)}",
          f"Unexpected nitrogen levels: {sorted(actual_nitrogen)}")

    # Non-negative reflectance (allow slightly negative due to instrument noise)
    very_negative = (df["reflectance"] < -100).sum()
    check(very_negative == 0,
          "No implausibly negative reflectances",
          f"{very_negative} reflectances below -100")

    # Consistent wavelength count per measurement
    wl_counts = df.groupby(["water_content_percent", "nitrogen_mg_kg", "replicate"])["wavelength_nm"].count()
    consistent = wl_counts.nunique() == 1
    check(consistent,
          f"All measurements have {wl_counts.iloc[0]} wavelength points",
          f"Inconsistent wavelength counts: {wl_counts.value_counts().to_dict()}")

    # Replicate count per condition (should be >= 20)
    rep_counts = df.groupby(["water_content_percent", "nitrogen_mg_kg"])["replicate"].nunique()
    all_enough = (rep_counts >= 20).all()
    check(all_enough,
          f"All conditions have >= 20 replicates (min={rep_counts.min()}, max={rep_counts.max()})",
          f"Some conditions have < 20 replicates: {rep_counts[rep_counts < 20].to_dict()}")

    print(f"\n  Results: {checks_passed} passed, {checks_failed} failed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"\nRunning from: {Path.cwd()}\n")

    df = ingest_all(RECORDS_DIR, verbose=True)

    validate_dataset(df)

    print(f"\nSaving to {OUTPUT_CSV} …")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")
    print(f"✓ Saved {len(df):,} rows → {OUTPUT_CSV}")
    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
