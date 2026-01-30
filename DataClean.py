"""
DataClean.py

Cleans WHO Air Quality (AAP) data into a consistent CSV schema used by Lovelace_Algorithm.py.

Supports:
  - WHO AAP CSV like: AAP_2022_city_v9-Table 1.csv
  - WHO AAP Excel workbooks (.xlsx) like: who-aap-database-may2016.xlsx, aap_air_quality_database_2018_v14-*.xlsx

Output:
  cleaned_data.csv (default)

Usage examples:
  # Clean the 2022 CSV
  python3 DataClean.py --input "AAP_2022_city_v9-Table 1.csv" --output cleaned_data.csv

  # Clean a WHO Excel workbook (sheet name may be "database" or "Database")
  python3 DataClean.py --input "aap_air_quality_database_2018_v14-(1).xlsx" --sheet database --output cleaned_data.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd


STANDARD_COL_ORDER = [
    "who_region",
    "iso3",
    "who_country_name",
    "city_or_locality",
    "measurement_year",
    "pm2.5_(μg/m3)",
    "pm10_(μg/m3)",
    "no2_(μg/m3)",
    "pm25_temporal_coverage_(%)",
    "pm10_temporal_coverage_(%)",
    "no2_temporal_coverage_(%)",
    "reference",
    "number_and_type_of_monitoring_stations",
    "version_of_the_database",
    "status",
]


def _standardize_columns(cols) -> list[str]:
    out = []
    for c in cols:
        c = str(c).strip().replace("\n", " ").replace("\t", " ")
        c = c.lower().replace(" ", "_")
        while "__" in c:
            c = c.replace("__", "_")
        out.append(c)
    return out


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "measurement_year" in df.columns:
        df["measurement_year"] = pd.to_numeric(df["measurement_year"], errors="coerce").astype("Int64")

    for col in [
        "pm2.5_(μg/m3)",
        "pm10_(μg/m3)",
        "no2_(μg/m3)",
        "pm25_temporal_coverage_(%)",
        "pm10_temporal_coverage_(%)",
        "no2_temporal_coverage_(%)",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _clean_common(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(how="all").drop_duplicates()
    df = df.dropna(axis=1, how="all")

    df = _coerce_numeric(df)

    # Require year for downstream forecasting
    if "measurement_year" in df.columns:
        df = df.dropna(subset=["measurement_year"])

    # Require at least one PM2.5 field (concentration or coverage)
    keep_if_any = []
    if "pm2.5_(μg/m3)" in df.columns:
        keep_if_any.append("pm2.5_(μg/m3)")
    if "pm25_temporal_coverage_(%)" in df.columns:
        keep_if_any.append("pm25_temporal_coverage_(%)")
    if keep_if_any:
        df = df.dropna(subset=keep_if_any, how="all")

    # Median-impute numeric feature columns except the main target(s)
    targets = {"pm2.5_(μg/m3)", "pm25_temporal_coverage_(%)"}
    for col in df.select_dtypes(include="number").columns:
        if col not in targets:
            df[col] = df[col].fillna(df[col].median())

    # Sort for readability
    sort_cols = [c for c in ["who_country_name", "city_or_locality", "measurement_year"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    return df


def _load_csv_aap_2022(path: Path) -> pd.DataFrame:
    """
    Handles AAP_2022_city_v9-Table 1.csv column names.
    Example headers (as provided earlier):
      WHO Region, ISO3, WHO Country Name, City or Locality, Measurement Year,
      PM2.5 (μg/m3), PM10 (μg/m3), NO2 (μg/m3),
      PM25 temporal coverage (%), PM10 temporal coverage (%), NO2 temporal coverage (%), Reference, ...
    """
    df = pd.read_csv(path)
    original_cols = list(df.columns)

    # Build a case/space-insensitive mapping
    norm = {c: str(c).strip().lower() for c in original_cols}

    def find_col(expected: str) -> str | None:
        expected_norm = expected.strip().lower()
        for c, n in norm.items():
            if n == expected_norm:
                return c
        return None

    out = pd.DataFrame()

    mappings = {
        "who_region": "WHO Region",
        "iso3": "ISO3",
        "who_country_name": "WHO Country Name",
        "city_or_locality": "City or Locality",
        "measurement_year": "Measurement Year",
        "pm2.5_(μg/m3)": "PM2.5 (μg/m3)",
        "pm10_(μg/m3)": "PM10 (μg/m3)",
        "no2_(μg/m3)": "NO2 (μg/m3)",
        "pm25_temporal_coverage_(%)": "PM25 temporal coverage (%)",
        "pm10_temporal_coverage_(%)": "PM10 temporal coverage (%)",
        "no2_temporal_coverage_(%)": "NO2 temporal coverage (%)",
        "reference": "Reference",
        "number_and_type_of_monitoring_stations": "Number and type of monitoring stations",
        "version_of_the_database": "Version of the database",
        "status": "Status",
    }

    for std, exp in mappings.items():
        col = find_col(exp)
        if col is not None:
            out[std] = df[col]

    return out


def _detect_header_row(df_raw: pd.DataFrame, markers: set[str]) -> int:
    max_scan = min(len(df_raw), 60)
    for i in range(max_scan):
        row = df_raw.iloc[i].astype(str).str.strip().str.lower().tolist()
        row_set = set(row)
        hits = sum(1 for m in markers if m in row_set)
        if hits >= max(3, len(markers) // 2):
            return i
    return -1


def _load_excel_who(path: Path, sheet: str) -> pd.DataFrame:
    """
    Loads a WHO AAP Excel workbook where the true header row might not be row 0.
    """
    df_raw = pd.read_excel(path, sheet_name=sheet, header=None)
    header_row = _detect_header_row(df_raw, {"region", "iso3", "country", "year"})
    if header_row == -1:
        # Fall back to header=0
        df = pd.read_excel(path, sheet_name=sheet, header=0)
    else:
        df = pd.read_excel(path, sheet_name=sheet, header=header_row)

    df = df.dropna(axis=1, how="all")
    df.columns = _standardize_columns(df.columns)

    def pick(*candidates: str) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    out = pd.DataFrame()

    col_region = pick("region", "who_region")
    col_iso3 = pick("iso3")
    col_country = pick("country", "who_country_name")
    col_city = pick("city_or_locality", "city_town", "city", "town", "city/town")
    col_year = pick("year", "measurement_year")

    if col_region: out["who_region"] = df[col_region]
    if col_iso3: out["iso3"] = df[col_iso3]
    if col_country: out["who_country_name"] = df[col_country]
    if col_city: out["city_or_locality"] = df[col_city]
    if col_year: out["measurement_year"] = df[col_year]

    # find pollutant columns (varies by workbook)
    for c in df.columns:
        if "pm2.5" in c or "pm25" in c:
            if "temporal" in c and "coverage" in c:
                out["pm25_temporal_coverage_(%)"] = df[c]
            elif "annual" in c or "mean" in c or c in {"pm2.5", "pm25"}:
                out["pm2.5_(μg/m3)"] = df[c]

        if "pm10" in c:
            if "temporal" in c and "coverage" in c:
                out["pm10_temporal_coverage_(%)"] = df[c]
            elif "annual" in c or "mean" in c or c == "pm10":
                out["pm10_(μg/m3)"] = df[c]

        if "no2" in c:
            if "temporal" in c and "coverage" in c:
                out["no2_temporal_coverage_(%)"] = df[c]
            elif "annual" in c or "mean" in c or c == "no2":
                out["no2_(μg/m3)"] = df[c]

    if "reference" in df.columns:
        out["reference"] = df["reference"]
    if "number_and_type_of_monitoring_stations" in df.columns:
        out["number_and_type_of_monitoring_stations"] = df["number_and_type_of_monitoring_stations"]
    if "version_of_the_database" in df.columns:
        out["version_of_the_database"] = df["version_of_the_database"]
    if "status" in df.columns:
        out["status"] = df["status"]

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input WHO AAP data file (.csv or .xlsx).")
    ap.add_argument("--sheet", default="database", help="Excel sheet name (only used for .xlsx).")
    ap.add_argument("--output", default="cleaned_data.csv", help="Output cleaned CSV path.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path.resolve()}", file=sys.stderr)
        return 1

    try:
        if in_path.suffix.lower() == ".csv":
            df = _load_csv_aap_2022(in_path)
        elif in_path.suffix.lower() in {".xlsx", ".xlsm"}:
            # try the given sheet; if it fails, print available sheets
            try:
                df = _load_excel_who(in_path, args.sheet)
            except Exception:
                xls = pd.ExcelFile(in_path)
                print("ERROR: Failed to read the requested sheet.", file=sys.stderr)
                print("Available sheets:", xls.sheet_names, file=sys.stderr)
                raise
        else:
            print("ERROR: Unsupported file type. Use .csv or .xlsx", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"ERROR: Could not load input file: {e}", file=sys.stderr)
        return 1

    df = _clean_common(df)

    # Ensure stable column order (keep extras at end if present)
    cols = [c for c in STANDARD_COL_ORDER if c in df.columns] + [c for c in df.columns if c not in STANDARD_COL_ORDER]
    df = df[cols]

    out_path = Path(args.output)
    df.to_csv(out_path, index=False)

    print("Saved:", out_path.resolve())
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
