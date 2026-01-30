from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path("cleaned_data.csv")

# ISO3 buckets to split WHO "Americas" into North vs South America 
NORTH_AMERICA_ISO3 = {
    # North America
    "CAN", "USA", "MEX",
    # Central America
    "BLZ", "GTM", "HND", "SLV", "NIC", "CRI", "PAN",
    # Caribbean
    "ATG", "BHS", "BRB", "CUB", "DMA", "DOM", "GRD", "HTI", "JAM", "KNA", "LCA", "VCT", "TTO",
    # Territories
    "PRI", "BMU", "VIR", "VGB", "TCA", "AIA", "ABW", "CUW", "SXM", "GLP", "MTQ", "MSR",
}

SOUTH_AMERICA_ISO3 = {
    "ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUY", "PRY", "PER", "SUR", "URY", "VEN",
}

OCEANIA_ISO3 = {
    "AUS", "NZL", "PNG", "FJI", "SLB", "VUT", "WSM", "TON", "KIR", "TUV", "NRU",
    "MHL", "FSM", "PLW",
}

CONTINENTS = ["Africa", "Antarctica", "Asia", "Europe", "North America", "South America", "Oceania"]
CONTINENT_SET = set(CONTINENTS)


def fit_simple_lr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Fit y = b0 + b1*x using closed-form math:

      b1 = (n*Σ(xy) - Σx*Σy) / (n*Σ(x^2) - (Σx)^2)
      b0 = ȳ - b1*x̄
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 points to fit a line.")

    Sx = np.sum(x)
    Sy = np.sum(y)
    Sxx = np.sum(x * x)
    Sxy = np.sum(x * y)

    denom = (n * Sxx) - (Sx * Sx)
    if denom == 0:
        raise ValueError("Cannot fit: all years are identical.")

    b1 = ((n * Sxy) - (Sx * Sy)) / denom
    b0 = (Sy / n) - b1 * (Sx / n)
    return float(b0), float(b1)


def predict(b0: float, b1: float, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return b0 + b1 * x


def clamp_if_coverage(target_col: str, values: np.ndarray) -> np.ndarray:
    """
    For PM2.5 temporal coverage (%), predictions should be within [0, 100].
    This prevents impossible values (e.g., >100% coverage) when forecasting.
    """
    arr = np.asarray(values, dtype=float)
    if target_col == "pm25_temporal_coverage_(%)":
        return np.clip(arr, 0.0, 100.0)
    return arr


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - sse / sst) if sst != 0 else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def _norm_region(who_region: str | float) -> str:
    """
    Normalize WHO region strings such as:
      - "European Region"
      - "Region of the Americas"
      - "South East Asia Region"
      - "Western Pacific Region"
      - "Eastern Mediterranean Region"
      - "African Region"
    """
    if pd.isna(who_region):
        return ""
    r = str(who_region).strip().lower()
    r = r.replace("_", " ")
    r = " ".join(r.split()) 
    return r


def to_continent(who_region: str | float, iso3: str | float) -> str | None:
    """
    Map WHO region + ISO3 into a 7-continent label.
    Returns None if it cannot be mapped.
    """
    r = _norm_region(who_region)
    code = "" if pd.isna(iso3) else str(iso3).strip().upper()

    if "afric" in r:
        return "Africa"
    if "europe" in r:
        return "Europe"

    # WHO Americas
    if "americ" in r:
        if code in SOUTH_AMERICA_ISO3:
            return "South America"
        return "North America"

    # Asia-related WHO regions
    if "south east asia" in r or "south-east asia" in r or "southeast asia" in r:
        return "Asia"
    if "eastern mediterranean" in r:
        return "Asia"

    # Western Pacific which is Asia/Oceania
    if "western pacific" in r:
        if code in OCEANIA_ISO3:
            return "Oceania"
        return "Asia"

    # Antarctica 
    if "antarctica" in r:
        return "Antarctica"

    return None


def load_continent_panel() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError("cleaned_data.csv not found. Run DataClean.py first.")

    df = pd.read_csv(CSV_PATH)

    required = {"who_region", "iso3", "measurement_year"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"cleaned_data.csv missing required columns: {missing}")

    df["measurement_year"] = pd.to_numeric(df["measurement_year"], errors="coerce")

    # Targets
    for col in ["pm25_temporal_coverage_(%)", "pm2.5_(μg/m3)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["continent"] = df.apply(lambda row: to_continent(row["who_region"], row.get("iso3", "")), axis=1)
    df = df.dropna(subset=["continent", "measurement_year"])
    df = df[df["continent"].isin(CONTINENT_SET)]

    panel = (
        df.groupby(["continent", "measurement_year"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["continent", "measurement_year"])
    )
    return panel


def choose_continent(panel: pd.DataFrame, text: str) -> str:
    t = text.strip().lower()
    options = sorted(panel["continent"].dropna().unique().tolist())

    for o in options:
        if o.lower() == t:
            return o

    matches = [o for o in options if t in o.lower()]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(f"Continent '{text}' not found (or ambiguous). Options: {', '.join(options)}")


def fit_group(panel: pd.DataFrame, group: str, target_col: str) -> dict:
    sub = panel[panel["continent"] == group].copy()
    sub = sub.dropna(subset=[target_col])

    if len(sub) < 6:
        raise ValueError(
            f"Not enough continent-year points for '{group}' to fit a forecast model "
            f"(need ~6+, found {len(sub)})."
        )

    years = sorted(sub["measurement_year"].unique().tolist())
    holdout_years = years[-3:] if len(years) >= 6 else years[-2:]

    train = sub[~sub["measurement_year"].isin(holdout_years)]
    test = sub[sub["measurement_year"].isin(holdout_years)]

    b0, b1 = fit_simple_lr(
        train["measurement_year"].to_numpy(dtype=float),
        train[target_col].to_numpy(dtype=float),
    )

    y_test = test[target_col].to_numpy(dtype=float)
    y_hat = clamp_if_coverage(target_col, predict(b0, b1, test["measurement_year"].to_numpy(dtype=float)))
    return {
        "group": group,
        "target_col": target_col,
        "b0": b0,
        "b1": b1,
        "history": sub.sort_values("measurement_year"),
        "holdout_years": holdout_years,
        "r2": r2(y_test, y_hat),
        "mae": mae(y_test, y_hat),
    }


def observed_value_for_year_or_latest(history: pd.DataFrame, target_col: str, year: int) -> tuple[int, float]:
    hist = history.dropna(subset=[target_col]).copy()
    hist["measurement_year"] = hist["measurement_year"].astype(int)

    if (hist["measurement_year"] == year).any():
        val = float(hist.loc[hist["measurement_year"] == year, target_col].iloc[-1])
        return year, val

    prior = hist[hist["measurement_year"] <= year]
    if len(prior) == 0:
        y0 = int(hist["measurement_year"].min())
        v0 = float(hist.loc[hist["measurement_year"] == y0, target_col].iloc[-1])
        return y0, v0

    y_latest = int(prior["measurement_year"].max())
    v_latest = float(prior.loc[prior["measurement_year"] == y_latest, target_col].iloc[-1])
    return y_latest, v_latest


def plot_fit_and_forecast(model: dict, year: int, outpath: Path) -> float:
    hist = model["history"]
    years_obs = hist["measurement_year"].to_numpy(dtype=int)
    y_obs = hist[model["target_col"]].to_numpy(dtype=float)

    min_year = int(np.min(years_obs))
    max_year = int(np.max(years_obs))
    grid_end = max(max_year + 10, year)

    grid = np.arange(min_year, grid_end + 1)
    y_line = clamp_if_coverage(model["target_col"], predict(model["b0"], model["b1"], grid))
    y_year = float(clamp_if_coverage(model["target_col"], predict(model["b0"], model["b1"], np.array([year])))[0])
    plt.figure()
    plt.scatter(years_obs, y_obs)
    plt.plot(grid, y_line)

    # Label every observed dot with its year
    for x, y in zip(years_obs, y_obs):
        if np.isfinite(y):
            plt.annotate(
                str(int(x)),
                (float(x), float(y)),
                textcoords="offset points",
                xytext=(6, 6),
                ha="left",
                fontsize=8,
            )

    # Forecast point (requested year)
    plt.scatter([year], [y_year], marker="x")
    plt.annotate(
        f"Pred {year}",
        (float(year), float(y_year)),
        textcoords="offset points",
        xytext=(8, -10),
        ha="left",
        fontsize=9,
    )

    plt.xlabel("Year")
    plt.ylabel(model["target_col"])
    plt.title(f"{model['target_col']} vs Year: {model['group']}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()

    return y_year



def plot_holdout(model: dict, outpath: Path) -> None:
    hist = model["history"]
    test = hist[hist["measurement_year"].isin(model["holdout_years"])].copy()

    years = test["measurement_year"].to_numpy(dtype=int)
    y_true = test[model["target_col"]].to_numpy(dtype=float)
    y_pred = clamp_if_coverage(model["target_col"], predict(model["b0"], model["b1"], years.astype(float)))
    # Unit/label helper for clearer annotations
    if model["target_col"] == "pm25_temporal_coverage_(%)":
        unit = "%"
        short_name = "PM2.5 coverage"
    elif model["target_col"] == "pm2.5_(μg/m3)":
        unit = "μg/m³"
        short_name = "PM2.5"
    else:
        unit = ""
        short_name = model["target_col"]

    plt.figure()
    plt.scatter(y_true, y_pred)

    # Reference line: perfect predictions (y = x)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx])

    # Label every dot with year + actual/pred values (so it's not just "93", "94", etc.)
    for a, p, yr in zip(y_true, y_pred, years):
        label = f"{yr}: A={a:.1f}{unit}, P={p:.1f}{unit}"
        plt.annotate(
            label,
            (a, p),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            fontsize=8,
        )

    plt.xlabel(f"Actual {short_name} ({unit})" if unit else f"Actual {short_name}")
    plt.ylabel(f"Predicted {short_name} ({unit})" if unit else f"Predicted {short_name}")
    plt.title(f"Holdout: {model['group']}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()




def save_next_10_years_csv(model: dict, outpath: Path) -> None:
    """
    Save a simple 10-year forecast table starting from the last observed year + 1.
    """
    hist = model["history"]
    max_year = int(hist["measurement_year"].max())
    future_years = list(range(max_year + 1, max_year + 11))

    preds = clamp_if_coverage(model["target_col"], predict(model["b0"], model["b1"], np.array(future_years, dtype=float)))
    out = pd.DataFrame({
        "continent": model["group"],
        "measurement_year": future_years,
        model["target_col"] + "_forecast": preds,
    })
    out.to_csv(outpath, index=False)



def main() -> int:
    try:
        panel = load_continent_panel()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    default_target = "pm25_temporal_coverage_(%)"
    alt_target = "pm2.5_(μg/m3)"

    print("WHO Air Quality Forecast (7 Continents)")
    print("Targets:")
    print(f"  1) {default_target}")
    print(f"  2) {alt_target}\n")

    target_choice = input("Choose target [1/2] (press Enter for 1): ").strip() or "1"
    target_col = default_target if target_choice == "1" else alt_target

    if target_col not in panel.columns:
        print(f"ERROR: Target column '{target_col}' not found in cleaned_data.csv", file=sys.stderr)
        return 1

    options = sorted(panel["continent"].dropna().unique().tolist())
    print(f"Available continents: {', '.join(options)}\n")

    try:
        group = choose_continent(panel, input("Continent: "))
        year = int(input("Year to predict (e.g., 2030): ").strip())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    try:
        model = fit_group(panel, group, target_col)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    hist_years = model["history"]["measurement_year"].astype(int)
    min_year, max_year = int(hist_years.min()), int(hist_years.max())

    current_year = datetime.now().year
    obs_year, obs_val = observed_value_for_year_or_latest(model["history"], target_col, current_year)
    pred_val = float(clamp_if_coverage(target_col, predict(model["b0"], model["b1"], np.array([year])))[0])
    print("\n--- Summary ---")
    print(f"Continent: {model['group']}")
    print(f"Years in data: {min_year}–{max_year} (holdout: {model['holdout_years']})")
    print(f"Holdout R²: {model['r2']:.3f}")
    print(f"Holdout MAE: {model['mae']:.3f}")

    if obs_year == current_year:
        print(f"\nObserved for {current_year}: {obs_val:.3f}  (column: {target_col})")
    else:
        print(f"\nObserved for {current_year}: not available in dataset")
        print(f"Latest observed year ≤ {current_year}: {obs_year} value = {obs_val:.3f}  (column: {target_col})")

    print(f"Predicted for {year}: {pred_val:.3f}  (column: {target_col})")

    out_plot1 = Path("continent_fit_and_forecast.png")
    out_plot2 = Path("holdout_actual_vs_pred.png")
    out_csv = Path("forecast_next_10_years.csv")

    _ = plot_fit_and_forecast(model, year, out_plot1)
    plot_holdout(model, out_plot2)
    save_next_10_years_csv(model, out_csv)

    print("\nSaved outputs:")
    print(f"- {out_plot1.resolve()}")
    print(f"- {out_plot2.resolve()}")
    print(f"- {out_csv.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
