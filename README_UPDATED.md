# Lovelace Algorithm — WHO Air Quality Forecast (7 Continents)

## Project Summary
This project implements a **simple machine learning model (linear regression)** in Python and applies it to the **World Health Organization (WHO) Air Quality Database**. The model learns a relationship between **measurement year** and an air-quality variable, then uses that relationship to estimate values for a user-selected year. The dataset is global (multiple regions worldwide), which supports discussion about **representation, data gaps, and inequality**.

The regression model is:

**y = b0 + b1·year**

By default, this project predicts **PM2.5 temporal coverage (%)**. It can also predict **PM2.5 concentration (μg/m³)**.

To reduce missing-data issues for individual countries, the model aggregates records into the **7 continents** (Africa, Asia, Europe, North America, South America, Oceania; Antarctica is included as a category but typically absent in the dataset).

---

## Files
- `DataClean.py` — converts a raw WHO file (CSV or Excel) into `cleaned_data.csv`
- `cleaned_data.csv` — cleaned dataset used by the model (generated)
- `Lovelace_Algorithm.py` — main program (trains, evaluates, predicts, and creates plots)
- `continent_fit_and_forecast.png` — trend line with labeled points + prediction point
- `holdout_actual_vs_pred.png` — evaluation scatter (labeled with year, actual, predicted)
- `forecast_next_10_years.csv` — 10-year forecast table (starting after the last observed year)
- `REFLECTION.md` — written reflection (submitted in Canvas; stored here for completeness)

---

## Dependencies
Install packages:

```bash
python3 -m pip install numpy pandas matplotlib openpyxl
```

---

## How to Run

### 1) Clean the dataset

If you have a WHO CSV (example):

```bash
python3 DataClean.py --input "AAP_2022_city_v9-Table 1.csv" --output cleaned_data.csv
```

If you have a WHO Excel workbook (example):

```bash
python3 DataClean.py --input "aap_air_quality_database_2018_v14-(1).xlsx" --sheet database --output cleaned_data.csv
```

### 2) Train + predict + generate plots

```bash
python3 Lovelace_Algorithm.py
```

The program will prompt you for:
- target variable (1 or 2)
- continent (e.g., Europe, North America, South America)
- year to predict (e.g., 2030)

---

## Outputs
After running `Lovelace_Algorithm.py`, the following are generated in the repo directory:

- `continent_fit_and_forecast.png`
- `holdout_actual_vs_pred.png`
- `forecast_next_10_years.csv`

---

## Notes / Assumptions
- The model uses **closed-form linear regression** (normal-equation sums) rather than a black-box library.
- For **PM2.5 temporal coverage (%)**, predicted values are **clamped to [0, 100]** to avoid impossible results (e.g., >100% coverage) when forecasting.
- The holdout plot is included to show model evaluation; holdout R² may be low/negative because the dataset is small and trends are not perfectly linear.

---

## Canvas Submission
Submit:
- Link to this GitLab repo
- Demo video showing the plots and prediction output
- Written reflection (about 500 words, including positionality)
