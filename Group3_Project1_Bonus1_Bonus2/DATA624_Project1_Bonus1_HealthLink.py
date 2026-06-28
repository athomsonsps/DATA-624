"""
DATA 624 Project 1 - Bonus #1
Texas HealthLink Python Forecasting Script - FIXED VERSION

Purpose:
This script supports Bonus #1 by reimplementing the Texas HealthLink
forecasting workflow in Python. It compares Python forecasting models,
selects the best model for each clinic using a holdout test, and produces
12-month forecasts for each clinic and the systemwide total.

Input needed:
healthlink_bonus1.csv

Required columns:
clinic, month, visits

Outputs created in a folder named bonus1_python_outputs:
1. python_bonus1_clinic_accuracy.csv
2. python_bonus1_clinic_forecasts.csv
3. python_bonus1_system_forecast.csv
4. python_bonus1_model_summary.csv
5. python_bonus1_report_summary.txt
6. python_bonus1_system_forecast_plot.png
7. python_bonus1_clinic_forecast_plot.png

Models used:
1. ETS / Exponential Smoothing
2. SARIMA / seasonal ARIMA-style model
3. Regression with trend and monthly seasonality

This script is designed to run outside RStudio. It does not need reticulate.

Author: Group 3
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings("ignore")


FORECAST_HORIZON = 12
HOLDOUT_MONTHS = 12


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred):
    """Mean absolute percentage error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def make_month_features(dates):
    """Create monthly dummy variables."""
    dates = pd.to_datetime(dates)
    month_numbers = pd.Series(dates.month)
    month_dummies = pd.get_dummies(month_numbers, prefix="month", drop_first=True)
    return month_dummies.reset_index(drop=True)


def regression_forecast(train_series, forecast_dates):
    """
    Regression model with trend and monthly seasonal dummies.
    This is comparable to regression with seasonality in R.
    """
    train_dates = train_series.index

    train_x = pd.DataFrame({"trend": np.arange(1, len(train_series) + 1)})
    train_x = pd.concat(
        [train_x.reset_index(drop=True), make_month_features(train_dates)],
        axis=1
    )

    future_x = pd.DataFrame({
        "trend": np.arange(len(train_series) + 1, len(train_series) + len(forecast_dates) + 1)
    })
    future_x = pd.concat(
        [future_x.reset_index(drop=True), make_month_features(forecast_dates)],
        axis=1
    )

    future_x = future_x.reindex(columns=train_x.columns, fill_value=0)

    model = LinearRegression()
    model.fit(train_x, train_series.values)

    predictions = model.predict(future_x)
    return np.maximum(predictions, 0)


def ets_forecast(train_series, steps):
    """ETS / Exponential Smoothing forecast."""
    model = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated"
    )
    fit = model.fit(optimized=True)
    predictions = fit.forecast(steps)
    return np.maximum(np.asarray(predictions), 0)


def sarima_forecast(train_series, steps):
    """
    Seasonal ARIMA-style forecast using statsmodels SARIMAX.
    This uses a simple, stable model form for the class project.
    """
    model = SARIMAX(
        train_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit = model.fit(disp=False)
    predictions = fit.forecast(steps)
    return np.maximum(np.asarray(predictions), 0)


def clean_input_data(data_path):
    """Load and validate HealthLink data."""
    if not data_path.exists():
        raise FileNotFoundError(
            "\nCannot find the input data file.\n"
            f"Expected file: {data_path}\n"
            "Put healthlink_bonus1.csv in the same folder as this script, "
            "or pass the file path using --data.\n"
        )

    df = pd.read_csv(data_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required_cols = {"clinic", "month", "visits"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            "The data file is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    df = df[["clinic", "month", "visits"]].copy()
    df["clinic"] = df["clinic"].astype(str).str.strip()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["visits"] = pd.to_numeric(df["visits"], errors="coerce")

    bad_rows = df[df["month"].isna() | df["visits"].isna()]
    if len(bad_rows) > 0:
        raise ValueError(
            "Some rows have invalid month or visits values. "
            "Fix the CSV before running the script."
        )

    df = df.sort_values(["clinic", "month"]).reset_index(drop=True)

    return df


def get_monthly_series(df, clinic):
    """Return one monthly time series for one clinic."""
    clinic_df = df[df["clinic"] == clinic].copy()
    series = clinic_df.set_index("month")["visits"].sort_index().asfreq("MS")

    # Small safety rule: interpolate any missing month created by asfreq.
    # This should not happen if the data is complete, but it prevents a crash.
    if series.isna().any():
        series = series.interpolate(method="linear").ffill().bfill()

    return series


def evaluate_models_for_series(series, clinic_name):
    """Compare Python models using the last 12 months as the holdout set."""
    if len(series) <= HOLDOUT_MONTHS + 24:
        raise ValueError(f"Not enough data for holdout testing: {clinic_name}")

    train = series.iloc[:-HOLDOUT_MONTHS]
    test = series.iloc[-HOLDOUT_MONTHS:]
    forecast_dates = test.index

    model_predictions = {}

    model_functions = {
        "ETS": lambda: ets_forecast(train, HOLDOUT_MONTHS),
        "SARIMA": lambda: sarima_forecast(train, HOLDOUT_MONTHS),
        "Regression_Seasonality": lambda: regression_forecast(train, forecast_dates),
    }

    for model_name, model_func in model_functions.items():
        try:
            model_predictions[model_name] = model_func()
        except Exception as exc:
            print(f"{model_name} failed for {clinic_name}: {exc}")

    if not model_predictions:
        raise RuntimeError(f"All models failed for {clinic_name}.")

    rows = []
    for model_name, predictions in model_predictions.items():
        rows.append({
            "clinic": clinic_name,
            "model": model_name,
            "RMSE": rmse(test.values, predictions),
            "MAE": float(mean_absolute_error(test.values, predictions)),
            "MAPE": mape(test.values, predictions)
        })

    accuracy = pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    best_model = accuracy.loc[0, "model"]

    return accuracy, best_model


def forecast_best_model(series, best_model, horizon):
    """Fit the selected model on all available data and forecast 12 months."""
    last_date = series.index.max()
    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS"
    )

    if best_model == "ETS":
        predictions = ets_forecast(series, horizon)
    elif best_model == "SARIMA":
        predictions = sarima_forecast(series, horizon)
    elif best_model == "Regression_Seasonality":
        predictions = regression_forecast(series, future_dates)
    else:
        raise ValueError(f"Unknown model: {best_model}")

    return future_dates, predictions


def create_plots(df, clinic_forecasts, system_forecast, output_dir):
    """Create system and clinic forecast plots."""
    historical_system = (
        df.groupby("month", as_index=False)["visits"]
        .sum()
        .rename(columns={"visits": "system_visits"})
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        historical_system["month"],
        historical_system["system_visits"],
        label="Historical system visits"
    )
    plt.plot(
        system_forecast["month"],
        system_forecast["system_total_forecast_visits"],
        label="Python forecast"
    )
    plt.title("Texas HealthLink Systemwide Visits: Python Forecast")
    plt.xlabel("Month")
    plt.ylabel("Visits")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "python_bonus1_system_forecast_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    for clinic in sorted(df["clinic"].unique()):
        actual = df[df["clinic"] == clinic]
        forecast = clinic_forecasts[clinic_forecasts["clinic"] == clinic]
        plt.plot(actual["month"], actual["visits"], alpha=0.65)
        plt.plot(forecast["month"], forecast["forecast_visits"], linestyle="--", alpha=0.9)

    plt.title("Texas HealthLink Clinic Forecasts: Python Models")
    plt.xlabel("Month")
    plt.ylabel("Visits")
    plt.tight_layout()
    plt.savefig(output_dir / "python_bonus1_clinic_forecast_plot.png", dpi=150)
    plt.close()


def write_report_summary(model_summary, system_forecast, output_dir):
    """Create a short text summary that can be pasted into the report."""
    best_models = (
        model_summary["selected_python_model"]
        .value_counts()
        .reset_index()
    )
    best_models.columns = ["model", "number_of_clinics"]

    avg_rmse = model_summary["holdout_RMSE"].mean()
    first_month = system_forecast["month"].min().strftime("%Y-%m")
    last_month = system_forecast["month"].max().strftime("%Y-%m")
    avg_system_forecast = system_forecast["system_total_forecast_visits"].mean()

    lines = []
    lines.append("Bonus #1 Python Forecasting Summary")
    lines.append("")
    lines.append("I reimplemented the Texas HealthLink forecasting workflow in Python.")
    lines.append("The Python workflow tested ETS, SARIMA, and regression with monthly seasonality.")
    lines.append("The last 12 months were used as a holdout period to compare model accuracy.")
    lines.append("")
    lines.append("Selected model count by clinic:")
    for _, row in best_models.iterrows():
        lines.append(f"- {row['model']}: {row['number_of_clinics']} clinic(s)")
    lines.append("")
    lines.append(f"Average clinic holdout RMSE: {avg_rmse:,.2f}")
    lines.append(f"Forecast horizon: {first_month} to {last_month}")
    lines.append(f"Average monthly systemwide forecast: {avg_system_forecast:,.0f} visits")
    lines.append("")
    lines.append("Main takeaway:")
    lines.append(
        "Python can reproduce the same forecasting workflow used in R. "
        "R is still stronger for direct time series reporting, while Python is useful "
        "when the forecast needs to connect with machine learning workflows, dashboards, "
        "or production systems."
    )

    report_text = "\n".join(lines)
    (output_dir / "python_bonus1_report_summary.txt").write_text(report_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="DATA 624 Bonus #1 Texas HealthLink Python Forecasting"
    )
    parser.add_argument(
        "--data",
        default="healthlink_bonus1.csv",
        help="Path to HealthLink CSV file. Default: healthlink_bonus1.csv"
    )
    parser.add_argument(
        "--output",
        default="bonus1_python_outputs",
        help="Output folder. Default: bonus1_python_outputs"
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = clean_input_data(data_path)

    print("\nDATA 624 Bonus #1 - Python Forecasting")
    print("--------------------------------------")
    print(f"Rows loaded: {len(df)}")
    print(f"Clinics: {', '.join(sorted(df['clinic'].unique()))}")
    print(f"Date range: {df['month'].min().date()} to {df['month'].max().date()}")
    print(f"Output folder: {output_dir.resolve()}")

    all_accuracy = []
    all_forecasts = []
    model_summary = []

    for clinic in sorted(df["clinic"].unique()):
        print(f"\nWorking on clinic: {clinic}")
        series = get_monthly_series(df, clinic)

        accuracy, best_model = evaluate_models_for_series(series, clinic)
        all_accuracy.append(accuracy)

        future_dates, predictions = forecast_best_model(series, best_model, FORECAST_HORIZON)

        forecast_df = pd.DataFrame({
            "clinic": clinic,
            "month": future_dates,
            "python_model": best_model,
            "forecast_visits": np.round(predictions, 0).astype(int)
        })
        all_forecasts.append(forecast_df)

        model_summary.append({
            "clinic": clinic,
            "selected_python_model": best_model,
            "holdout_RMSE": float(accuracy.loc[0, "RMSE"]),
            "holdout_MAE": float(accuracy.loc[0, "MAE"]),
            "holdout_MAPE": float(accuracy.loc[0, "MAPE"])
        })

    accuracy_out = pd.concat(all_accuracy, ignore_index=True)
    forecasts_out = pd.concat(all_forecasts, ignore_index=True)
    model_summary_out = pd.DataFrame(model_summary)

    system_forecast = (
        forecasts_out
        .groupby("month", as_index=False)["forecast_visits"]
        .sum()
        .rename(columns={"forecast_visits": "system_total_forecast_visits"})
    )
    system_forecast["method"] = "Bottom-up sum of clinic forecasts"

    accuracy_out.to_csv(output_dir / "python_bonus1_clinic_accuracy.csv", index=False)
    forecasts_out.to_csv(output_dir / "python_bonus1_clinic_forecasts.csv", index=False)
    system_forecast.to_csv(output_dir / "python_bonus1_system_forecast.csv", index=False)
    model_summary_out.to_csv(output_dir / "python_bonus1_model_summary.csv", index=False)

    create_plots(df, forecasts_out, system_forecast, output_dir)
    write_report_summary(model_summary_out, system_forecast, output_dir)

    print("\nFiles created:")
    for file_path in sorted(output_dir.iterdir()):
        print(f"- {file_path.name}")

    print("\nSelected Python models:")
    print(model_summary_out.to_string(index=False))

    print("\nDone. Use these outputs in Bonus #1 for the R vs Python comparison.")


if __name__ == "__main__":
    main()
