# DATA 624 – HomeWork: Exercises 1–4

This folder contains the R Markdown homework assignments for **Exercises 1 through 4** of the DATA 624 Predictive Analytics course.

## Exercises Overview

| Exercise | Topic |
|----------|-------|
| Ex. 1 | Introduction to time series data and basic exploration |
| Ex. 2 | Time series decomposition and seasonal adjustment |
| Ex. 3 | Exponential smoothing and forecasting methods |
| Ex. 4 | ARIMA models and model selection |

## File Contents

Each exercise submission typically includes:

- **R Markdown source (`.Rmd`)** – the primary document with code and written analysis
- **Knitted output (`.html` or `.pdf`)** – the rendered version for review
- **Plots** – visualizations embedded in the knitted output
- **Model summaries** – printed outputs from fitted forecasting models
- **Written interpretation** – academic discussion of findings

## How to Open and Knit in RStudio

1. Open RStudio.
2. Go to **File → Open File** and select the `.Rmd` file for the exercise you want to review.
3. Install any packages that are called with `library()` at the top of the file if they are not already installed. For example:
```r
install.packages(c("forecast", "fpp3", "tidyverse", "ggplot2"))
```
4. Click the **Knit** button (or press `Ctrl+Shift+K` / `Cmd+Shift+K`) to render the full document.
5. Alternatively, run individual code chunks interactively using **Ctrl+Enter** / **Cmd+Enter**.

## Notes

- All analysis is performed in R. Ensure R (≥ 4.0) and RStudio are installed.
- Data sets used in the exercises are from the `fpp3` or `fpp2` packages unless otherwise noted.
- Output files (`.html`, `.pdf`) are included so the work can be reviewed without knitting.
