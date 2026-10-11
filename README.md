# SDSS object classification

A personal project learning how to classify galaxies, stars and quasars using
photometric measurements. It uses pandas, SQL, scikit-learn and Matplotlib.

## What's in the folder?

```text
data/       The original SDSS CSV and CSV files made during the analysis
src/        Small Python scripts, numbered in the order to run them
sql/        Five queries for data checks, summaries, features and error analysis
outputs/    tables/ contains CSV results; figures/ contains PNG plots
report/     The short report, its PDF and the ZIP for Overleaf
```

The scripts run from top to bottom. Each has comments explaining the main steps.

| Script | What it does |
| --- | --- |
| `01_prepare_data.py` | Reads the CSV, runs SQL quality checks and calculates colours. |
| `02_summarise_data.py` | Runs SQL to count classes and summarise their measurements. |
| `03_split_data.py` | Saves the shared training and test sets. |
| `04_cross_validate.py` | Compares models within the training set and selects the forest leaf size. |
| `05_compare_models.py` | Fits the models and measures their test performance. |
| `06_feature_importance.py` | Measures which features are useful to the selected forest. |
| `07_analyse_errors.py` | Measures accuracy in bins and uses SQL to investigate mistakes. |
| `08_plot_sample.py` | Plots class counts, colours and measurement distributions. |
| `09_plot_models.py` | Plots model scores, the confusion matrix and feature importance. |
| `10_plot_accuracy.py` | Plots how accuracy changes across measurement bins. |
| `11_plot_mistakes.py` | Plots mistakes in colour space and checks prediction confidence. |

Two small helpers keep repeated settings in one place: `model_helpers.py` contains
feature names and forest settings; `plot_helpers.py` contains plot colours, fonts
and functions for reading tables and saving figures. You do not run these separately.

## SQL used in the project

The queries run through DuckDB on the pandas tables supplied by the Python scripts.
They produce the data and results used in the analysis.

| Query | Question it answers | SQL used |
| --- | --- | --- |
| [01_quality_checks.sql](sql/01_quality_checks.sql) | Are there duplicate IDs, missing values or invalid measurements? | `WHERE`, `COUNT`, `DISTINCT`, `GROUP BY`, `HAVING`, a CTE and `UNION ALL` |
| [02_create_features.sql](sql/02_create_features.sql) | Which measurements and colour differences go into the models? | `SELECT`, arithmetic, aliases and `ORDER BY` |
| [03_class_summary.sql](sql/03_class_summary.sql) | How large is each class, and what are its average measurements? | `GROUP BY`, `COUNT`, `AVG`, a CTE and `SUM(...) OVER ()` |
| [04_error_summary.sql](sql/04_error_summary.sql) | Which pairs of classes are confused most often? | `INNER JOIN`, filtering, grouped counts and a window total |
| [05_confident_mistakes.sql](sql/05_confident_mistakes.sql) | What are the five most confident wrong predictions in each class? | `INNER JOIN`, `CASE`, CTEs and `ROW_NUMBER() OVER (PARTITION BY ...)` |

A **CTE** is a named intermediate query written with `WITH`. A **window function**
calculates a total or rank while keeping individual rows. The error queries join
catalogue measurements to held-out predictions using the unique object ID.

[quality_checks.csv](outputs/tables/quality_checks.csv) records the number of problems
found; all counts are zero for this sample.
[confident_mistakes.csv](outputs/tables/confident_mistakes.csv) contains five examples per true class,
ranked by prediction confidence, with ties broken by object ID. Its high/medium/low
bands use thresholds of 0.9 and 0.7 and are only descriptive labels. These examples
are for inspecting mistakes, not choosing model settings from the test set.

## Data and results

The input is `data/sdss_sample.csv`: 50,000 objects, with 29,036 galaxies,
16,326 stars and 4,638 quasars. It is included so the same sample can be used again.
The original notes describe it as an SDSS DR17 SpecPhoto export, but the download
query and `zWarning` column were not retained, so the exact selection cannot be
checked independently.

The code keeps the original row order and uses seed 42. There are 40,000 training
objects and 10,000 test objects. Model inputs are five magnitudes, four colours
and five magnitude uncertainties. IDs and spectroscopic redshift are not inputs.
Scaling and model selection use only training data.

Start with these tables in `outputs/tables/`:

- `model_comparison.csv`: accuracy and F1 scores for all five model comparisons.
- `classification_report_random_forest.csv`: precision, recall and F1 by class.
- `confusion_matrix_random_forest.csv`: counts of correct and incorrect predictions.
- `accuracy_by_magnitude.csv`: how accuracy changes for fainter objects.

The other tables describe cross-validation, feature importance and error trends.
Accuracy and F1 are stored as fractions, so an accuracy of `0.95` means 95%.
The selected Random Forest reaches **95.00% accuracy** and **0.911 macro F1**;
the forest without uncertainty features reaches **93.11% accuracy**.
Test results were inspected during earlier development, so a fresh sample would
be needed for a fully independent evaluation.

Scripts 01, 03 and 05 create five working files in `data/`: `prepared_data.csv`,
`train_data.csv`, `test_data.csv`, `train_test_split.csv` and `test_predictions.csv`.
They can be regenerated, so Git ignores them. The original input, result tables
and PNG figures are included. `.venv/` is just the local Python environment and
is also ignored.
