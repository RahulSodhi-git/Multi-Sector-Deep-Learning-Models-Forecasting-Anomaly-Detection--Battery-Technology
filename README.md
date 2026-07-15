# Battery-Technology Anomaly Detection

An **unsupervised anomaly-detection** pipeline for battery sensor telemetry
(voltage, current, internal resistance, temperature). It improves data quality with
a rolling statistical filter, then flags abnormal operating points using an
**ensemble of outlier detectors**. Part of a multi-sector deep-learning /
machine-learning portfolio (battery-technology sector).

---

## Approach

1. **Data-quality pass** — a **rolling z-score** over each sensor stream to catch
   local statistical deviations that a global threshold would miss.
2. **Ensemble anomaly detection** — combine three complementary detectors:
   - **Rolling Z-score** (statistical),
   - **Isolation Forest** (tree-based isolation),
   - **Local Outlier Factor (LOF)** (density-based).
   A point is scored by agreement across methods for robust flags.
3. **Evaluation** — confusion matrix / metrics on the labelled portion of the data.

**Features used:** `voltage`, `current`, `internal_resistance`, `temperature`.

---

## MLOps pipeline (MLflow + Dagster)

Beyond the notebook, the workflow is designed to run as an orchestrated,
**reproducible pipeline** rather than ad-hoc cells:

- **Dagster** orchestrates the stages as a graph of assets:

  ```
  ingest_sensor_data ─► clean_and_rolling_zscore ─► fit_anomaly_ensemble ─► evaluate
  ```

  Each stage is a materializable asset, so the pipeline can be re-run, scheduled,
  and inspected in the Dagster UI, with clear data lineage between steps.

- **MLflow** tracks every run — logging parameters (rolling-window size,
  contamination rate, Isolation Forest / LOF settings), metrics (precision, recall,
  anomaly counts), and the fitted models as artifacts — so experiments are
  comparable and reproducible from the MLflow UI (`mlflow ui`).

> Note: this repository currently contains the analysis notebook. The
> Dagster job definitions and MLflow tracking wiring are being reintegrated and
> will be added here.

---

## Getting started

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter openpyxl
jupyter notebook batterytechmodels.ipynb
```

> **Data:** the notebook loads a battery sensor spreadsheet. Update the `file_path`
> near the top of the notebook to point at your own dataset (an `.xlsx` with the
> sensor columns above). The dataset is not included in this repository.

---

## Tech stack

**Python** · **scikit-learn** (Isolation Forest, LOF) · **SciPy** ·
**pandas / NumPy** · **matplotlib / seaborn** · **MLflow** (experiment tracking) ·
**Dagster** (pipeline orchestration)
