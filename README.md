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
**pandas / NumPy** · **matplotlib / seaborn**
