# Inventory Replenishment: Demand Clustering & Forecasting Pipeline

A clustering-driven replenishment engine that projects 12 weeks of optimal inventory levels for 1 million (store, SKU) pairs of perishable goods.

## Overview

This project was developed at **CIMAT (Centro de Investigación en Matemáticas)** in February 2026 for **NEXO MX**, a retail client operating **20,000 stores** across northern Mexico that sell a catalog of **50 perishable products**.

Retailers selling perishables face a two-sided risk every week: order too little and lose sales to stock-outs; order too much and lose the full cost of the product to spoilage. The client needed a data-driven way to decide, for every store-SKU combination and every week over a 12-week horizon, exactly how much inventory to have on hand at the start of the week — balancing lost-sales risk against spoilage risk.

The solution reframes the problem as a variant of the **Newsvendor Problem** (single-period inventory optimization under demand uncertainty), calibrated per demand segment via unsupervised clustering, and validated with a discrete-event style simulation that compares projected profit against the client's current (baseline) inventory policy.

Raw input: ~2.3 years of daily sales history (Jan 2022 – May 2024) for 50 SKUs × 20,000 stores, pre-aggregated into weekly time-series statistics per (store, SKU) pair — the model itself never touches the raw transactional history, only the derived features, which keeps the pipeline tractable at 1M-row scale.

## Approach

The pipeline runs in five stages (see `main.py`):

1. **Data Loading** (`src/data_loader.py`) — Reads three sheets from the client's Excel workbook (`Inventario`, `CatSku`, `Resultados`) plus a separately pre-computed weekly time-series feature table (`features_seriestemporales.csv`). Includes robust column-name matching (Spanish/English aliases) and auto-detection of the CSV delimiter (`|`, `;`, `,`, tab).

2. **Demand Profile Clustering** (`src/clustering.py`) — Groups (store, SKU) pairs into demand archetypes using **K-Means** on 8 standardized features (mean weekly sales, coefficient of variation, 90th percentile, intermittency, lag-1 autocorrelation, demand-to-shelf-life ratio, pack-rounding ratio, and Newsvendor critical ratio). The number of clusters is chosen automatically by **silhouette score** (K = 2–6, sampled for scalability) unless fixed via CLI. Clusters are then semantically labeled A–D (stable / volatile / intermittent / trending) and mapped to a policy parameter set (safety-stock offset α and target-service multiplier) — i.e., clustering is used not just for insight but to calibrate the downstream inventory policy per segment.

3. **Weekly Demand Forecast** (`src/forecaster.py`) — Computes a weekly demand forecast as historical mean + a per-cluster safety floor (α), rather than re-deriving it from raw history — an intentional simplification justified in the accompanying technical report, since the mean feature is already the MLE demand estimator under i.i.d. weekly demand.

4. **Replenishment Simulation** (`src/reabasto.py`) — Implements a periodic-review **(T, S) inventory policy** applied sequentially across all 12 weeks per pair: order quantity = max(0, target level − current inventory), rounded up to the nearest case-pack size (ceiling, never round-down, to avoid under-ordering), producing the `INV_IDEAL` figure the store should stock each Monday. Carry-over inventory between weeks follows **FIFO** logic tied to each SKU's shelf life — stock is written off as spoilage once its shelf life has elapsed.

5. **Profit Evaluation** (`src/evaluator.py`) — Simulates net profit under the proposed policy vs. the client's current inventory baseline, using `units_sold × margin − spoiled_units × cost` as the objective, and reports the aggregate uplift.

6. **Reporting** (`src/writer.py`) — Writes a wide-format CSV (one row per store-SKU pair, one column per projected week) and a formatted Excel summary workbook (cluster breakdown, feature table, and base-vs-proposed profit comparison).

## Tech Stack

- **Python 3.12**
- **pandas / numpy** — data wrangling and vectorized computation at 1M-row scale
- **scikit-learn** (`KMeans`, `StandardScaler`, `silhouette_score`) — unsupervised demand clustering
- **openpyxl** — reading multi-sheet Excel input and generating a styled Excel summary report
- Standard library: `argparse` (CLI), `math`, `time`

## Results

Run on the full client dataset (**1,000,000 store-SKU pairs × 12 projected weeks = 12,000,000 evaluated observations**):

| Cluster | Profile | Pairs | Share | Mean weekly demand |
|---|---|---:|---:|---:|
| A | High & stable demand | 172,902 | 17.3% | 12.47 |
| B | High & volatile demand | 370,930 | 37.1% | 4.03 |
| C | Intermittent demand | 427,255 | 42.7% | 3.95 |
| D | Trending / cyclical | 28,913 | 2.9% | 29.17 |

**Simulated profit comparison (baseline current-inventory policy vs. proposed model):**

| Metric | Value |
|---|---:|
| Baseline scheme profit | $347,197,373.99 MXN |
| Proposed scheme profit | $478,940,129.40 MXN |
| Absolute uplift | $131,742,755.41 MXN |
| **Relative uplift** | **+37.9%** |

These figures come directly from the pipeline's own simulation output (`resumen_abasto.xlsx`, sheet `Simulacion`), which compares the two policies against the same demand proxy — they represent the model's internal estimate of improvement, not an externally audited A/B test result.

Full derivation of the objective function, the (T,S) policy, the FIFO spoilage logic and worked numerical examples are documented in the accompanying technical report (`ProyectoAbasto_reporte.pdf`).

## How to Run

```bash
cd ABASTO_PROYECTOV1/ABASTO_PROYECTO
pip install -r requirements.txt

python main.py --excel CIMAT_BaseDatos.xlsx --features features_seriestemporales.csv
```

Optional arguments:

```bash
python main.py \
  --excel CIMAT_BaseDatos.xlsx \
  --features features_seriestemporales.csv \
  --out-csv resultados_abasto.csv \
  --out-xlsx resumen_abasto.xlsx \
  --n-clusters 4          # force K instead of auto-selecting via silhouette score
```

**Outputs:**
- `resultados_abasto.csv` — `INV_IDEAL` (ideal starting inventory) for every store-SKU pair × 12 weeks
- `resumen_abasto.xlsx` — cluster summary, feature table, and base-vs-proposed profit simulation

## Author

Naikary Paloma Martínez Velázquez — CIMAT, Feb. 2026
