# Prescription-Medication-Price-Prediction-End-to-End-ML-Pipeline

## The question

Given a drug's current-year characteristics (price, claims volume, competition,
brand/generic status, drug class), can we predict its price per dosage unit next
year — and does that beat just guessing **next year's price = this year's price**?

Drug pricing is notoriously sticky, so that "persistence baseline" is a deceptively
strong competitor. Whether a model can meaningfully beat it tells you how much real
signal is in the features vs. how much is just inertia.

## Data

[CMS Medicare Part D Spending by Drug](https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-medicaid-spending-by-drug/medicare-part-d-spending-by-drug)
— public, free, claims-based pricing data for thousands of drugs across multiple years
(brand/generic name, manufacturer, claims volume, beneficiaries, and average spending
per dosage unit per year).

**Caveat:** this reflects *gross Medicare claims cost* (Medicare + plan + beneficiary
payments combined), not what an individual pays out-of-pocket. It's a strong proxy for
pricing *trends*, not retail/cash prices.

The notebook downloads this data directly via the CMS Data API. If that's unreachable
in your environment, it falls back to a synthetic dataset with the same schema and
realistic brand/generic/competition dynamics so the pipeline still runs end-to-end —
but the results below are from the real CMS dataset unless noted otherwise.

## Pipeline

1. Data download (CMS API, with synthetic fallback)
2. Cleaning & reshaping to long format (one row per drug-year)
3. Feature engineering — lag prices (log-transformed), price growth rate, claims per
   beneficiary, competition buckets, manufacturer frequency encoding
4. EDA — price distributions, brand vs. generic gaps, trends by drug class, biggest
   year-over-year movers
5. Time-based train/test split + **persistence baseline**
6. Modeling — Ridge regression (standardized features) and XGBoost
7. Evaluation against the persistence baseline (MAE, RMSE, MAPE, R²)
8. Walk-forward validation across all available years
9. Feature importance
10. Model export (`joblib`) for deployment

## Results

| Model | R² (log) | MAE ($) | MAPE |
|---|---|---|---|
| Persistence baseline (last year's price) | 0.998 | $10.68 | 6.05% |
| Ridge Regression | 0.957 | $33.18 | 28.39% |
| XGBoost | 0.998 | $15.95 | 5.97% |

## Key findings

- **XGBoost essentially ties the persistence baseline.** A narrow MAPE improvement
  (5.97% vs 6.05%) is not a meaningful win — drug prices are dominated by inertia,
  and the engineered features add at most marginal signal on top of "last year's
  price."
- **Ridge required a fix to be usable.** Initially it scored R²=0.82 / MAPE=84%
  despite using the same features as XGBoost. Inspecting the coefficients showed the
  lag-price feature was being fed in raw dollars against a log-scaled target — a
  linear model can't represent that relationship, so it dumped its weight onto
  categorical proxies (drug class, brand/generic) instead. Log-transforming the lag
  feature (`log_price_lag_1`) fixed this, bringing Ridge to R²=0.957 / MAPE=28.4%,
  with `log_price_lag_1` correctly dominating the coefficients.
- **Implication:** predicting absolute price is nearly solved by persistence. The
  more useful framing is likely **predicting deviations from persistence** — i.e.,
  which drugs will see unusually large price changes — where competition,
  drug class, and claims volume are more likely to carry real signal.

## Limitations

- No rebate data (CMS can't publish manufacturer rebates), so list-price dynamics may
  not reflect net cost to payers.
- Annual granularity only — can't capture intra-year price timing.
- Cross-sectional model trained across all drugs; a dedicated time-series model per
  high-spend drug would likely do better for those specific drugs.

## Next steps

- Reframe as predicting price *change* (or classifying ">X% increase") rather than
  absolute price.
- Merge FDA NDC Directory / RxNorm therapeutic class, GoodRx cash prices, or WAC list
  prices for richer features.
- Per-drug time series (ARIMA/Prophet) for the highest-spend drugs.
- Hyperparameter tuning (Optuna/GridSearchCV) and periodic retraining as CMS releases
  new annual data.

## Running it

Open `drug_price_prediction_pipeline.ipynb` in Google Colab (or Jupyter) and run all
cells — dependencies install in the first cell. No API keys required.
