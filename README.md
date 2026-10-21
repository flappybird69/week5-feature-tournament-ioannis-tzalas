# Week 5 Machine Learning & Regularization — Ioannis Tzalas

Continues [Week 1](https://github.com/flappybird69/week1-financial-analysis-ioannis-tzalas), [Week 2](https://github.com/flappybird69/week2-data-hygiene-ioannis-tzalas), [Week 3](https://github.com/flappybird69/week3-fintech-assignment-ioannis-tzalas), and [Week 4](https://github.com/flappybird69/week4-time-series-forecasting-ioannis-tzalas), using the five-asset universe: **SPY, EUR/USD, BTC, GLD, and ETH**.

- [`W5_assignment_Tzalas.ipynb`](W5_assignment_Tzalas.ipynb): Executed assignment notebook containing the junior agent draft code audit, the feature factory and prompt ledger, the time-ordered cross-validation tournament, leakage quantification, the sealed holdout showdown for mean and variance, GARCH(1,1)-t comparisons, the 10-rule ML evaluation suite, and the ElasticNet / log-variance bonus analyses.
- [`W2_own_calendar_prices.csv`](W2_own_calendar_prices.csv): The frozen Week 2 price panel retaining each asset’s native observed calendar independently (including 7-day crypto weekends and 5-day NYSE/FX calendars).
- [`requirements.txt`](requirements.txt): Environment dependencies for exact reproduction.

## Methodology & Protocol
- **Holdout Boundary:** Pre-registered at **2024-10-01** (declared before any model estimation or hyperparameter tuning, identical to Week 4).
- **Leak Detection:** Every feature column is audited with the executable `as_of_audit()` across 20 independent historical truncation dates to guarantee zero forward look-ahead.
- **Tournament Architecture:** Time-ordered 5-fold cross-validation (`TimeSeriesSplit`) with feature standardization (`StandardScaler`) strictly encapsulated inside each fold pipeline. Regularization penalties tuned against training-fold mean benchmarks, selecting the one-standard-error (1-SE) penalty.
- **Holdout Showdown:** Evaluated out-of-sample from 2024-10-01 to 2026-09-15 with quarterly rolling refits (`REFIT=63`). Diebold–Mariano tests with HAC standard errors determine incremental skill against the expanding historical mean.
- **Variance Forecasts:** Penalized Lasso on squared returns ($r_{t+1}^2$) discovers the Heterogeneous Autoregressive (HAR) volatility structure, evaluated via QLIKE loss and compared against historical constant, EWMA(0.94), and Week 4's GARCH(1,1)-t.
- **Analyst Agent Constitution:** Eval Suite v1.0 enforces 10 executable assertions governing temporal integrity, feature stability, and model reporting claims.

## Reproduction
Install dependencies in a Python 3.14 environment:
```bash
pip install -r requirements.txt
```
Open `W5_assignment_Tzalas.ipynb` and select **Kernel → Restart & Run All**. All figures, tables, and audit logs regenerate deterministically offline.
