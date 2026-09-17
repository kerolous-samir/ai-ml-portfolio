# SuperKart — Sales Forecasting & Model Deployment

> A tuned regression model for quarterly store-level sales revenue, packaged the whole way through
> to production: **Flask REST API → Streamlit UI → Docker → Hugging Face Spaces**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipelines-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
![R2](https://img.shields.io/badge/Test%20R²-0.933-brightgreen)

## Business Context

A sales forecast underpins planning across an organisation — it drives procurement, regional
strategy and territory coverage, and it reduces pipeline risk. **SuperKart** runs supermarkets and
food marts across tier-1 to tier-3 cities and needs accurate revenue forecasts for the coming
quarter to optimise inventory.

Critically, SuperKart did not just want a model — they wanted a **deployable forecasting service**
that could be integrated into their decision systems and used across the store network. That
deployment requirement is what separates this project from a standard regression exercise.

## Objective

Forecast `Product_Store_Sales_Total` at the product–store level, then operationalise the model as a
running service.

## Dataset

**8,763 rows × 12 columns.**

| Column | Description |
|--------|-------------|
| `Product_Id`, `Product_Weight` | Product identifier and weight |
| `Product_Sugar_Content` | Low / Regular / No sugar |
| `Product_Allocated_Area` | Share of store display area given to the product |
| `Product_Type`, `Product_MRP` | Category and maximum retail price |
| `Store_Id`, `Store_Establishment_Year` | Store identifier and opening year |
| `Store_Size` | High / Medium / Small |
| `Store_Location_City_Type` | Tier 1 / Tier 2 / Tier 3 |
| `Store_Type` | Store format |
| `Product_Store_Sales_Total` | **Target** — total sales revenue |

**Feature engineering:** `Store_Age` derived from `Store_Establishment_Year`; product categories
consolidated; sugar-content labels normalised.

## Approach

1. **EDA** — univariate and bivariate analysis of every product and store attribute against revenue.
2. **Preprocessing inside a `Pipeline`** — imputation, scaling and one-hot encoding fitted within each CV fold to prevent leakage.
3. **Model building** — Random Forest and XGBoost regressors, cross-validated on RMSE.
4. **Hyperparameter tuning** — grid search over tree depth, estimator count, learning rate, subsampling and column sampling.
5. **Serialisation** — best pipeline persisted with `joblib` as `superkart_best_model.pkl`.
6. **Deployment** — Flask prediction API, Streamlit front end, Dockerfile, and automated upload to Hugging Face Spaces.

## Results

| Model | CV RMSE | Train RMSE | **Test RMSE** | Test MAE | **Test R²** | Adjusted R² |
|-------|:-------:|:----------:|:-------------:|:--------:|:-----------:|:-----------:|
| Random Forest (baseline) | 290.5 | 106.9 | 281.6 | 108.1 | 0.9305 | 0.9300 |
| XGBoost (baseline) | 304.5 | 112.5 | 303.9 | 137.1 | 0.9191 | 0.9185 |
| **Random Forest (tuned)** — *selected* | — | 170.1 | **276.99** | **106.6** | **0.9328** | **0.9323** |
| XGBoost (tuned) | — | 231.4 | 288.8 | 122.7 | 0.9269 | 0.9264 |

**Selected model:** tuned Random Forest — `n_estimators=300`, `min_samples_split=10`, `max_depth=None`.

Tuning did more than move the test score: the train/test RMSE gap narrowed from **106.9 / 281.6**
(a clear overfit) to **170.1 / 277.0**, so the tuned model generalises considerably better while
also scoring higher. It explains **93.3%** of variance in store-level sales revenue.

## Deployment Architecture

```
                 ┌──────────────────────┐
  User input ───▶│  Streamlit frontend  │
                 └──────────┬───────────┘
                            │ HTTP POST /predict
                            ▼
                 ┌──────────────────────┐
                 │   Flask REST API     │
                 │  (app.py, Docker)    │
                 └──────────┬───────────┘
                            │ joblib.load
                            ▼
                 ┌──────────────────────┐
                 │ superkart_best_model │
                 │       .pkl           │
                 └──────────────────────┘
            Both services hosted on Hugging Face Spaces
```

The notebook writes out `app.py`, `requirements.txt` and the `Dockerfile`, then programmatically
creates the backend (Docker SDK) and frontend (Streamlit SDK) Spaces and uploads each artefact.

> **Security note.** The Hugging Face token is read from the `HF_TOKEN` environment variable —
> never hard-coded. Set it before running the deployment cells:
> ```bash
> export HF_TOKEN="your_token_here"
> ```

## Key Insights & Recommendations

- **`Product_MRP` is the strongest driver of revenue.** Consumers pay for premium items — keep high-MRP products in stock and prominently displayed.
- **Shelf space converts.** Larger `Product_Allocated_Area` correlates with higher sales; give display area to high-margin and seasonal lines.
- **Store profile matters.** Large stores in Tier-1 cities outperform smaller outlets in lower tiers — prioritise metro expansion with larger footprints.
- **Category effects are real.** Snack foods, dairy and soft drinks outperform, as do regular-sugar items over low/no-sugar alternatives — useful for promotions, with health trends worth monitoring.
- **Feed procurement directly.** Product–store level forecasts can drive automatic re-ordering, cutting both stockouts and overstock.
- **Scenario planning.** With R² above 0.93 the model is accurate enough to simulate the revenue impact of price or shelf-space changes.

## Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `pandas` · `NumPy` · `Flask` · `Streamlit` · `Docker` ·
`Hugging Face Hub` · `joblib` · `Matplotlib` · `Seaborn`

## Files

| File | Description |
|------|-------------|
| [`notebook.ipynb`](notebook.ipynb) | Full pipeline — EDA through to deployment |
| [`report.html`](report.html) | Rendered HTML report — **[view live](https://kerolous-samir.github.io/ai-ml-portfolio/05-superkart-sales-forecasting/report.html)** |
