# SuperKart — Sales Forecasting & Model Deployment

> A tuned regression model for quarterly store-level sales revenue, packaged the whole way through
> to production: **Flask REST API → Streamlit UI → Docker → Hugging Face Spaces**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipelines-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Live API](https://img.shields.io/badge/Live%20API-Hugging%20Face%20Spaces-FFD21E?logo=huggingface&logoColor=black)](https://kerolous-samir-superkart-backend.hf.space)

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

**8,763 rows × 12 columns.** No missing values and no duplicate rows.

| Column | Description |
|--------|-------------|
| `Product_Id`, `Product_Weight` | Product identifier and weight (mean 12.65) |
| `Product_Sugar_Content` | `Low Sugar` / `Regular` / `No Sugar`, plus ~110 rows of a stray `reg` variant left un-merged |
| `Product_Allocated_Area` | Share of store display area given to the product |
| `Product_Type` | 16 categories — Fruits and Vegetables, Snack Foods, Frozen Foods, Dairy, Household, … |
| `Product_MRP` | Maximum retail price (mean 147.03) |
| `Store_Id`, `Store_Establishment_Year` | Store identifier and opening year (1987–2009) |
| `Store_Size` | High / Medium / Small |
| `Store_Location_City_Type` | Tier 1 / Tier 2 / Tier 3 |
| `Store_Type` | Departmental Store / Supermarket Type1 / Supermarket Type2 / Food Mart |
| `Product_Store_Sales_Total` | **Target** — total sales revenue (mean 3,464.00, std 1,065.63, range 33–8,000) |

**Feature engineering:** `Store_Age` derived from `Store_Establishment_Year`; `Product_Category`
extracted from the two-letter `Product_Id` prefix; `MRP_per_weight` added as a
price-density ratio. Outliers capped by the IQR rule on `Product_Weight`,
`Product_Allocated_Area`, `Product_MRP` and `MRP_per_weight`.

## Approach

1. **EDA** — univariate distributions for all five numeric and all five categorical attributes; bivariate analysis limited to a correlation heatmap of the numeric features plus `Product_MRP`, `Product_Sugar_Content` and `Store_Size` against revenue.
2. **Preprocessing inside a `Pipeline`** — `StandardScaler` on numeric columns and `OneHotEncoder(handle_unknown='ignore')` on categoricals, fitted within each CV fold to prevent leakage. The dataset has no missing values, so no imputation step is required.
3. **Model building** — Random Forest and XGBoost regressors, 5-fold cross-validated on RMSE.
4. **Hyperparameter tuning** — `GridSearchCV` (cv=3) over tree depth, estimator count, minimum split size, learning rate, subsampling and column sampling.
5. **Serialisation** — best pipeline persisted with `joblib` as `superkart_best_model.pkl` (~38 MB).
6. **Deployment** — Flask prediction API, Streamlit front end, a Dockerfile per service, and automated upload to two Hugging Face Spaces.

## Results

| Model | CV RMSE | Train RMSE | **Test RMSE** | Test MAE | **Test R²** | Adjusted R² |
|-------|:-------:|:----------:|:-------------:|:--------:|:-----------:|:-----------:|
| Random Forest (baseline) | 290.5 | 106.9 | 281.6 | 108.1 | 0.9305 | 0.9300 |
| XGBoost (baseline) | 304.5 | 112.5 | 303.9 | 137.1 | 0.9191 | 0.9185 |
| **Random Forest (tuned)** — *selected* | — | 170.1 | **276.99** | **106.6** | **0.9328** | **0.9323** |
| XGBoost (tuned) | — | 231.4 | 288.8 | 122.7 | 0.9269 | 0.9264 |

"Baseline" here means the untuned regressor, not a naive predictor. The CV RMSE column is blank for
the tuned rows because `GridSearchCV.best_score_` was never recorded.

**Selected model:** tuned Random Forest — `n_estimators=300`, `min_samples_split=10`,
`max_depth=None`. Tuned XGBoost settled on `n_estimators=200`, `learning_rate=0.05`, `max_depth=5`,
`subsample=0.8`, `colsample_bytree=1.0`.

Tuning did more than move the test score: the train/test RMSE gap narrowed from **106.9 / 281.6**
(a clear overfit) to **170.1 / 277.0**, so the tuned model generalises considerably better while
also scoring higher.

**Read the headline R² of 0.933 as an optimistic upper bound, not a clean generalisation
estimate** — the final model was chosen by comparing test RMSE, so the same split served as both
selection and reporting set. See [Known Limitations](#known-limitations).

## Deployment Architecture

```
                 ┌──────────────────────┐
  User input ───▶│  Streamlit frontend  │
                 │    (Docker Space)    │
                 └──────────┬───────────┘
                            │ HTTP POST /predict
                            ▼
                 ┌──────────────────────┐
                 │   Flask REST API     │
                 │  gunicorn, :7860     │
                 │    (Docker Space)    │
                 └──────────┬───────────┘
                            │ joblib.load
                            ▼
                 ┌──────────────────────┐
                 │ superkart_best_model │
                 │       .pkl           │
                 └──────────────────────┘
            Both services hosted on Hugging Face Spaces
```

The notebook writes out `app.py`, `requirements.txt` and a `Dockerfile` for each service, then
programmatically creates both Spaces — **both on the Docker SDK** — and uploads each artefact. The
frontend image is a Docker Space that runs `streamlit run app.py --server.port 7860` internally.

The same code is extracted into [`deployment/`](deployment) as real runnable files, with its own
README covering the API contract, field reference and local/Docker run instructions.

**The backend is live:**

```bash
curl -X POST https://kerolous-samir-superkart-backend.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"Product_Id":"FD6114","Product_Weight":12.5,"Product_Sugar_Content":"Low Sugar",
       "Product_Allocated_Area":0.03,"Product_Type":"Snack Foods","Product_MRP":147.0,
       "Store_Id":"OUT004","Store_Establishment_Year":2009,"Store_Size":"Medium",
       "Store_Location_City_Type":"Tier 2","Store_Type":"Supermarket Type2"}'
# {"predictions": [ ... ]}
```

Free-tier Spaces sleep when idle, so the first request after a pause may take ~30 seconds to
cold-start the container.

> **Security note.** The Hugging Face token is read from the `HF_TOKEN` environment variable —
> never hard-coded. Set it before running the deployment cells:
> ```bash
> export HF_TOKEN="your_token_here"
> ```

## Key Insights & Recommendations

- **`Product_MRP` is the strongest measured driver of revenue** (r = 0.79 against the target). Revenue scales with price point — keep high-MRP products in stock and prominently displayed.
- **`Product_Weight` runs a close second** (r = 0.74), and correlates 0.53 with MRP — heavier, pricier lines carry the revenue.
- **Two attributes that do *not* move revenue.** `Product_Allocated_Area` shows essentially zero linear correlation with sales (r = −0.00), so display-area share is not supported as a sales driver in this dataset. Sugar content is the same story: median revenue is ~3,430–3,470 across `Low Sugar`, `Regular` and `No Sugar` alike. Neither justifies a merchandising change here.
- **Store size separates outlets cleanly.** Median revenue runs ≈4,100 for High, ≈3,500 for Medium and ≈1,800 for Small stores — the sharpest split among the categorical variables that were actually tested against revenue. Note that city tier and store format were never tested against revenue, so no tier-based expansion claim is made here.
- **Feed procurement directly.** Product–store level forecasts can drive automatic re-ordering, cutting both stockouts and overstock.
- **Scenario planning, with caveats.** Price is the strongest measured lever, so MRP-change simulation is the natural use case — but re-validate on a clean hold-out before acting on the numbers, given the selection bias noted below.

## Known Limitations

Documented honestly rather than papered over. These are real properties of the current notebook.

- **The final model was selected on the test set.** Cell 30 picks between the tuned models by comparing `Test_RMSE`, so the same 20% split was used for both selection and reporting. The headline test RMSE / R² are therefore optimistically biased; a three-way train/validation/test split would be the correct fix.
- **Outlier capping leaks across the split.** The IQR bounds in cell 21 are computed on the full dataframe *before* `train_test_split`, so test-set values inform the caps applied to training rows. `Store_Age` is likewise derived from the global maximum establishment year (2009). Scaling and one-hot encoding are correctly inside the `Pipeline` and refit per fold; only these pre-split steps leak.
- **No naive baseline was established.** "Baseline" in the results table means the untuned regressor. Without a mean-predictor or linear-regression floor, R² = 0.933 has nothing cheap to beat — for reference, a mean predictor would score RMSE ≈ 1,066 on this target.
- **Adjusted R² understates its own penalty.** `adj_r2_score` (cell 24) is passed `X_test`, so k = 12 raw columns rather than the ~40 columns the one-hot encoder actually produces. The adjustment is therefore too small.
- **The Streamlit UI sends categories the model never saw.** The form offers `Store_Size` = `"Low"`, which does not exist in training (levels are High / Medium / Small), and lowercases every `Product_Sugar_Content` and `Product_Type` value (`"low sugar"`, `"snack foods"`) where training used title case. `OneHotEncoder(handle_unknown='ignore')` encodes these as all-zero rows, so no error is raised — the model silently ignores those features and still returns a confident-looking number. The year field also accepts values up to 2025 against a training maximum of 2009, producing negative `Store_Age`.
- **A dirty category was never cleaned.** ~110 rows carry `reg` instead of `Regular` (cell 17). It survives into training as its own one-hot level rather than being merged.
- **Bivariate EDA is partial.** Cell 19 contains four plots. `Product_Type`, `Store_Type` and `Store_Location_City_Type` are never analysed against revenue, so no product-category or city-tier conclusion is available from this notebook.
- **No feature importances were computed.** There is no `feature_importances_`, permutation importance or SHAP analysis anywhere. Every driver statement above rests on the cell-19 correlation heatmap alone, which captures linear association only.
- **Unrendered output.** `df.describe()` is emitted through `print()` rather than as a rendered DataFrame (cell 14), so the statistical summary appears as unformatted text in `report.html`.
- **The model artefact is not versioned.** `superkart_best_model.pkl` is ~38 MB and is not committed; `deployment/` ships code only, so the backend must be rebuilt from the notebook to run locally.

## Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `pandas` · `NumPy` · `Flask` · `Streamlit` · `Docker` ·
`Hugging Face Hub` · `joblib` · `Matplotlib` · `Seaborn`

## Files

| File | Description |
|------|-------------|
| [`notebook.ipynb`](notebook.ipynb) | Full pipeline — EDA through to deployment |
| [`report.html`](report.html) | Rendered HTML report — **[view live](https://kerolous-samir.github.io/ai-ml-portfolio/05-superkart-sales-forecasting/report.html)** |
| [`deployment/`](deployment) | Extracted backend + frontend source, Dockerfiles and API reference |
