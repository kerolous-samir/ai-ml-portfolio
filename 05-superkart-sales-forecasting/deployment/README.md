# SuperKart Forecasting Service — Deployment

The production artefacts for the SuperKart sales-forecasting model, extracted from the project
notebook into real, runnable files. A Flask API serves the trained scikit-learn pipeline; a
Streamlit front end calls it. Both ship as Docker containers on Hugging Face Spaces.

## Live Service

The backend is deployed and publicly callable:

**`https://kerolous-samir-superkart-backend.hf.space`**

```bash
curl -X POST https://kerolous-samir-superkart-backend.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Product_Id": "FD6114",
    "Product_Weight": 12.5,
    "Product_Sugar_Content": "low sugar",
    "Product_Allocated_Area": 0.03,
    "Product_Type": "snack foods",
    "Product_MRP": 147.0,
    "Store_Id": "OUT004",
    "Store_Establishment_Year": 2009,
    "Store_Size": "Medium",
    "Store_Location_City_Type": "Tier 2",
    "Store_Type": "Supermarket Type2"
  }'
```

```json
{"predictions": [3356.8719455112805]}
```

| Space | SDK | Link |
|-------|-----|------|
| Backend (Flask API) | Docker | [superkart-backend](https://huggingface.co/spaces/kerolous-samir/superkart-backend) |
| Frontend (Streamlit) | Docker | [superkart-frontend](https://huggingface.co/spaces/kerolous-samir/superkart-frontend) |

> Free-tier Spaces sleep when idle — the first request after a pause may take ~30 seconds to
> cold-start the container.

## Architecture

```
  Browser
     │
     ▼
┌─────────────────────┐        ┌──────────────────────┐
│ Streamlit frontend  │ POST   │  Flask backend       │
│ (Docker Space)      ├───────▶│  (Docker Space)      │
│ form → JSON payload │ /predict│  gunicorn :7860     │
└─────────────────────┘        └──────────┬───────────┘
                                          │ joblib.load
                                          ▼
                             ┌──────────────────────────┐
                             │ superkart_best_model.pkl │
                             │ tuned RandomForest inside│
                             │ a full sklearn Pipeline  │
                             └──────────────────────────┘
```

## API

### `GET /`
Health check. Returns a plain-text banner.

### `POST /predict`
Accepts a single JSON object or an array of objects. Returns `{"predictions": [...]}`.

**Required fields**

| Field | Type | Example |
|-------|------|---------|
| `Product_Id` | string | `"FD6114"` |
| `Product_Weight` | float | `12.5` |
| `Product_Sugar_Content` | string | `"low sugar"` \| `"regular"` \| `"no sugar"` |
| `Product_Allocated_Area` | float | `0.03` |
| `Product_Type` | string | `"snack foods"` |
| `Product_MRP` | float | `147.0` |
| `Store_Id` | string | `"OUT004"` |
| `Store_Establishment_Year` | int | `2009` |
| `Store_Size` | string | `"Medium"` |
| `Store_Location_City_Type` | string | `"Tier 2"` |
| `Store_Type` | string | `"Supermarket Type2"` |

Errors return `{"error": "..."}` with status `400` (bad input) or `500` (inference failure).

## Serving-Time Feature Engineering

The API recreates the training pipeline's derived features before calling the model. Getting this
wrong is the most common source of train/serve skew, so each is deliberate:

| Feature | Rule | Why |
|---------|------|-----|
| `Product_Category` | first 2 characters of `Product_Id` | matches training |
| `Store_Age` | **`2009 - Store_Establishment_Year`** | 2009 is the training set's max establishment year, **frozen as a constant**. Training derived it via `.max()`; recomputing that on a one-row request would make `Store_Age` always 0 and silently skew every prediction |
| `MRP_per_weight` | `Product_MRP / Product_Weight` | zero weights coerced to `NaN` rather than dividing by zero |
| `Product_Id`, `Store_Id` | dropped | identifiers, not predictors |

## Running Locally

The model artefact (`superkart_best_model.pkl`) is produced by the project notebook and is not
committed here — train it first, or drop your own copy into `backend/`.

**Backend**

```bash
cd backend
pip install -r requirements.txt
# place superkart_best_model.pkl in this directory
gunicorn --bind 0.0.0.0:7860 app:app
```

**Frontend**

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Point `BACKEND_URL` in `frontend/app.py` at your backend before running.

**Docker**

```bash
cd backend && docker build -t superkart-api . && docker run -p 7860:7860 superkart-api
```

## Files

```
deployment/
├── backend/
│   ├── app.py            # Flask API, feature engineering, /predict
│   ├── Dockerfile        # python:3.9-slim + gunicorn on :7860
│   └── requirements.txt  # pinned versions
└── frontend/
    ├── app.py            # Streamlit input form → backend call
    ├── Dockerfile
    └── requirements.txt
```
