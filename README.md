# AI & Machine Learning Portfolio

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Reports](https://img.shields.io/badge/Live-Reports-2ea44f?logo=github)](https://kerolous-samir.github.io/ai-ml-portfolio/)

End-to-end applied machine learning projects spanning **exploratory data analysis, supervised
learning, ensemble methods, neural networks, computer vision, model deployment, and
retrieval-augmented generation (RAG)**.

Every project follows the same lifecycle: business framing → data quality assessment → EDA →
preprocessing → model building → hyperparameter tuning → evaluation against a business-aligned
metric → actionable recommendations.

### 🟢 Try the deployed model right now

Project 05 is live in production on Hugging Face Spaces. No signup, no clone:

```bash
curl -X POST https://kerolous-samir-superkart-backend.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"Product_Id":"FD6114","Product_Weight":12.5,"Product_Sugar_Content":"low sugar",
       "Product_Allocated_Area":0.03,"Product_Type":"snack foods","Product_MRP":147.0,
       "Store_Id":"OUT004","Store_Establishment_Year":2009,"Store_Size":"Medium",
       "Store_Location_City_Type":"Tier 2","Store_Type":"Supermarket Type2"}'
# → {"predictions":[3356.8719455112805]}
```

The Flask API, Streamlit UI and both Dockerfiles are real, reviewable files under
[`05-superkart-sales-forecasting/deployment/`](05-superkart-sales-forecasting/deployment) —
not strings inside a notebook.

### On the numbers in this README

Every metric below is taken from the committed notebook outputs. Where a result is inflated by a
methodological flaw — test-set selection, resampling before the split, probable image leakage —
the table says so, and each project's README carries a **Known Limitations** section spelling out
exactly what is and is not trustworthy. Headline numbers without that context would be easy to
write and impossible to defend.

---

## Projects

| # | Project | Domain | Problem Type | Key Techniques | Headline Result |
|:-:|---------|--------|--------------|----------------|-----------------|
| 01 | **[FoodHub Order Analysis](01-foodhub-data-analysis)** | Food delivery | Exploratory data analysis | pandas, univariate/bivariate analysis, groupby aggregation, commission-tier modelling | Demand and fulfilment patterns across **1,898 orders**; preparation time (≈27 min) exceeds delivery (≈24 min) |
| 02 | **[AllLife Bank Loan Targeting](02-alllife-bank-loan-targeting)** | Banking / marketing | Binary classification | Decision trees, cost-complexity (post) pruning, GridSearchCV pre-pruning, ROC-AUC | Interpretable tree, **0.997 ROC-AUC**; headline accuracy is *test-set-selected* and the 9.6% base rate makes accuracy weak — see Known Limitations |
| 03 | **[EasyVisa Approval Prediction](03-easyvisa-approval-prediction)** | Immigration / HR | Binary classification | Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, class balancing, pipelines | **0.817 test F1** (tuned Random Forest) across 6 benchmarked models and 3 class-balance regimes; resampling preceded the split, so tuned scores are leakage-inflated |
| 04 | **[ReneWind Predictive Maintenance](04-renewind-predictive-maintenance)** | Renewable energy | Imbalanced classification | **Neural network implemented from scratch in NumPy** — manual backprop, SGD & Adam, dropout, class weighting | **0.8145 F1** at 0.944 precision / 0.716 recall; 7-model ablation, selected arm runs *unweighted* |
| 05 | **[SuperKart Sales Forecasting](05-superkart-sales-forecasting)** | Retail | Regression + **live deployment** | Random Forest, XGBoost, sklearn Pipelines, **Flask API, Streamlit UI, Docker, Hugging Face Spaces** | **R² 0.933, RMSE ≈ 277** — and a **[live, callable API](https://kerolous-samir-superkart-backend.hf.space)** |
| 06 | **[HelmNet Helmet Detection](06-helmnet-helmet-detection)** | Workplace safety | Computer vision | Custom CNN vs 3 **VGG16 transfer-learning** variants, augmentation, OpenCV | 95/95 on the test split — but three different architectures all scored flawlessly, which points to **near-duplicate leakage**, not generalisation |
| 07 | **[Medical RAG Assistant](07-medical-rag-assistant)** | Healthcare / NLP | Retrieval prototype | TF-IDF + NearestNeighbors retrieval, PDF corpus ingestion, prompt & `k` ablations | Retriever works (**top-1 on 5/5 questions**); the generator is **not yet conditioned on retrieved context** — documented, not hidden |

---

## Skills Demonstrated

**Languages & Core** &nbsp;·&nbsp; Python · NumPy · pandas · SQL-style aggregation

**Machine Learning** &nbsp;·&nbsp; Logistic Regression · Decision Trees · Random Forest · AdaBoost ·
Gradient Boosting · XGBoost · Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) ·
Cross-validation · Class imbalance handling (under/over-sampling, class weights) · sklearn Pipelines

**Deep Learning** &nbsp;·&nbsp; TensorFlow / Keras · CNN architecture design · Transfer learning (VGG16) ·
Dropout & regularisation · Data augmentation · Backpropagation implemented from first principles

**NLP & Retrieval** &nbsp;·&nbsp; TF-IDF retrieval · NearestNeighbors search · PDF corpus ingestion ·
Prompt-variant and retrieval-depth ablations · Critical evaluation of retrieval/generation wiring

**MLOps & Deployment** &nbsp;·&nbsp; Flask REST API · Streamlit · Docker · Hugging Face Spaces ·
Model serialisation (joblib) · Train/serve skew control (frozen training-time constants)

**Visualisation & Communication** &nbsp;·&nbsp; Matplotlib · Seaborn · Translating model output into
business recommendations

**Evaluation & Rigour** &nbsp;·&nbsp; Identifying test-set selection and resampling leakage ·
Recognising when accuracy misleads under class imbalance · Auditing one's own results and
publishing the caveats

---

## Repository Structure

```
ai-ml-portfolio/
├── 01-foodhub-data-analysis/
│   ├── README.md          # business problem, approach, findings
│   ├── notebook.ipynb     # executed notebook with full output
│   └── report.html        # standalone rendered report
├── 02-alllife-bank-loan-targeting/
├── 03-easyvisa-approval-prediction/
├── 04-renewind-predictive-maintenance/
├── 05-superkart-sales-forecasting/
│   ├── README.md
│   ├── notebook.ipynb
│   ├── report.html
│   └── deployment/        # real, runnable service code
│       ├── README.md      # live endpoint + API contract
│       ├── backend/       # Flask API + Dockerfile
│       └── frontend/      # Streamlit UI + Dockerfile
├── 06-helmnet-helmet-detection/
├── 07-medical-rag-assistant/
├── requirements.txt
└── LICENSE
```

Each project folder contains the same three artefacts:

- **`README.md`** — the business context, methodology, results and recommendations.
- **`notebook.ipynb`** — the full executed notebook, rendered directly by GitHub.
- **`report.html`** — a self-contained HTML report that opens in any browser, no Python needed.

---

## Viewing the Notebooks

GitHub renders `.ipynb` files natively — just click any `notebook.ipynb` above.

If a large notebook times out in GitHub's viewer, open it through
[nbviewer](https://nbviewer.org/) by pasting the file URL.

Every project also ships a rendered HTML report hosted on GitHub Pages — no download,
no Python, just a link:

| Project | Live report |
|---------|-------------|
| FoodHub Order Analysis | [view](https://kerolous-samir.github.io/ai-ml-portfolio/01-foodhub-data-analysis/report.html) |
| AllLife Bank Loan Targeting | [view](https://kerolous-samir.github.io/ai-ml-portfolio/02-alllife-bank-loan-targeting/report.html) |
| EasyVisa Approval Prediction | [view](https://kerolous-samir.github.io/ai-ml-portfolio/03-easyvisa-approval-prediction/report.html) |
| ReneWind Predictive Maintenance | [view](https://kerolous-samir.github.io/ai-ml-portfolio/04-renewind-predictive-maintenance/report.html) |
| SuperKart Sales Forecasting | [view](https://kerolous-samir.github.io/ai-ml-portfolio/05-superkart-sales-forecasting/report.html) |
| HelmNet Helmet Detection | [view](https://kerolous-samir.github.io/ai-ml-portfolio/06-helmnet-helmet-detection/report.html) |
| Medical RAG Assistant | [view](https://kerolous-samir.github.io/ai-ml-portfolio/07-medical-rag-assistant/report.html) |

## Running Locally

```bash
git clone https://github.com/kerolous-samir/ai-ml-portfolio.git
cd ai-ml-portfolio

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab
```

> **A note on data.** The datasets used in these projects are distributed as part of the
> coursework and are not redistributed here. Each notebook documents its data dictionary and the
> expected file, so the analysis is fully auditable. Notebooks load data from a local path —
> update that path to point at your own copy before re-running.

---
These servers ran in production inside a multi-tenant virtualization platform before being extracted, de-branded and released here.
---

## License

Released under the [MIT License](LICENSE).

## Contact

**Kerolous Samir** — [GitHub](https://github.com/kerolous-samir)

If any of this is useful to you, a ⭐ on the repo is always appreciated.
