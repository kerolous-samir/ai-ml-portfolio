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

---

## Projects

| # | Project | Domain | Problem Type | Key Techniques | Headline Result |
|:-:|---------|--------|--------------|----------------|-----------------|
| 01 | **[FoodHub Order Analysis](01-foodhub-data-analysis)** | Food delivery | Exploratory data analysis | pandas, univariate/bivariate analysis, groupby aggregation, visual storytelling | Quantified demand drivers across 1.9K orders; identified 29% of revenue concentrated in high-value orders |
| 02 | **[AllLife Bank Loan Targeting](02-alllife-bank-loan-targeting)** | Banking / marketing | Binary classification | Decision trees, cost-complexity (post) pruning, GridSearchCV pre-pruning, ROC-AUC | **98.9% accuracy, 0.997 ROC-AUC** with a post-pruned, fully interpretable tree |
| 03 | **[EasyVisa Approval Prediction](03-easyvisa-approval-prediction)** | Immigration / HR | Binary classification | Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost, class balancing, pipelines | **0.82 F1 / 0.87 recall** with a tuned Random Forest across 6 benchmarked models |
| 04 | **[ReneWind Predictive Maintenance](04-renewind-predictive-maintenance)** | Renewable energy | Imbalanced classification | **Neural network implemented from scratch in NumPy** — SGD & Adam, dropout, class weighting | **0.8145 F1** at 98.2% accuracy on highly imbalanced turbine-failure data |
| 05 | **[SuperKart Sales Forecasting](05-superkart-sales-forecasting)** | Retail | Regression + **deployment** | Random Forest, XGBoost, sklearn Pipelines, **Flask API, Streamlit UI, Docker, Hugging Face Spaces** | **R² 0.933, RMSE ≈ 277** — served through a deployable API and web UI |
| 06 | **[HelmNet Helmet Detection](06-helmnet-helmet-detection)** | Workplace safety | Computer vision | CNNs from scratch, data augmentation, **VGG16 transfer learning**, OpenCV | **100% test accuracy** (95 held-out images) with VGG16 transfer learning |
| 07 | **[Medical RAG Assistant](07-medical-rag-assistant)** | Healthcare / NLP | Retrieval-augmented generation | TF-IDF retrieval, nearest-neighbour search, prompt engineering, PDF corpus ingestion, groundedness evaluation | Grounded clinical Q&A with retrieval-depth (`k`) and prompt-variant ablations |

---

## Skills Demonstrated

**Languages & Core** &nbsp;·&nbsp; Python · NumPy · pandas · SQL-style aggregation

**Machine Learning** &nbsp;·&nbsp; Logistic Regression · Decision Trees · Random Forest · AdaBoost ·
Gradient Boosting · XGBoost · Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) ·
Cross-validation · Class imbalance handling (under/over-sampling, class weights) · sklearn Pipelines

**Deep Learning** &nbsp;·&nbsp; TensorFlow / Keras · CNN architecture design · Transfer learning (VGG16) ·
Dropout & regularisation · Data augmentation · Backpropagation implemented from first principles

**NLP & GenAI** &nbsp;·&nbsp; Retrieval-augmented generation · TF-IDF & vector retrieval · Prompt engineering ·
Transformers · Output groundedness/relevance evaluation

**MLOps & Deployment** &nbsp;·&nbsp; Flask REST API · Streamlit · Docker · Hugging Face Spaces ·
Model serialisation (joblib) · Reproducible environments

**Visualisation & Communication** &nbsp;·&nbsp; Matplotlib · Seaborn · Translating model output into
business recommendations

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

## License

Released under the [MIT License](LICENSE).

## Contact

**Kerolous Samir** — [GitHub](https://github.com/kerolous-samir)

If any of this is useful to you, a ⭐ on the repo is always appreciated.
