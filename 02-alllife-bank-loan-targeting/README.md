# AllLife Bank — Personal Loan Campaign Targeting

> A fully interpretable decision-tree model that identifies which liability (deposit) customers
> are most likely to convert into personal-loan customers.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Decision%20Trees-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
![Type](https://img.shields.io/badge/Type-Binary%20Classification-blue)
![ROC AUC](https://img.shields.io/badge/ROC--AUC-0.997-brightgreen)

## Business Context

AllLife Bank has a growing customer base that is overwhelmingly made up of **liability customers**
(depositors). Its **asset customer** (borrower) base is small, and management wants to grow it to
earn more interest income — while retaining those customers as depositors.

A campaign run the previous year converted just over **9%** of targeted liability customers. The
retail marketing department wants better targeting to lift that conversion ratio.

## Objective

Predict whether a liability customer will accept a personal loan, identify which customer
attributes drive that decision, and define the segments the marketing team should prioritise.

## Dataset

**5,000 customers × 14 attributes.** Target: `Personal_Loan` (1 = accepted the offer).

| Column | Description |
|--------|-------------|
| `Age`, `Experience` | Age and years of professional experience |
| `Income` | Annual income (thousand USD) |
| `Family` | Family size |
| `CCAvg` | Average monthly credit-card spend (thousand USD) |
| `Education` | 1 = Undergrad, 2 = Graduate, 3 = Advanced/Professional |
| `Mortgage` | Value of house mortgage, if any |
| `Securities_Account`, `CD_Account`, `Online`, `CreditCard` | Existing product relationships |
| `Personal_Loan` | **Target** — accepted the loan in the last campaign |

## Approach

1. **Data quality** — confirmed zero missing values; corrected **52 negative `Experience`** entries.
2. **EDA** — distribution and relationship analysis of income, education, family size and product holdings against loan uptake.
3. **Preprocessing** — dropped identifier columns, encoded categoricals, stratified train/test split.
4. **Baseline model** — unconstrained decision tree.
5. **Regularisation, two ways** —
   - **Post-pruning** via cost-complexity pruning, sweeping `ccp_alpha`.
   - **Pre-pruning** via `GridSearchCV` over `max_depth`, `min_samples_split`, `min_samples_leaf`.
6. **Selection** — compared on accuracy and ROC-AUC, then extracted human-readable decision rules.

## Results

| Model | Accuracy | ROC-AUC |
|-------|:--------:|:-------:|
| Baseline decision tree | 0.981 | 0.949 |
| Pre-pruned (`max_depth=5, min_samples_split=20, min_samples_leaf=1`) | 0.985 | 0.997 |
| **Post-pruned (`ccp_alpha = 0.00093`)** — *selected* | **0.989** | **0.997** |

The post-pruned tree was selected: it delivers the best accuracy, matches the best ROC-AUC, and
stays shallow enough for the marketing team to read the rules directly.

### Learned decision logic (top levels)

```
Income <= 98.5
├── CCAvg <= 2.95                          → no loan
└── CCAvg > 2.95
    ├── CD_Account = 0                     → no loan
    └── CD_Account = 1                     → LOAN
Income > 98.5
├── Education = Undergrad
│   ├── Family <= 2                        → no loan
│   └── Family > 2 and Income > 113.5      → LOAN
└── Education = Graduate / Professional
    ├── Income <= 114.5 and CCAvg > 2.45   → LOAN
    └── Income > 114.5                     → LOAN
```

**Income is the dominant split**, followed by education, credit-card spend and family size.

## Key Insights & Recommendations

- **High income & education** — customers earning **> $80k** with graduate or professional degrees show the strongest propensity. Target them with premium loan offers.
- **Family size** — families of **3–4** are more receptive than singles; use life-stage messaging (education, home improvement).
- **Cross-sell** — holding a **CD account** or credit card lifts uptake. Bundle a preferential CD rate with a pre-approved loan.
- **Digital is not enough** — the `Online` flag contributes only marginally; pair email/SMS with branch or relationship-manager follow-up.
- **Operationalise** — target the top-scoring deciles, personalise messaging on income and family cues, and **retrain quarterly** to track behavioural drift.

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Files

| File | Description |
|------|-------------|
| [`notebook.ipynb`](notebook.ipynb) | Full executed analysis and modelling |
| [`report.html`](report.html) | Rendered HTML report — **[view live](https://kerolous-samir.github.io/ai-ml-portfolio/02-alllife-bank-loan-targeting/report.html)** |
