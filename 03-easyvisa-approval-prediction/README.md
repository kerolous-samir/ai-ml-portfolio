# EasyVisa — US Visa Certification Prediction

> Six ensemble classifiers benchmarked and tuned to predict whether a US work-visa petition will
> be certified or denied, so reviewers can triage a growing application backlog.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Ensembles-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-006ACC)](https://xgboost.readthedocs.io/)
![F1](https://img.shields.io/badge/Test%20F1-0.817-brightgreen)

## Business Context

The US Department of Labor's **Office of Foreign Labor Certification (OFLC)** processes hundreds of
thousands of employer applications each year — **775,979 applications covering 1,699,957 positions
in FY2016 alone**, a 9% year-on-year increase. Reviewing every case manually is slow and
resource-intensive.

OFLC engaged EasyVisa to build a data-driven solution that shortlists applicants with a higher
probability of approval and surfaces the factors that drive certification.

## Objective

Predict the visa case status (**Certified** or **Denied**) from employer and applicant attributes,
and identify the drivers that most influence the outcome.

## Dataset

Employer and applicant attributes per petition. Target: `case_status`.

| Feature group | Columns |
|---------------|---------|
| Applicant | `continent`, `education_of_employee`, `has_job_experience`, `requires_job_training` |
| Employer | `no_of_employees`, `yr_of_estab`, `region_of_employment` |
| Position | `prevailing_wage`, `unit_of_wage`, `full_time_position` |
| Target | `case_status` — Certified / Denied |

## Approach

1. **EDA** — univariate and bivariate analysis of every feature against case status.
2. **Data preparation** — outlier assessment, one-hot encoding of categoricals, stratified split.
3. **Class imbalance** — models benchmarked on **undersampled** and **oversampled** training data.
4. **Model bake-off** — Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting and XGBoost, each cross-validated on **F1**.
5. **Hyperparameter tuning** — grid search over the three strongest candidates, inside an sklearn `Pipeline` so preprocessing is fitted within each CV fold (no leakage).
6. **Final selection** — chosen on test F1 with recall weighted heavily, since missing a certifiable applicant is the costlier error.

## Results

### Model bake-off (undersampled data)

| Rank | Model | CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1 |
|:----:|-------|:-----:|:-------------:|:--------------:|:-----------:|:-------:|
| 1 | Gradient Boosting | 0.703 | 0.708 | 0.706 | 0.712 | 0.709 |
| 2 | AdaBoost | 0.692 | 0.694 | 0.690 | 0.703 | 0.696 |
| 3 | Logistic Regression | 0.692 | 0.693 | 0.691 | 0.697 | 0.694 |
| 4 | Random Forest | 0.673 | 0.688 | 0.692 | 0.679 | 0.685 |
| 5 | XGBoost | 0.677 | 0.685 | 0.689 | 0.674 | 0.681 |
| 6 | Decision Tree | 0.616 | 0.625 | 0.627 | 0.621 | 0.624 |

### After hyperparameter tuning

| Model | CV Best F1 | Test Accuracy | Test Precision | Test Recall | Test F1 |
|-------|:----------:|:-------------:|:--------------:|:-----------:|:-------:|
| **Random Forest** — *selected* | **0.821** | **0.739** | 0.769 | **0.871** | **0.817** |
| XGBoost | 0.819 | 0.737 | 0.770 | 0.865 | 0.815 |
| Gradient Boosting | 0.820 | 0.736 | 0.768 | 0.867 | 0.815 |

Tuning lifted F1 from **0.709 → 0.817** (+15%), driven mainly by a jump in recall to **0.871** —
the model now catches the large majority of certifiable applications.

**Final model — classification report**

```
              precision    recall  f1-score   support
   Denied (0)      0.65      0.47      0.55      2539
Certified (1)      0.77      0.87      0.82      5105
     accuracy                          0.74      7644
```

### Top drivers of certification

| Rank | Feature | Importance |
|:----:|---------|:----------:|
| 1 | Education — High School | 0.227 |
| 2 | Has job experience (Y) | 0.151 |
| 3 | Annual wage | 0.138 |
| 4 | Education — Master's | 0.114 |
| 5 | Number of employees | 0.068 |
| 6 | Year employer established | 0.060 |
| 7 | Education — Doctorate | 0.049 |
| 8 | Continent — Europe | 0.046 |

## Key Insights & Recommendations

- **Education is the single strongest signal.** High-school-only education is the dominant negative predictor, while Master's and Doctorate holders are markedly more likely to be certified. Employers should prioritise advanced-degree candidates or sponsor further education.
- **Prior job experience matters.** `has_job_experience` is the second-ranked driver — experienced applicants should be routed to a fast-track review lane.
- **Wage level is a compliance signal.** Prevailing wage ranks third; petitions at or above the regional benchmark fare better.
- **Employer profile counts.** Larger, longer-established employers see higher certification rates.
- **Operational use.** Deploy the model as a triage layer: auto-prioritise high-probability petitions for fast review and route low-probability ones to detailed manual scrutiny, cutting reviewer load without changing decision authority.

## Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `joblib`

## Files

| File | Description |
|------|-------------|
| [`notebook.ipynb`](notebook.ipynb) | Full executed analysis, bake-off and tuning |
| [`report.html`](report.html) | Rendered HTML report — **[view live](https://kerolous-samir.github.io/ai-ml-portfolio/03-easyvisa-approval-prediction/report.html)** |
