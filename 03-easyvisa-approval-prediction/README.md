# EasyVisa — US Visa Certification Prediction

> Six classifiers benchmarked across three class-balance regimes, with three grid-searched,
> to predict whether a US work-visa petition will be certified or denied — so reviewers can triage a
> growing application backlog.

[![Python](https://img.shields.io/badge/Python-Colab-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-006ACC)](https://xgboost.readthedocs.io/)
![Tuned test F1](https://img.shields.io/badge/Tuned%20test%20F1-0.817-blue)
![Majority-class baseline](https://img.shields.io/badge/majority--class%20baseline%20F1-0.801-lightgrey)

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

**25,480 petitions × 12 columns.** Target `case_status` is imbalanced: **17,018 Certified (66.8%)**
vs **8,462 Denied (33.2%)**.

| Feature group | Columns |
|---------------|---------|
| Applicant | `continent`, `education_of_employee`, `has_job_experience`, `requires_job_training` |
| Employer | `no_of_employees`, `yr_of_estab`, `region_of_employment` |
| Position | `prevailing_wage`, `unit_of_wage`, `full_time_position` |
| Target | `case_status` — Certified / Denied |

For modelling, `prevailing_wage` and `unit_of_wage` are collapsed into a derived `annual_wage`
(Hour × 2080, Week × 52, Month × 12, Year × 1) and dropped along with `case_id`, leaving nine
predictors.

## Approach

1. **EDA** — univariate distributions for four features (education, region of employment, job
   experience, case status), with `describe()` and value counts covering the rest, plus five
   bivariate comparisons against case status — education, continent, job experience, annual wage vs
   status, and
   unit of wage.
2. **Data preparation** — 33 negative `no_of_employees` values corrected with `.abs()` (cell 24);
   `annual_wage` derived from the wage unit; **IQR capping** (`clip` at the 1.5 × IQR fences) applied
   in place to `no_of_employees` and `annual_wage` (cell 56); numeric features standard-scaled and
   categoricals one-hot encoded (`drop='first'`) inside a `ColumnTransformer`; stratified 70/30 split.
3. **Class imbalance** — the same six models re-run on **oversampled** and **undersampled** copies of
   the data, built with `sklearn.utils.resample` (cell 58).
4. **Model bake-off** — Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting
   and XGBoost, each 5-fold cross-validated on **F1**.
5. **Hyperparameter tuning** — `GridSearchCV(scoring='f1', cv=3)` over the three candidates the
   notebook nominates (Random Forest, Gradient Boosting, XGBoost; cell 78 cites "baseline
   performance", though Random Forest ranks 5th of 6 on the original split and 1st only on the
   oversampled one), fitted on the **original imbalanced**
   training split. The preprocessor sits inside the `Pipeline`, so scaling and encoding are fitted
   within each CV fold. Grids: RF — `n_estimators` [100, 200], `max_features` [sqrt, log2],
   `min_samples_leaf` [1, 2, 5], `max_depth` [None, 10, 20]; GB — `n_estimators` [100, 200, 300],
   `learning_rate` [0.1, 0.05, 0.01], `max_depth` [3, 5]; XGB — `n_estimators` [100, 150],
   `learning_rate` [0.1, 0.05], `max_depth` [3, 5], `subsample` 0.8, `colsample_bytree` 0.8.
6. **Final selection** — the tuned candidate with the highest test F1 (top row of
   `tuned_models_df.sort_values('Test_F1', ascending=False)`). Grid search optimised scikit-learn's
   standard `f1` scorer, i.e. equal precision/recall weighting; no recall weighting, custom scorer,
   cost function or threshold shift was applied.

## Results

Three training regimes were benchmarked, each with **its own test set**. The tables below are
therefore *not* interchangeable — see [Known Limitations](#known-limitations).

### Baseline bake-off — original data (7,644-row imbalanced test set, 66.8% Certified)

| Rank | Model | CV F1 | Train Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 |
|:----:|-------|:-----:|:--------------:|:-------------:|:--------------:|:-----------:|:-------:|
| 1 | Gradient Boosting | 0.819 | 0.751 | 0.736 | 0.769 | 0.863 | 0.813 |
| 2 | AdaBoost | 0.815 | 0.734 | 0.726 | 0.751 | 0.882 | 0.811 |
| 3 | Logistic Regression | 0.812 | 0.730 | 0.721 | 0.748 | 0.879 | 0.808 |
| 4 | XGBoost | 0.802 | 0.839 | 0.723 | 0.765 | 0.846 | 0.803 |
| 5 | Random Forest | 0.798 | 1.000 | 0.709 | 0.758 | 0.827 | 0.791 |
| 6 | Decision Tree | 0.741 | 1.000 | 0.642 | 0.737 | 0.722 | 0.730 |

Random Forest and Decision Tree reach 0.9999 train accuracy against ~0.7 test accuracy — textbook
overfitting at default depth.

### Bake-off — undersampled data (balanced ~5,077-row test set)

| Rank | Model | CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1 |
|:----:|-------|:-----:|:-------------:|:--------------:|:-----------:|:-------:|
| 1 | Gradient Boosting | 0.703 | 0.708 | 0.706 | 0.712 | 0.709 |
| 2 | AdaBoost | 0.692 | 0.694 | 0.690 | 0.703 | 0.696 |
| 3 | Logistic Regression | 0.692 | 0.693 | 0.691 | 0.697 | 0.694 |
| 4 | Random Forest | 0.673 | 0.688 | 0.692 | 0.679 | 0.685 |
| 5 | XGBoost | 0.677 | 0.685 | 0.689 | 0.674 | 0.681 |
| 6 | Decision Tree | 0.616 | 0.625 | 0.627 | 0.621 | 0.624 |

F1 drops by roughly 0.10 versus the table above purely because the positive-class rate falls from
66.8% to 50% — not because the models got worse.

### Bake-off — oversampled data (not trustworthy, see Limitations)

| Rank | Model | CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1 |
|:----:|-------|:-----:|:-------------:|:--------------:|:-----------:|:-------:|
| 1 | Random Forest | 0.800 | 0.833 | 0.880 | 0.772 | 0.822 |
| 2 | Decision Tree | 0.767 | 0.807 | 0.858 | 0.737 | 0.793 |
| 3 | XGBoost | 0.745 | 0.759 | 0.771 | 0.738 | 0.754 |
| 4 | Gradient Boosting | 0.711 | 0.706 | 0.701 | 0.719 | 0.710 |
| 5 | AdaBoost | 0.698 | 0.688 | 0.681 | 0.707 | 0.694 |
| 6 | Logistic Regression | 0.691 | 0.687 | 0.682 | 0.701 | 0.692 |

Duplicated rows leak across this split (cell 58), so these numbers are reported for completeness
only and were not used for model selection.

### After hyperparameter tuning (original data, imbalanced 7,644-row test set)

| Model | CV Best F1 | Test Accuracy | Test Precision | Test Recall | Test F1 |
|-------|:----------:|:-------------:|:--------------:|:-----------:|:-------:|
| **Random Forest** — *selected* | **0.821** | **0.739** | 0.769 | **0.871** | **0.817** |
| Gradient Boosting | 0.820 | 0.736 | 0.768 | 0.867 | 0.815 |
| XGBoost | 0.819 | 0.737 | 0.770 | 0.865 | 0.815 |

On the same original test set, tuning lifted test F1 from **0.813** (best untuned model, Gradient
Boosting) to **0.817** (tuned Random Forest) — a gain of about **+0.4%**. Recall is essentially
unchanged at **0.871** versus 0.863 for untuned Gradient Boosting, and untuned AdaBoost already
reached **0.882**. The undersampled bake-off is scored on a different, class-balanced test set and
cannot be subtracted from these figures.

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
| 9 | Region — Midwest | 0.028 |
| 10 | Region — West | 0.018 |

These are impurity (Gini) importances from the tuned Random Forest. They rank features by magnitude
only and say nothing about the direction of the effect.

## Key Insights & Recommendations

- **Education is the single strongest signal.** Certification rates rise steeply with qualification:
  **34.0%** for High School (1,164/3,420), **62.2%** Bachelor's, **78.6%** Master's and **87.2%**
  Doctorate. "Education — High School" is also the top-ranked model feature (0.227). Employers should
  prioritise advanced-degree candidates or sponsor further education.
- **Prior job experience matters.** Applicants with experience are certified **74.5%** of the time
  versus **56.1%** without, and `has_job_experience` is the second-ranked driver (0.151) — experienced
  applicants are good candidates for a fast-track review lane.
- **Wage level is a compliance signal.** Annualised wage is the third-ranked model feature (0.138),
  and certified petitions cluster at higher annual wages than denied ones. Yearly salary offers are
  certified **69.9%** of the time against **34.6%** for hourly offers, so salaried, full-time framing
  is the stronger position. Mean annual wage also varies by region (Northeast ~$225k highest, Midwest
  ~$159k lowest), but no per-region certification comparison was run.
- **Employer profile counts, direction unknown.** Company size and age (`no_of_employees` 0.068,
  `yr_of_estab` 0.060) are the highest-ranked employer-profile features, but neither was tested against
  case status in the EDA, so no directional claim ("larger" or "longer-established") is supported here.
- **Continent is doing work in the model.** Raw certification rates range from **57.9%** (South
  America) to **79.2%** (Europe), and `continent_Europe` is the 8th-ranked feature. In a
  visa-adjudication context this needs a fairness review before any operational use (see Limitations).
- **Operational use.** The intended deployment is a triage layer — auto-prioritise high-probability
  petitions and route low-probability ones to detailed manual scrutiny, without changing decision
  authority. On current numbers that case is not yet made: the model's edge over a
  predict-everything-Certified baseline is ~1.6 F1 points, and it catches under half of denials.

## Known Limitations

Documented from the executed notebook as it stands; none of these have been fixed.

1. **The headline F1 barely beats a trivial baseline.** The test set is 66.8% Certified (5,105/7,644),
   so always predicting "Certified" scores F1 **0.801** against the tuned model's **0.817**. Denied-class
   recall is **0.47** (cell 80) — the model misses more than half of all denials, which is the class a
   triage system exists to catch.
2. **The bake-off tables are computed on different test sets and are not comparable.** Cell 72 scores on
   a balanced ~5,077-row undersampled test set; cells 78/80 score on the original 7,644-row imbalanced
   test set. No "improvement" can be obtained by subtracting one table from the other; the like-for-like
   tuning gain against the cell-68 baseline is +0.4%.
3. **The oversampled results leak.** Cell 58 oversamples the *full* dataset with
   `resample(..., replace=True)` **before** `train_test_split`, so identical duplicated minority rows
   land in both `X_train_over` and `X_test_over`. Random Forest's 0.822 F1 there is inflated by
   memorised duplicates (on the original split the same untuned model hits 0.9999 train accuracy, cell 68; cell 70 reports no train accuracy for the oversampled run). The undersampled set avoids duplication
   (`replace=False`) but is likewise carved out of the full dataset before splitting.
4. **Outlier capping precedes the split.** Cell 56 computes IQR fences on the full dataset and clips in
   place, so test rows influence the thresholds for `annual_wage` and `no_of_employees` — the 3rd- and
   5th-ranked features. The `Pipeline` fits scaling and encoding per fold, but this step sits outside it.
5. **No fairness or disparate-impact analysis.** `continent` is an active feature in a visa-approval
   model, certification rates vary from 57.9% to 79.2% across continents, and no group-level error rate,
   selection rate or disparate-impact ratio is computed anywhere in the notebook.
6. **Feature importances are impurity-based.** They are direction-agnostic and biased toward
   high-cardinality numerics (`no_of_employees` reaches 602,069 before capping). No SHAP, permutation
   importance or partial-dependence check was run, so "driver" here means "split criterion", not
   "cause".
7. **Single split, single seed, narrow margins.** Every number comes from one 70/30 split at
   `random_state=42`, with no repeats, no confidence intervals and no held-out validation set. The three
   tuned models finish within 0.002 F1 (0.817 / 0.815 / 0.815), which is well inside single-split noise,
   so the choice of Random Forest is effectively arbitrary.
8. **Winning hyperparameters are only partly recorded.** The stored `Best_Params` output in cell 78 is
   truncated by pandas; only Random Forest's `max_depth=10`, Gradient Boosting's `learning_rate=0.05`
   and XGBoost's `colsample_bytree=0.8` survive the truncation, so the
   selected configurations cannot be fully reconstructed from the artefacts in this repo.
9. **Not reproducible as-is, and nothing is persisted.** Cell 12 mounts Google Drive and reads a
   hard-coded Colab path; the CSV is not in this repo. No trained model is exported (`joblib` is
   imported only for a threading backend), so "deploy" remains aspirational. The `histogram_boxplot`
   helper (cell 28) is defined and never called.
10. **The notebook's own conclusion contradicts its results.** Cell 81 recommends "the tuned Gradient
    Boosting model" for deployment, while cell 80 selects Random Forest. The tables above follow cell 80.

## Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `joblib`

## Files

| File | Description |
|------|-------------|
| [`notebook.ipynb`](notebook.ipynb) | Full executed analysis, bake-off and tuning |
| [`report.html`](report.html) | Rendered HTML report — **[view live](https://kerolous-samir.github.io/ai-ml-portfolio/03-easyvisa-approval-prediction/report.html)** |
