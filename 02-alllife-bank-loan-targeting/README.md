# AllLife Bank — Personal Loan Campaign Targeting

> A fully interpretable decision-tree model that identifies which liability (deposit) customers
> are most likely to convert into personal-loan customers.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
![Type](https://img.shields.io/badge/Type-Binary%20Classification-blue)
![Positive class](https://img.shields.io/badge/Positive%20class-9.6%25-lightgrey)

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

**5,000 customers × 14 attributes.** Target: `Personal_Loan` (1 = accepted the offer), positive in
**9.6%** of rows (480 of 5,000). No missing values in any column.

| Column | Description |
|--------|-------------|
| `Age`, `Experience` | Age and years of professional experience |
| `Income` | Annual income (thousand USD) — median 64, 75th percentile 98, max 224 |
| `Family` | Family size (1–4) |
| `CCAvg` | Average monthly credit-card spend (thousand USD) — median 1.5, max 10.0 |
| `Education` | 1 = Undergrad, 2 = Graduate, 3 = Advanced/Professional |
| `Mortgage` | Value of house mortgage, if any — median 0, max 635 |
| `Securities_Account`, `CD_Account`, `Online`, `CreditCard` | Existing product relationships (0/1) |
| `Personal_Loan` | **Target** — accepted the loan in the last campaign |

`ID` and `ZIPCode` are identifiers and are dropped before modelling.

## Approach

1. **Data quality** — confirmed zero missing values; corrected **52 negative `Experience`** entries
   to the median of the non-negative values.
2. **EDA** — distribution of `Mortgage`, counts of credit-card holders (1,470 of 5,000 = 29.4%),
   and a bivariate comparison of `Income` against loan uptake.
3. **Preprocessing** — dropped identifier columns (`ID`, `ZIPCode`); all remaining features are
   already numeric (`Education` is ordinal 1/2/3, product flags are 0/1), so no encoding was
   required; 70/30 stratified train/test split (`random_state=42`).
4. **Baseline model** — unconstrained decision tree.
5. **Regularisation, two ways** —
   - **Post-pruning** via cost-complexity pruning, sweeping `ccp_alpha`.
   - **Pre-pruning** via `GridSearchCV` (5-fold, scored on ROC-AUC) over `max_depth`,
     `min_samples_split`, `min_samples_leaf`.
6. **Selection** — compared on accuracy and ROC-AUC, then exported human-readable decision rules
   and feature importances from the winner.

## Results

All figures are on the 1,500-row held-out test split (≈144 positives).

| Model | Accuracy | ROC-AUC | Test errors |
|-------|:--------:|:-------:|:-----------:|
| Always predict "no loan" † | 0.904 | — | 144 / 1500 |
| Baseline decision tree | 0.981 | 0.949 | 29 / 1500 |
| Pre-pruned (`max_depth=5, min_samples_split=20, min_samples_leaf=1`) | 0.985 | 0.997 | 23 / 1500 |
| **Post-pruned (`ccp_alpha = 0.00093`)** — *selected* ‡ | **0.989** | **0.997** | 17 / 1500 |

† Reference point, not computed in the notebook — it follows directly from the 9.6% base rate.
Accuracy on this problem starts at ~90% for free, which is why the headline numbers below look
larger than the lift they represent.

‡ The selection code picks on ROC-AUC (`post_pruned_auc >= pre_pruned_auc`; 0.9968 vs 0.9966).
**The `ccp_alpha` for this model was chosen by maximising accuracy on the test set**, so its 0.989 /
0.997 are optimistic — see [Known Limitations](#known-limitations). The pre-pruned model, tuned by
cross-validation on the training set alone, is the only *tuned* model with a clean held-out
score. The untuned baseline's 0.981 / 0.949 are clean too; only the selected post-pruned row saw
the test set during model selection.

### Learned decision logic (complete tree, 9 leaves)

```
Income <= 98.5
├── CCAvg <= 2.95                          → no loan
└── CCAvg > 2.95
    ├── CD_Account = 0                     → no loan
    └── CD_Account = 1                     → LOAN
Income > 98.5
├── Education = Undergrad
│   ├── Family <= 2                        → no loan
│   └── Family > 2
│       ├── Income <= 113.5                → no loan
│       └── Income > 113.5                 → LOAN
└── Education = Graduate / Professional
    ├── Income <= 114.5 and CCAvg <= 2.45  → no loan
    ├── Income <= 114.5 and CCAvg > 2.45   → LOAN
    └── Income > 114.5                     → LOAN
```

**Income is the dominant split**, followed by education, family size and credit-card spend.
Feature importances read off the final model's plot: `Income` ≈ 0.47, `Education` ≈ 0.34,
`Family` ≈ 0.15, `CCAvg` ≈ 0.04, `CD_Account` ≈ 0.01. Every other feature — `Age`, `Experience`,
`Mortgage`, `Securities_Account`, `Online`, `CreditCard` — has importance **0** and appears nowhere
in the tree.

## Key Insights & Recommendations

- **High income & education** — the model's root split is `Income <= 98.5` (≈ the 75th percentile of
  income). Customers with a graduate or professional degree earning **> $114.5k** are an
  unconditional LOAN leaf; between **$98.5k and $114.5k** they also need `CCAvg > 2.45`. Target that
  group with premium loan offers.
- **Family size matters only in one branch** — among **undergraduates** earning > $98.5k, the tree
  predicts LOAN only for `Family > 2` **and** `Income > 113.5`. Family size is the third-strongest
  driver overall (≈0.15 importance) but it never acts on its own.
- **CD account is the one useful cross-sell signal** — below the $98.5k income line, the only route
  to a LOAN leaf is `CCAvg > 2.95` **and** `CD_Account = 1`. Bundling a preferential CD rate with a
  pre-approved loan is the defensible play. `CreditCard` ownership, by contrast, has **zero**
  importance in this model and should not be used for targeting.
- **Digital channel is not a signal** — the `Online` flag has zero feature importance and is never
  split on. It carries no targeting information here; channel strategy has to come from elsewhere.
- **Operationalise the rules, don't score deciles** — this is a 9-leaf tree, so it emits a handful of
  distinct probabilities, not a smooth score. Ship it as four eligibility rules the marketing team
  can read, measure conversion per rule in a live campaign, and refit as behaviour drifts.
- **Validate before spending** — the reported lift comes from a model tuned against its own test
  set. Re-fit with clean, nested validation and re-measure before committing campaign budget.

## Known Limitations

Stated plainly, because they bound how far the numbers above can be trusted. **The notebook has not
been corrected; this section documents it as it stands.**

- **Test-set selection (leakage).** Cell 27 sweeps every `ccp_alpha` from the pruning path and keeps
  the one that maximises `accuracy_score(y_test, clf.predict(X_test))`. The test set was therefore
  used to choose a hyperparameter, and the selected model's 0.989 accuracy / 0.997 ROC-AUC are
  optimistic. There is no validation split and no nested cross-validation.
- **Accuracy is the wrong headline metric.** With a 9.6% positive rate, always predicting "no loan"
  scores ~90.4%. The 0.981 → 0.989 range across all three models is 29 vs 17 errors out of 1,500 —
  a real but small lift that accuracy makes look larger than it is.
- **No positive-class metrics at all.** `classification_report` and `confusion_matrix` are imported
  in cell 7 and never called. Precision, recall and F1 for loan acceptors — the numbers a campaign
  actually depends on — are never reported.
- **The evaluation criterion was never written.** Markdown cell 23, under the heading "Model
  Evaluation Criterion", contains a single stray `*`. The rationale for whether a false positive
  (wasted contact) or a false negative (missed loan) is costlier to the bank is simply absent, so no
  metric choice in the notebook is justified.
- **EDA is a stub relative to its own brief.** Cell 16 is the only EDA code cell; it produces three
  artifacts and ends with the comment `# Additional EDA as per rubric can be added similarly`. Its
  `labeled_barplot` helper is defined and never called. The correlation table, age-bracket uptake
  rates and education uptake rates written in markdown cell 17 have **no generating code and no
  output anywhere in the notebook** — they are unverified prose, and none of them are repeated here.
- **Claimed preprocessing steps that were not performed.** Cell 19 scopes outlier detection and
  treatment; `Mortgage` (median 0, max 635) and `CCAvg` are heavily right-skewed and nothing was
  done about either. Trees are largely insensitive to this, so the practical cost is low — but the
  step was listed, not executed.
- **Single split, single seed.** One 70/30 split at `random_state=42`, with no repeated splits or
  confidence intervals. On 1,500 test rows a six-error difference is well inside the noise, so the
  ranking of the three models is not established.
- **Not reproducible from a clone.** Cell 9 mounts Google Drive and reads
  `/content/drive/My Drive/Loan_Modelling.csv`. The dataset is not redistributed in this repository,
  so the notebook cannot be re-run as written without editing the path.
- **Misleading caption in cell 29.** The rules block is printed under "Decision Rules for final
  model (truncated to depth 3)", but the code slices the first 30 lines of `export_text` and the
  tree is only 25 lines long — the output is actually the complete tree, not a depth-3 excerpt.

- **The notebook's own insights section is stale.** Markdown cell 31 still recommends targeting
  customers earning > $80k and scoring "top-scoring deciles". Neither is supported by the selected
  model: its income splits are 98.5 / 113.5 / 114.5, and a 9-leaf tree emits only a handful of
  distinct probabilities, so decile ranking is not meaningful. That cell has not been corrected.

## Tech Stack

`Python 3.10` · `scikit-learn 1.2.2` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Files

| File | Description |
|------|-------------|
| [`notebook.ipynb`](notebook.ipynb) | Full executed analysis and modelling |
| [`report.html`](report.html) | Rendered HTML report — **[view live](https://kerolous-samir.github.io/ai-ml-portfolio/02-alllife-bank-loan-targeting/report.html)** |
