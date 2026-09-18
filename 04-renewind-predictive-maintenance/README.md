# ReneWind — Wind Turbine Failure Prediction

> A neural network **implemented from scratch in NumPy** — forward pass, backpropagation, two
> optimisers, dropout and class weighting — applied to imbalanced sensor data to predict
> generator failures before they happen.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch%20NN-013243?logo=numpy&logoColor=white)](https://numpy.org/)
![Type](https://img.shields.io/badge/Type-Imbalanced%20Classification-blue)
![F1](https://img.shields.io/badge/Test%20F1-0.8145-brightgreen)
![Recall](https://img.shields.io/badge/Test%20Recall-0.7163-orange)

## Business Context

Wind is one of the most mature renewable technologies, and the US Department of Energy promotes
**predictive maintenance** as a route to operational efficiency. The premise: failure patterns are
predictable, and replacing a component *before* it fails is far cheaper than replacing it after.

**ReneWind** collects sensor readings across its turbines — environmental factors (temperature,
humidity, wind speed) and component telemetry (gearbox, tower, blades, brake). The data is shared
in **ciphered form** because sensor schemas are commercially confidential.

## Objective

Build and tune classification models to identify impending generator failures so components can be
repaired before breaking, minimising total maintenance cost.

### The cost asymmetry that drives the metric

| Prediction | Meaning | Cost incurred |
|------------|---------|---------------|
| **True positive** | Failure correctly predicted | Repair cost — *moderate* |
| **False negative** | Real failure, missed | **Replacement cost — highest** |
| **False positive** | Predicted failure, none occurred | Inspection cost — *lowest* |

Because a missed failure is the most expensive outcome, **recall is prioritised** and models are
selected on **F1**, not accuracy — accuracy is misleading on data this imbalanced. The cost ordering
is given qualitatively in the brief; no currency figures are supplied, so no expected-cost model is
computed in this notebook.

## Dataset

- **40 ciphered predictors** (`V1`–`V40`) plus the binary `Target` — 41 columns
- **20,000 training** observations, **5,000 test** observations
- Split 80/20 with stratification into **16,000 training** and **4,000 validation** rows
- Missing values are confined to two columns: `V1` (18) and `V2` (18) — 36 cells in total
- Target: `1` = failure, `0` = no failure — **heavily imbalanced**. The balanced class weights
  computed in cell 20, `{0: 0.529, 1: 9.009}`, imply failures are roughly **5.5%** of the
  training split

## Approach

1. **EDA** — distributions of five representative predictors (`V1`–`V5`); correlation of every
   predictor with the target. The strongest are `V18` (−0.293), `V21` (+0.256), `V15` (+0.249),
   `V7` (+0.237) and `V16` (+0.231).
2. **Preprocessing** — median imputation for the 36 missing values; `StandardScaler` fitted on the
   training split only and applied to validation and test (essential, since neural nets are
   sensitive to input range).
3. **Class weighting** — `compute_class_weight('balanced')` yields `{0: 0.529, 1: 9.009}`. This was
   evaluated as an ablation arm (Models 5 and 6), *not* used in the selected model.
4. **Custom neural network** — a `SimpleNeuralNetwork` class written from first principles: He
   initialisation, ReLU hidden layers, sigmoid output, binary cross-entropy (optionally
   class-weighted), hand-derived backpropagation, gradient-descent and **Adam** update rules,
   per-layer **inverted dropout**, and a fixed 0.5 decision threshold.
5. **Ablation study** — **seven** configurations (Models 0–6) varying width, depth, dropout,
   optimiser, learning rate, epoch count and class weighting.
6. **Selection** — highest validation F1, then the winning configuration retrained on the combined
   train + validation set (20,000 rows) and evaluated **once** on the held-out test set.

## Model Ablation

All seven configurations, with the two hyperparameters that also vary between rows shown explicitly:

| Model | Architecture | Optimiser | Dropout | Class weights | Learning rate | Epochs |
|:-----:|--------------|-----------|:-------:|:-------------:|:-------------:|:------:|
| 0 | `[64]` | SGD | — | No | 0.01 | 40 |
| 1 | `[64, 32]` | SGD | — | No | 0.01 | 40 |
| 2 | `[64, 32]` | SGD | 0.3 | No | 0.01 | 40 |
| 3 | `[64, 32]` | Adam | — | No | 0.005 | 40 |
| **4** | **`[128, 64]`** | **Adam** | **0.3** | **No** | **0.005** | **50** |
| 5 | `[64, 32]` | SGD | — | Yes | 0.01 | 40 |
| 6 | `[128, 64]` | Adam | 0.3 | Yes | 0.005 | 50 |

**Selected: Model 4** — highest validation F1 (cell 42: *"Selected best model based on validation
F1-score: Model 4 (Adam, [128,64], dropout 0.3)"*). Note that the selected configuration uses **no
class weights**; cell 35 passes `class_weights=None`, and cell 44 consequently retrains the final
model unweighted. Both class-weighted arms, Models 5 and 6, lost the selection.

## Results — final model on the held-out test set

| Metric | Score |
|--------|:-----:|
| Accuracy | **0.9816** |
| Precision | **0.9439** |
| Recall | **0.7163** |
| **F1-score** | **0.8145** |

**Reading these numbers honestly:** precision of 0.94 means almost every predicted failure is real —
very few wasted inspections. Recall of 0.72 means roughly **28% of true failures are still missed**,
and each miss carries the highest cost in the table above. The accuracy figure of 98.2% is inflated
by class imbalance — at roughly 5-6% positives, a model that never predicts failure would already
score about 94% — and should not be quoted on its own. Closing the recall gap is the single
highest-value next step: lowering the decision threshold below the hard-coded 0.5 (trading cheap
false positives for expensive false negatives) and gathering more failure-event data.

## What the Ablation Does and Does Not Establish

The notebook computes training and validation F1, precision and recall for all seven models, but the
comparison table is never rendered (see Known Limitations), so **no per-model metric appears
anywhere in the executed notebook or the HTML report**. The only comparative result on record is the
name of the winner. That supports exactly two statements:

- **The selected configuration** is the wider two-layer network with dropout trained by Adam
  (`[128, 64]`, dropout 0.3, lr 0.005, 50 epochs).
- **Class weighting did not win.** It was applied only in Models 5 and 6, and neither beat the
  unweighted Model 4 on validation F1 — consistent with the final model's high precision (0.94) and
  weaker recall (0.72).

The remaining design choices are stated here as *rationale*, not as measured contributions:

- **Depth** — a second hidden layer is intended to capture nonlinear sensor interactions the
  single-layer baseline cannot.
- **Dropout** — regularisation against overfitting by randomly deactivating units during training.
- **Adam over SGD** — adaptive per-parameter step sizes to converge faster within a small update
  budget.

Comparing rows, `0 → 1` (depth), `1 → 2` (dropout), `1 → 5` and `4 ↔ 6` (class weights) each vary a
single factor. Two comparisons do not: `1 → 3` switches optimiser *and* halves the learning rate,
and `3 → 4` changes layer width, dropout and epoch count at once.

## Recommendations

1. **Deploy into the maintenance workflow** so predicted failures trigger preventive scheduling.
2. **Monitor false negatives closely** and tune the decision threshold toward recall — a false
   positive costs an inspection, a false negative costs a generator. The 0.5 threshold used here was
   never tuned.
3. **Collect more data around failure events** to strengthen the minority class.
4. **Retrain regularly** as turbines age and operating conditions shift.
5. **Investigate the influential features** (`V18`, `V21`, `V15`, `V7`, `V16`) with engineering
   teams — they may point at physical failure mechanisms worth designing out.
6. **Attach real costs** to repair, replacement and inspection so the operating point can be chosen
   by expected cost rather than by F1.

## Known Limitations

These are properties of the notebook as executed. They are documented, not fixed.

- **The model comparison table never renders.** In cell 42, `results_df_sorted` is assigned and then
  followed by further statements, so it is not the cell's final expression and is never displayed;
  cells 26–39 print only `Model N completed`. Every per-model F1, precision and recall is computed
  and discarded, so no claim about how much any single design change contributed can be supported.
- **The baseline is degenerate.** Model 0 predicts no positives at all on either split — scikit-learn
  raises `UndefinedMetricWarning: Precision is ill-defined ... no predicted samples` twice in cell 26
  — so its precision, recall and F1 are 0. "Improvement over baseline" here means improvement over a
  model that never fires.
- **The `sgd` optimiser is full-batch gradient descent.** `fit()` runs one forward/backward pass over
  the entire 16,000-row matrix per epoch with no mini-batching, so 40 epochs is 40 parameter updates.
  This is the likely reason the baseline never escapes the majority class, and it makes the
  SGD-versus-Adam comparison a contest between two update rules under an unusually small update
  budget rather than a general statement about the optimisers.
- **Two ablation comparisons are confounded.** Learning rate drops from 0.01 to 0.005 at the same
  step the optimiser switches to Adam, and the final configuration changes width, dropout and epoch
  count simultaneously. Those two transitions show which *combinations* did better, not the
  independent effect of any one factor.
- **Single seed, single run per configuration.** Every model is constructed with `random_state=42`
  and trained once, so the gaps between configurations are not separated from run-to-run variance.
- **The decision threshold is never tuned.** `predict()` hard-codes `> 0.5` despite the cost argument
  that explicitly favours recall. No threshold sweep, ROC curve or precision-recall curve is
  produced, so the reported recall is the recall at an untuned default rather than at a cost-optimal
  operating point.
- **No confusion matrix or cost model.** The project is motivated by a repair-versus-replace cost
  asymmetry, but no confusion matrix is displayed and no monetary cost is ever computed, so
  "minimising total maintenance cost" is argued rather than measured.
- **EDA is partial.** Only `V1`–`V5` are plotted individually (cell 16, labelled in the code as "a
  few representative features"); the other 35 predictors are never plotted on their own. In the same
  cell, `summary_stats = train_df.describe().T` is computed but not the final expression, so no
  per-feature summary table is displayed either.
- **The notebook's own closing prose contradicts its code.** Cell 46 states that the best model used
  class weights and names `V1`, `V2`, `V7`, `V13`, `V22` as the strongest-correlated features; cells
  35/42 and cell 18 respectively contradict both. This README follows the executed output. The
  notebook markdown has not been corrected.
- **TensorFlow is installed but never used.** Cell 8 pins and installs `tensorflow==2.18.0`; it is
  never imported. The network is pure NumPy, and the install is leftover scaffolding.

## Tech Stack

`Python` · `NumPy` (neural network from scratch) · `scikit-learn` (preprocessing & metrics) · `pandas` · `Matplotlib` · `Seaborn`

## Files

| File | Description |
|------|-------------|
| [`notebook.ipynb`](notebook.ipynb) | Full executed analysis, NN implementation and ablation |
| [`report.html`](report.html) | Rendered HTML report — **[view live](https://kerolous-samir.github.io/ai-ml-portfolio/04-renewind-predictive-maintenance/report.html)** |
