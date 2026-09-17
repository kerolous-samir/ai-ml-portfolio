# ReneWind — Wind Turbine Failure Prediction

> A neural network **implemented from scratch in NumPy** — forward pass, backpropagation, SGD and
> Adam optimisers, dropout and class weighting — applied to imbalanced sensor data to predict
> generator failures before they happen.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch%20NN-013243?logo=numpy&logoColor=white)](https://numpy.org/)
![Type](https://img.shields.io/badge/Type-Imbalanced%20Classification-blue)
![F1](https://img.shields.io/badge/Test%20F1-0.8145-brightgreen)

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
selected on **F1**, not accuracy — accuracy is misleading on data this imbalanced.

## Dataset

- **40 ciphered predictors** (`V1`–`V40`)
- **20,000 training** observations, **5,000 test** observations
- Target: `1` = failure, `0` = no failure — **heavily imbalanced** toward no-failure

## Approach

1. **EDA** — distribution of each predictor; `V1`, `V2`, `V7`, `V13` and `V22` show the strongest correlation with failure.
2. **Preprocessing** — median imputation for the small number of missing values; `StandardScaler` (essential, since neural nets are sensitive to input range).
3. **Class weighting** — `compute_class_weight('balanced')` to penalise misclassifying the minority failure class.
4. **Custom neural network** — a `SimpleNeuralNetwork` class written from first principles, supporting configurable hidden layers, learning rate, epochs, **SGD and Adam** optimisers, per-layer **dropout**, and class weights.
5. **Ablation study** — five architectures trained to isolate the contribution of depth, dropout, optimiser choice and class weighting.
6. **Selection** — best validation F1, then a single final evaluation on the held-out test set.

## Model Ablation

| Model | Architecture | Optimiser | Dropout | Class weights |
|:-----:|--------------|-----------|:-------:|:-------------:|
| 0 | `[64]` | SGD | — | No |
| 1 | `[64, 32]` | SGD | — | No |
| 2 | `[64, 32]` | SGD | 0.3 | No |
| 3 | `[64, 32]` | Adam | — | No |
| **4** | **`[128, 64]`** | **Adam** | **0.3** | **Yes** |

**Selected: Model 4** — highest validation F1.

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
by class imbalance and should not be quoted on its own. Closing the recall gap is the single
highest-value next step — by lowering the decision threshold (trading cheap false positives for
expensive false negatives) and by gathering more failure-event data.

## What Each Change Contributed

- **Depth** — a second hidden layer improved the model's ability to capture nonlinear sensor interactions over the single-layer baseline.
- **Dropout** — reduced overfitting by randomly deactivating neurons during training.
- **Adam over SGD** — accelerated convergence and generally improved validation performance.
- **Class weights** — the largest single gain in **recall**, and therefore in F1.

## Recommendations

1. **Deploy into the maintenance workflow** so predicted failures trigger preventive scheduling.
2. **Monitor false negatives closely** and tune the decision threshold toward recall — a false positive costs an inspection, a false negative costs a generator.
3. **Collect more data around failure events** to strengthen the minority class.
4. **Retrain regularly** as turbines age and operating conditions shift.
5. **Investigate the influential features** (`V1`, `V2`, `V7`, `V13`, `V22`) with engineering teams — they may point at physical failure mechanisms worth designing out.

## Tech Stack

`Python` · `NumPy` (neural network from scratch) · `scikit-learn` (preprocessing & metrics) · `pandas` · `Matplotlib` · `Seaborn`

## Files

| File | Description |
|------|-------------|
| [`notebook.ipynb`](notebook.ipynb) | Full executed analysis, NN implementation and ablation |
| [`report.html`](report.html) | Rendered HTML report — **[view live](https://kerolous-samir.github.io/ai-ml-portfolio/04-renewind-predictive-maintenance/report.html)** |
