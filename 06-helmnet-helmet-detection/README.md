# HelmNet — Safety Helmet Detection

> Image classification for workplace safety compliance: does the worker in this frame have a
> helmet on? A custom CNN benchmarked against **VGG16 transfer learning** variants.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
![Type](https://img.shields.io/badge/Type-Computer%20Vision-blue)

## Business Context

Workplace safety in hazardous environments — construction sites, industrial plants — depends on
consistent helmet use to prevent head injuries from falling objects and machinery. Manual
monitoring is error-prone and does not scale across large operations.

**SafeGuard Corp** wants an automated image-analysis system that detects helmet compliance from
camera feeds, improving enforcement accuracy and scaling oversight without adding headcount.

## Objective

Classify images into two categories:

- **With Helmet** — worker is wearing a safety helmet
- **Without Helmet** — worker is not

## Dataset

| Property | Value |
|----------|-------|
| Source | `images_proj.npy` + `Labels_proj.csv` (loaded from Google Drive in Colab) |
| Total images | **631**, loaded shape `(631, 200, 200, 3)` |
| Labels | Binary `0` / `1` — **320 label-0** / **311 label-1**, essentially balanced |
| Preprocessing | RGB → grayscale via `cv2.COLOR_RGB2GRAY` → `(631, 200, 200, 1)`, scaled to `[0, 1]` as `float32` |
| Train / Validation / Test | **441 / 95 / 95** — stratified, `test_size=0.15`, then `0.15/0.85` ≈ `0.176` of the remainder so validation is also 15% of the original 631, `random_state=42` |
| Seeding | `tf.keras.utils.set_random_seed(812)` |

The notebook's problem statement describes 311 *with helmet* and 320 *without helmet* images, which
matches the label counts (`0` → 320, `1` → 311). No cell in the notebook asserts that mapping
programmatically, so the class names used below are inferred from those counts, not verified in code.

## Approach

1. **EDA** — five random images sampled per class with their labels; class balance checked with a `value_counts` bar plot.
2. **Preprocessing** — grayscale conversion (the notebook's stated reason is reduced computational cost while retaining spatial structure; no colour-vs-grayscale ablation was run), stratified three-way split, normalisation to `[0, 1]`.
3. **Model 1 — CNN from scratch** (cell 34): three `Conv2D` → `BatchNormalization` → `MaxPooling2D` blocks (32/64/128 filters) → `Flatten` → `Dense(128)` → `Dropout(0.5)` → `Dense(1, sigmoid)`. Adam at `1e-4`, trained through an augmentation generator (rotation 15°, width/height shift 0.1, shear 0.1, zoom 0.1, horizontal flip) for up to 30 epochs with early stopping on `val_loss` (patience 5, `restore_best_weights=True`).
4. **Model 2 — VGG16 base** (cell 38): grayscale replicated across three channels, ImageNet-pretrained VGG16 convolutional base frozen as a fixed feature extractor, with `Flatten` → `Dense(128)` → `Dropout(0.5)` → `Dense(1, sigmoid)`. Adam `1e-4`, 15 epochs, batch 32.
5. **Model 3 — VGG16 base + FFNN head** (cell 42): the same frozen base with a wider classifier — `Dense(256)` → `Dropout(0.5)` → `Dense(128)` → `Dropout(0.5)` → `Dense(1, sigmoid)` — testing whether extra head capacity helps. 15 epochs.
6. **Model 4 — VGG16 base + FFNN + augmentation** (cell 47): the deeper head trained through an augmentation generator (rotation 20°, shifts 0.1, shear 0.1, zoom 0.1, horizontal flip, `fill_mode='nearest'`) for 20 epochs.
7. **Evaluation** — per-model validation classification report and confusion matrix, then a final test-set report for the selected model.

So the benchmark is **one from-scratch CNN against three VGG16 configurations**, not four independent architectures.

## Results

| Model | Validation accuracy | Test accuracy |
|-------|--------------------:|--------------:|
| Model 1 — CNN from scratch (augmented) | 0.5053 | 0.5053 |
| Model 2 — VGG16 base (frozen) | **1.0000** | **1.0000** |
| Model 3 — VGG16 base + FFNN | **1.0000** | **1.0000** |
| Model 4 — VGG16 base + FFNN + augmentation | **1.0000** | **1.0000** |

*Reconstructed from the printed output of cells 34, 38, 42 and 47 — the notebook assembles this
table in cell 51 but never renders it (see Known Limitations).*

**Selected model: VGG16 base (frozen feature extractor).** Cell 51 picks it with
`np.argmax` over validation accuracy; because Models 2, 3 and 4 all tie at `1.0000`, `argmax`
returns the first of the tie. The selection is a tie-break, not a win.

Final test-set report for the selected model (cell 53; labels as the notebook prints them —
`0.0` ≈ without helmet, `1.0` ≈ with helmet):

```
              precision    recall  f1-score   support

         0.0     1.0000    1.0000    1.0000        48
         1.0     1.0000    1.0000    1.0000        47

    accuracy                         1.0000        95
```

The VGG16 base reached **100% validation accuracy in its first epoch** and validation loss fell
from `0.0428` to `8.0e-04` over 15 epochs.

> **Read this number carefully.** 95/95 on a 95-image test set drawn from 631 images is not
> evidence of generalisation. Three architecturally different models — a compact head, a deeper
> head, and a deeper head under augmentation — all score exactly 1.0000 on both validation and
> test, and the base model is already perfect after one epoch. That pattern is the signature of
> **near-duplicate images leaking across the split**, and no de-duplication or leakage audit was
> run before the stratified split in cell 25. Until that check is done, treat `1.0000` as
> unvalidated.

## Model Comparison Insights

- **The three VGG16 variants are indistinguishable here.** All three score 1.0000 on validation and
  1.0000 on test, so the experiment cannot rank them. Extra head capacity (Model 3) and augmentation
  (Model 4) neither helped nor hurt — there was no measurable headroom left to move.
- **Model 1 did not merely underperform; it failed.** Its 0.5053 is exactly the majority-class rate
  (48 of 95) and its recall on class 1 is 0.0000 — it emits a single class for every input. This is
  a broken baseline, not a demonstration that from-scratch CNNs overfit small datasets.
- **The comparison therefore has one real finding**: pretrained VGG16 features separate these two
  classes trivially on this data. Whether that reflects the task or the split is unresolved.
- Among the tied models, the base configuration carries the smallest classification head
  (`Dense(128)` vs `Dense(256)` → `Dense(128)`), making it the cheapest at inference. That is a
  compute argument for selecting it, not an accuracy one.

## Known Limitations

1. **Probable train/test leakage; the headline metric is not trustworthy.** Three different
   architectures each score 95/95 on test and 95/95 on validation, and Model 2 is at 1.0000
   validation accuracy after a single epoch. No duplicate or near-duplicate detection (perceptual
   hashing, embedding-distance check) was run before the stratified split in cell 25, so
   near-identical frames may sit on both sides of it. The perfect score should not be read as
   generalisation.
2. **Model 1 is a degenerate single-class predictor and the notebook never diagnoses it.** Training
   accuracy climbs to ~0.99 while validation accuracy sits flat at 0.5053 across all six epochs and
   validation loss rises monotonically (1.20 → 8.28). Early stopping with `restore_best_weights=True`
   therefore restored the epoch-1 weights. The class-1 recall of 0.0000 is stated in the output and
   left uninvestigated; untested candidates include the BatchNorm train/inference statistics
   mismatch on a 441-image training set and the short effective training budget.
3. **The model comparison table is never displayed.** Cell 51 builds `performance_df`, but the
   bare `performance_df` expression is followed by further statements, so Jupyter renders nothing —
   only the `print` of the best model name appears. The comparison table above was reassembled by
   hand from the individual model cells.
4. **No model artifact and no inference path.** The notebook contains no `model.save()` call, no
   persisted preprocessing. Single images pass through `predict` only inside the visualisation cells
   (36, 40, 44, 49), and `rgb_to_gray` / `to_rgb` exist only as in-notebook helpers, so there is no
   reusable, exportable inference path. Nothing from this run
   can be loaded or served; the deployment recommendations below are direction, not a shipped system.
5. **VGG16 inputs do not match the pretrained weights' expected distribution.** Images are
   converted to grayscale (cell 23) and then replicated across three channels (cell 38), so a
   colour-pretrained backbone is fed a colour-free signal. Inputs are scaled to `[0, 1]` rather than
   passed through `keras.applications.vgg16.preprocess_input`. Both choices are workable but neither
   is the intended input pipeline for these weights.
6. **The evaluation utilities are dead code.** `model_performance_classification` and
   `plot_confusion_matrix` (cells 31–32) are defined and never called, and both call
   `target.to_numpy()`, which would raise on the NumPy label arrays this notebook actually uses.
7. **Single split, single seed, and the test set was visible during development.** Every number
   comes from one stratified split (`random_state=42`) with no cross-validation or repeated runs, so
   each metric is a single point estimate on 95 images with no interval attached. Test accuracy was
   also printed inside each model cell before the selection step in cell 51, so the test set was
   observed throughout model development rather than held back for a single final measurement.
8. **Class-name mapping is inferred, not verified.** The notebook prints and plots labels only as
   `0` and `1`; the "with helmet" / "without helmet" naming rests on the count match with the
   problem statement. A mislabelled mapping would invert every per-class metric reported here.

## Deployment Recommendations

1. **Resolve the leakage question first.** De-duplicate the image set and re-split by source
   (camera, site or session) rather than at random, then re-measure. No deployment decision should
   rest on the current 1.0000.
2. **Fix or retire Model 1** so the benchmark compares four working models instead of three
   saturated ones and one that predicts a constant.
3. **Persist the selected model and its preprocessing** so there is a loadable artifact and a
   single-image inference path to test against.
4. **Validate against real site imagery** — varied lighting, angle, occlusion, helmet colour and
   distance — before trusting the model in the field.
5. **Fine-tune the upper VGG16 layers** at a low learning rate (~`1e-5`) to adapt to site-specific
   visual conditions, once a clean split is in place.
6. **For edge devices**, consider MobileNetV3 or EfficientNet-Lite for a lighter inference footprint.
7. **Pilot at a single site** before system-wide rollout, log real-world false positives and
   negatives for retraining, and build a compliance analytics dashboard alongside.

## Tech Stack

`Python` · `TensorFlow` / `Keras` · `OpenCV` · `scikit-learn` · `NumPy` · `pandas` · `Matplotlib` · `Seaborn` · `Google Colab`

## Files

| File | Description |
|------|-------------|
| [`notebook.ipynb`](notebook.ipynb) | Full executed pipeline — one from-scratch CNN plus three VGG16 variants |
| [`report.html`](report.html) | Rendered HTML report — **[view live](https://kerolous-samir.github.io/ai-ml-portfolio/06-helmnet-helmet-detection/report.html)** |
