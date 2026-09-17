# HelmNet — Safety Helmet Detection

> Image classification for workplace safety compliance: does the worker in this frame have a
> helmet on? Custom CNNs benchmarked against **VGG16 transfer learning**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
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
| Total images | **631** |
| Original resolution | 200 × 200 × 3 (RGB) |
| Class balance | **320 without helmet** / **311 with helmet** — essentially balanced |
| Train / Validation / Test | **441 / 95 / 95** |
| Preprocessing | Converted to grayscale (200 × 200 × 1), normalised to `[0, 1]` |

## Approach

1. **EDA** — random image sampling per class with labels; verified class balance.
2. **Preprocessing** — grayscale conversion (helmet detection is shape-driven, not colour-driven), stratified splitting, normalisation.
3. **Model 1 — baseline CNN** built from scratch: stacked Conv2D + MaxPooling blocks into dense layers.
4. **Model 2 — CNN with augmentation**: random flips, rotations and zooms to expand effective training data.
5. **Model 3 — feed-forward baseline** on flattened pixels, as a control to demonstrate the value of convolution.
6. **Model 4 — VGG16 transfer learning**: ImageNet-pretrained convolutional base as a frozen feature extractor with a custom classification head.
7. **Evaluation** — training/validation curves, confusion matrices and classification reports per model.

## Results

**Selected model: VGG16 transfer learning (base configuration).**

Final performance on the 95-image held-out test set:

```
              precision    recall  f1-score   support
Without Helmet   1.0000    1.0000    1.0000        48
   With Helmet   1.0000    1.0000    1.0000        47
      accuracy                       1.0000        95
```

The VGG16 base converged fast — **validation accuracy reached 100% within the first epoch** and
validation loss fell steadily to ~1e-3 over 15 epochs, indicating a clean, well-separated problem
for pretrained features.

> **Interpreting this result.** A perfect score on **95 test images** from a **631-image** dataset
> should be read as *"the pretrained features separate these two classes cleanly on this data"* —
> not as a guarantee of production accuracy. The dataset is small and likely more visually
> consistent than real CCTV footage (lighting, angle, occlusion, helmet colour, distance). Before
> trusting this in the field it needs validation against real site imagery. The deployment plan
> below is written accordingly.

## Model Comparison Insights

- **VGG16 transfer learning won** on accuracy, stability and generalisation — its pretrained convolutional filters extract strong visual features even from a small dataset.
- **Simple CNNs overfit quickly** given only 441 training images.
- **Heavier augmentation and the FFNN variant offered no meaningful improvement** — with transfer learning already saturating this dataset, added complexity bought nothing.
- The base VGG16 configuration gives the best **accuracy-to-compute trade-off**, which matters for real-time CCTV inference.

## Deployment Recommendations

1. **Deploy the VGG16 base model** for helmet compliance monitoring.
2. **Log real-world false positives and negatives** and feed them back into retraining — this is how the small-dataset risk above gets retired.
3. **Fine-tune the upper VGG16 layers** at a low learning rate (~`1e-5`) to adapt to site-specific visual conditions.
4. **For edge devices**, consider MobileNetV3 or EfficientNet-Lite for a lighter inference footprint.
5. **Pilot at a single site** before system-wide rollout; build a compliance analytics dashboard alongside.
6. **Retrain periodically** on new imagery covering varied lighting, angles and helmet colours.

## Tech Stack

`Python` · `TensorFlow` / `Keras` · `OpenCV` · `scikit-learn` · `NumPy` · `pandas` · `Matplotlib` · `Seaborn`

## Files

| File | Description |
|------|-------------|
| [`notebook.ipynb`](notebook.ipynb) | Full executed pipeline with all four model variants |
| [`report.html`](report.html) | Standalone HTML report |
