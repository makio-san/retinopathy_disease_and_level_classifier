# Diabetic retinopathy classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/makio-san/retinopathy_disease_and_level_classifier/blob/main/cnn_retinopathy_classifier.ipynb)

A deep learning pipeline for diabetic retinopathy grading on Gaussian-filtered retinal images. EfficientNetB3 (ImageNet weights) is the backbone, trained in two phases (new head first, then a gentle fine-tune) with a class-weighted loss and on-GPU augmentation. Everything runs in one Colab notebook: [`cnn_retinopathy_classifier.ipynb`](cnn_retinopathy_classifier.ipynb).

## What it predicts

Five grades: `0 No DR`, `1 Mild`, `2 Moderate`, `3 Severe`, `4 Proliferative DR`. The notebook also derives two binary readouts from the same softmax output, **any DR** (grade 1 or higher) and **referable DR** (grade 2 or higher), and reports AUC, sensitivity and specificity for both.

## Data

[`sovitrath/diabetic-retinopathy-224x224-gaussian-filtered`](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered) on Kaggle: the APTOS 2019 training set (3,662 fundus images) resized to 224x224 and Gaussian filtered, CC0. It is split 80/10/10 into train / validation / test, stratified by grade with seed 42 (2,965 / 330 / 367 images).

## Run it on Colab

1. Open the notebook with the badge above and choose `Runtime -> Change runtime type -> GPU`.
2. Create a Kaggle API token at <https://www.kaggle.com/settings/api> ("Generate New Token") and add it as a Colab secret named `KAGGLE_API_TOKEN` (key icon in the left sidebar, *Notebook access* switched on). Uploading `kaggle.json` or `access_token` to `/content` works as well.
3. `Runtime -> Run all`. The trained model is saved as `dr_efficientnetb3.keras` in Colab's file panel; download it before the runtime ends.

Colab already ships TensorFlow, scikit-learn, pandas and matplotlib, so the notebook only installs the `kaggle` CLI.

## Results

| version | test accuracy | test loss | quadratic weighted kappa |
|---|---|---|---|
| v1: original notebook | 0.8038 | 0.6482 | not measured |
| v2: this notebook | _not run yet_ | _not run yet_ | _not run yet_ |

v1 stopped early at epoch 17 with roughly 92% training against 80% validation accuracy, so it was overfitting.

## What changed in v2

- **Pixel range.** Keras' EfficientNet rescales and normalizes inside the model and expects 0-255 input. v1 also divided by 255 in `ImageDataGenerator`, so the pixels were scaled down twice. v2 feeds raw 0-255 floats and checks the range on a sample batch.
- **Two-phase training.** The new head is trained first with the backbone frozen; then the backbone is unfrozen at a low learning rate with BatchNorm kept frozen. v1 trained everything from the first step, and its validation accuracy was 0.05 after epoch 1.
- **Class weights.** About half of the images are "No DR", so the loss uses the square root of the balanced class weights.
- **Metrics that fit the task.** Quadratic weighted kappa, a per-class report, a confusion matrix, and AUC / sensitivity / specificity for any DR and referable DR.
- **Input pipeline.** `tf.data` plus augmentation layers inside the model replace the CPU-bound `ImageDataGenerator`.

## Roadmap

- [ ] Run v2 on Colab and fill in the table above
- [ ] Ordinal regression head with thresholds tuned on the validation split
- [ ] Grad-CAM overlays to check what the model looks at
- [ ] Inference script and a small demo
