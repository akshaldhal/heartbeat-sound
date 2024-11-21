# Heartbeat Sound Classification

This project is a machine learning approach to classify heartbeat sounds into four categories:

- **Normal** (`0`)
- **Extrahls** (`1`)
- **Murmur** (`2`)
- **Extrastole** (`4`)

The model utilizes the [Dangerous Heartbeat Dataset (DHD)](https://www.kaggle.com/datasets/mersico/dangerous-heartbeat-dataset-dhd) from Kaggle to learn and predict heartbeat sound types. Initial experiments and model iterations are detailed below.

---

## Project Overview

### Objective

The goal of this project is to develop an efficient, lightweight machine learning model capable of classifying heartbeat sounds into distinct categories. This model, with its compact size, is intended to be deployable on mobile devices such as smartphones.

### Initial Approach

1. **Baseline Model**:
    - **Feature Extraction**: Used audio frequency as the primary feature.
    - **Classifier**: Basic neural network model.
    - **Result**: Accuracy capped at around 30%.

2. **Gradient Boosting**:
    - **Algorithm**: Implemented using "PerpetualBoosters" for gradient boosting.
    - **Result**: No significant improvement over the baseline.

### Enhanced Approach: Custom Preprocessing

To improve model performance, custom preprocessing techniques were developed, as follows:

1. **Preprocessor 1** (`preprocessor1.py`):
    - **Method**: Divided audio into frames based on delta time and extracted frequency-amplitude pairs from each frame using `librosa`.
    - **Padding**: End of data was padded to ensure uniform length across samples.
    - **Datasets**: Generated three datasets—mini, small, and main.
    - **Result**: Achieved ~72% accuracy with the main dataset. However, model size was still large (~770 MB).

2. **Preprocessor 2** (`preprocessor2.py`):
    - **Method**: Employed alignment padding, using Euclidean distance to align smaller samples with larger ones for minimal distance across padding. This ensures heartbeat samples are consistently aligned in the padded arrays.
    - **Datasets**: Due to processing time, only mini and small datasets were generated.
    - **Result**: Achieved ~73% accuracy using the small dataset, with a drastically reduced model size of ~6 MB—over 99% smaller than previous models, making it feasible for mobile deployment.

### Summary of Results

The current model achieves an accuracy of ~73% on the small dataset, with a compact model size that supports local execution on mobile devices (model size of ~6MB).

---

## Files

- **Data Preprocessing**
  - `preprocessor1.py`: Initial preprocessing script that segments audio and pads data to uniform length.
  - `preprocessor2.py`: Advanced preprocessing script for alignment padding based on Euclidean distance.

- **Model Training**
  - `train.py`: Script used to train the model on the processed datasets.

- **Model Inference**
  - `run_model.py`: Single script to load the trained model and run inference on a given audio file, uses `alignment_reference.pkl`, `amp_scaler.pkl`, `freq_scaler.pkl`, and `final_model_fcnn_classifier_16_8_7368.pth`.

---

## Getting Started

To train the model, execute the following steps:

1. Run `preprocessor1.py` or `preprocessor2.py` to prepare the dataset.
2. Use `train.py` to train the model on the processed dataset.
3. Run `run_model.py` to load the trained model and perform inference on a given audio file.

---

## Potential Future Improvements

- **Additional Feature Engineering**: Exploring features beyond frequency and amplitude, such as Mel-frequency cepstral coefficients (MFCCs) or spectral contrast.
- **Model Optimization**: Experimenting with quantization and pruning techniques to further reduce model size without sacrificing accuracy.
- **Other approachs**: Using some other approach like CNN, RNN, etc. instead of a fully connected neural network.

---

## Dataset

Dataset used: [Dangerous Heartbeat Dataset (DHD)](https://www.kaggle.com/datasets/mersico/dangerous-heartbeat-dataset-dhd).

---
