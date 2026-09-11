# Participant-Level Machine Learning Classification of Parkinson’s Disease Using Smartwatch Movement Data

CSCI 543 – Machine Learning, Fall 2026  
University of North Dakota  
Michelle Di Cicco

## Project Overview

This project applies supervised machine learning to the Parkinson’s Disease Smartwatch (PADS) dataset to distinguish participants with Parkinson’s disease from healthy controls using smartwatch accelerometer and gyroscope recordings.

The analysis focuses on participant-level generalization. All recordings from a participant are kept together during validation to prevent subject-level data leakage.

## Dataset

The project uses the public Parkinson’s Disease Smartwatch (PADS) dataset described in:

J. Varghese, A. Brenner, M. Fujarski, C. M. Van Alen, L. Plagwitz, and T. Warnecke, “Machine Learning in the Parkinson’s disease smartwatch (PADS) dataset,” *npj Parkinson’s Disease*, vol. 10, no. 1, p. 9, 2024. DOI: 10.1038/s41531-023-00625-7.

Dataset DOI: **10.13026/m0w9-zx22**

The primary binary-classification cohort contains **355 participants**: **276 with Parkinson’s disease** and **79 healthy controls**. Each participant completed **11 standardized motor tasks** using smartwatches on both wrists, yielding **7,810 recordings**.

Raw PADS data are not redistributed in this repository. See [`data/README.md`](data/README.md) for dataset access and expected file structure.

## Feature Engineering

Each recording contains three-axis accelerometer and three-axis gyroscope signals sampled at 100 Hz. Accelerometer and gyroscope vector magnitudes are also calculated, producing eight analyzed signals per recording.

Twelve features are extracted from each signal:

- mean
- standard deviation
- median
- minimum
- maximum
- range
- root mean square (RMS)
- interquartile range (IQR)
- skewness
- kurtosis
- dominant frequency
- spectral energy

This produces **96 numerical predictors per recording**.

## Models

Three supervised learning models are compared:

- Logistic Regression
- Support Vector Machine (RBF kernel)
- Random Forest

Class weighting is used to account for the unequal numbers of Parkinson’s disease and healthy-control participants.

## Validation Strategy

The primary evaluation uses **5-fold stratified participant-grouped cross-validation**. Decision thresholds are selected within each outer training fold using **3-fold stratified participant-grouped cross-validation** and Youden’s J statistic. Recording-level scores are averaged to participant-level scores before final evaluation.

Primary metrics include accuracy, balanced accuracy, precision, sensitivity, specificity, F1 score, and ROC-AUC.

## Main Results

| Model | Accuracy | Balanced Accuracy | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.789 ± 0.056 | 0.774 ± 0.032 | 0.794 ± 0.069 | 0.754 ± 0.038 | 0.851 ± 0.049 | 0.876 ± 0.029 |
| SVM | 0.775 ± 0.061 | **0.823 ± 0.022** | 0.740 ± 0.087 | **0.905 ± 0.056** | 0.834 ± 0.052 | **0.910 ± 0.024** |
| Random Forest | **0.800 ± 0.021** | 0.820 ± 0.022 | 0.786 ± 0.031 | 0.854 ± 0.042 | **0.858 ± 0.022** | **0.910 ± 0.020** |

Random Forest produced the highest overall accuracy and F1 score, while SVM produced the highest balanced accuracy and specificity. Both achieved a mean ROC-AUC of 0.910.

Task-specific analysis showed that no single motor task matched the performance of the combined 11-task model. Random Forest feature importance was distributed across many predictors, with dominant-frequency and IQR features among the strongest categories. Accelerometer and gyroscope features contributed almost equally to total importance.

## Repository Structure

```text
PADS-Parkinsons-ML/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── outputs/
│   ├── figures/
│   └── tables/
└── report/
    └── CS543_Project.pdf
```

## Notebook Workflow

`01_data_exploration.ipynb` inspects participant metadata, condition distributions, motor tasks, and representative raw smartwatch signals.

`02_feature_extraction.ipynb` converts raw accelerometer and gyroscope recordings into the 96-feature model-ready dataset.

`03_model_training.ipynb` develops and compares Logistic Regression, SVM, and Random Forest classifiers while preserving participant identity.

`04_model_evaluation.ipynb` performs the final nested participant-grouped evaluation, task-specific Random Forest analysis, and feature-importance analysis.

## Reproducibility

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

Download the PADS dataset separately using the source information in [`data/README.md`](data/README.md), place it in the expected local data directory, and run the notebooks in numerical order.

## Course Project

This repository was developed for CSCI 543 Machine Learning at the University of North Dakota. The project emphasizes real-world machine-learning application, model comparison, participant-independent evaluation, and model interpretability.
