# Notebooks

Run the analysis notebooks in the following order:

1. `01_data_exploration.ipynb` — explore participant metadata, condition distributions, task structure, and representative raw accelerometer/gyroscope signals.
2. `02_feature_extraction.ipynb` — convert raw smartwatch recordings into 96 statistical and frequency-domain features per recording.
3. `03_model_training.ipynb` — develop Logistic Regression, Support Vector Machine, and Random Forest models while preserving participant identity during train/test splitting.
4. `04_model_evaluation.ipynb` — perform the final 5-fold participant-grouped evaluation with nested threshold selection, task-specific analysis, and Random Forest feature-importance analysis.

The raw PADS dataset is expected locally under `data/PADS/` and is not committed to this repository.
