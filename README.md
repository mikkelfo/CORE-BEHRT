# CORE-BEHRT: A Carefully Optimized and Rigorously Evaluated BEHRT

## Overview
CORE-BEHRT aims to provide a robust framework for reproducing state-of-the-art results in EHR data analysis using a BEHRT-like model. Follow the steps below to reproduce the results.

## Prerequisites
Before you begin, ensure you have the necessary dependencies installed. This project may require:
- PyTorch
- transformers
- Numpy
- Pandas
- scikit_learn
- tqdm
- matplotlib
- (pyarrow if parquet files are used)

## Getting Started

### Data Preparation
To correctly prepare your data for processing, execute the scripts in the following order. Ensure your data adheres to the specified format before starting:

1. **Data Format**
   - **Required Data Format:**
     - **Patient Data**: Ensure the file `patients_info.csv` contains columns for `PID`, `DATE_OF_BIRTH` and other relevant background features (such as `RACE` or `GENDER`).
     - **Event Data**: The files `concept.{code_type}.csv` should include `TIMESTAMP`, `PID`, `ADMISSION_ID`, and `CONCEPT`.
   - Use the preprocessing tools available at [ehr_preprocess](https://github.com/kirilklein/ehr_preprocess.git) to convert your raw data into the required format.

2. **Feature Creation and Tokenization**
   - `main_create_data`: Stores features as dictionaries with list of lists as values and difference concept data streams as keys (concept, segment, age, abspos,...) holding the patient sequences. Tokenizes the features. Use data_pretrain.yaml config.

3. **Model Pre-training**
   - `main_pretrain`: Pre-trains a standard a BEHRT model on the tokenized features.

3. **Data Preparation for Fine-tuning**
   - `main_create_outcomes`: From the formatted data, creates a dictionary with the events of interest (abspos of first time occurrence). Example dictionary: {'PID':['p1', 'p2', ...], 'EVENT1':[5423, None, ...], ...}

4. **Model Fine-tuning**
   - `main_finetune_cv`: Performs 5-fold cross-validation + evaluation on a holdout-set.

### Evaluation and Visualization
To evaluate the model performance:
- `main_finetune_cv_get_stats`: Get basic statistics on patient sequences in train/val and test set.
- `main_finetune_cv_test`: Test models trained in CV on a dataset of your choice.
- `main_encode_censored_patients`: Encodes sequences using a trained model, specifically censored on a predetermined event.
- `main_encode_concepts`: Save encodings of concepts.

