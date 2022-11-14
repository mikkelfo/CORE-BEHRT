import pandas as pd
import numpy as np
import torch
from tokenizer import EHRTokenizer
from dataset.EHR import EHRDataset
from utils.args import setup_preprocess


def tokenize_data(args):
    df = pd.read_csv(args.data_file)
    key, code, date = args.identifiers

    # Remove NaN values and sort chronologically
    chrono_df = df[~df[date].isna()].sort_values(date)

    all_codes = []
    for _, patient in chrono_df.groupby(key):
        patient_codes = []
        prev_date = None
        for row in patient.itertuples():
            # Create new segment if: No other segments are present OR the date has changed
            if prev_date is None or prev_date != getattr(row, date):
                patient_codes.append([])            # Create new segment
            prev_date = getattr(row, date)                   # Set new previous date
            patient_codes[-1].append(getattr(row, code))     # Append to current segment
        all_codes.append(patient_codes)

    # Tokenize
    tokenizer = EHRTokenizer(args.vocabulary)
    outputs = tokenizer(all_codes)

    # Save to file
    tokenizer.save_vocab(args.vocabulary_file)
    with open(args.tokenized_file, 'wb') as f:
        torch.save(outputs, f)


def split_data(args):
    with open(args.tokenized_file, 'rb') as f:
        inputs = torch.load(f)

    N = len(inputs.input_ids)
    np.random.seed(0)

    indices = np.random.permutation(N)

    N_test = int(N*args.test_ratio)
    test_indices = indices[:N_test]
    train_indices = indices[N_test:]

    test_codes = [inputs.input_ids[ind] for ind in test_indices]
    train_codes = [inputs.input_ids[ind] for ind in train_indices]

    test_segments = [inputs.visit_segments[ind] for ind in test_indices]
    train_segments = [inputs.visit_segments[ind] for ind in train_indices]

    test_mask = [inputs.attention_mask[ind] for ind in test_indices]
    train_mask = [inputs.attention_mask[ind] for ind in train_indices]

    train_set = EHRDataset(train_codes, train_segments, train_mask)
    test_set = EHRDataset(test_codes, test_segments, test_mask)

    with open(args.train_file, 'wb') as f:
        torch.save(train_set, f)
    with open(args.test_file, 'wb') as f:
        torch.save(test_set, f)

    return train_set, test_set


if __name__ == '__main__':
    args = setup_preprocess()
    tokenize_data(args)         # Should only be done once
    split_data(args)            # Should only be done once

