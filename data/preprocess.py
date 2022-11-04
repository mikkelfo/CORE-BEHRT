import pandas as pd
import numpy as np
import torch
from tokenizer import EHRTokenizer


def pickle_data(file='data/20210526/Aktive_problemliste_diagnoser.csv', identifier="Key", date_column="NOTED_DATE"):
    df = pd.read_csv(file)

    # Remove NaN values and sort chronologically
    chrono_df = df[~df[date_column].isna()].sort_values(date_column)

    all_codes = []
    for _, patient in chrono_df.groupby(identifier):
        patient_codes = []
        prev_date = None
        for row in patient.itertuples():
            # Create new segment if: No other segments are present OR the date has changed
            if prev_date is None or prev_date != row.NOTED_DATE:
                patient_codes.append([])            # Create new segment
            prev_date = row.NOTED_DATE              # Set new previous date
            patient_codes[-1].append(row.Code)      # Append to current segment
        all_codes.append(patient_codes)

    # Tokenize
    tokenizer = EHRTokenizer()
    outputs = tokenizer(all_codes)

    # Save to file
    tokenizer.save_vocab('vocabulary.pt')
    with open('tokenized_output.pt', 'wb') as f:
        torch.save(outputs, f)


def split_testsets(inputs="tokenized_output.pt", test_ratio=0.2):
    with open(inputs, 'rb') as f:
        inputs = torch.load(f)

    N = len(inputs.input_ids)
    np.random.seed(0)

    indices = np.random.permutation(N)

    N_test = int(N*test_ratio)
    test_indices = indices[:N_test]
    train_indices = indices[N_test:]

    test_codes = [inputs.input_ids[ind] for ind in test_indices]
    train_codes = [inputs.input_ids[ind] for ind in train_indices]

    test_segments = [inputs.visit_segments[ind] for ind in test_indices]
    train_segments = [inputs.visit_segments[ind] for ind in train_indices]

    test_mask = [inputs.attention_mask[ind] for ind in test_indices]
    train_mask = [inputs.attention_mask[ind] for ind in train_indices]

    with open('features.train', 'wb') as f:
        torch.save((train_codes, train_segments, train_mask), f)
    with open('features.test', 'wb') as f:
        torch.save((test_codes, test_segments, test_mask), f)

    return (train_codes, train_segments, train_mask), (test_codes, test_segments, test_mask)


if __name__ == '__main__':
    pickle_data()       # Should only be done once
    split_testsets()    # Should only be done once

