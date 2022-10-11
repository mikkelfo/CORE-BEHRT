import pandas as pd
import numpy as np
import torch
from tokenizer import EHRTokenizer


def pickle_data(icd_file='data/20210526/Aktive_problemliste_diagnoser.csv'):
    df = pd.read_csv(icd_file)

    # Remove NaN values and sort chronologically
    chrono_df = df[~df['NOTED_DATE'].isna()].sort_values('NOTED_DATE')

    # Unwrap data into dicts
    code_dic = {}
    timestamps = {}
    for key, code, stamp in chrono_df.values:
        if key not in code_dic:
            code_dic[key] = []
            timestamps[key] = []
        code_dic[key].append(code)
        timestamps[key].append(stamp[:10])  # Only perserve year/month/day

    # Create visit segments
    visit_segments = {}
    for key, values in timestamps.items():
        segment = [0]
        visit_count = 0
        for i in range(1, len(values)):
            if values[i-1] != values[i]:    # Check if the date has changed, if so increment the counter
                visit_count += 1
            segment.append(visit_count)
        visit_segments[key] = segment

    # Tokenize
    tokenizer = EHRTokenizer()
    tokenized_codes = tokenizer(list(code_dic.values()))

    # Save to file
    tokenizer.save_vocab('vocabulary.pt')
    with open('tokenized_icd10.pt', 'wb') as f:
        torch.save(tokenized_codes, f)
    with open('visit_segments.pt', 'wb') as f:
        torch.save(list(visit_segments.values()), f)


def split_testsets(icd10="tokenized_icd10.pt", segments="visit_segments.pt", test_ratio=0.2):
    with open(icd10, 'rb') as f:
        codes = torch.load(f)
    with open(segments, 'rb') as f:
        segments = torch.load(f)

    N = len(codes)
    np.random.seed(0)

    indices = np.random.permutation(N)

    N_test = int(N*test_ratio)
    test_indices = indices[:N_test]
    train_indices = indices[N_test:]

    test_codes = [codes[ind] for ind in test_indices]
    train_codes = [codes[ind] for ind in train_indices]

    test_segments = [segments[ind] for ind in test_indices]
    train_segments = [segments[ind] for ind in train_indices]

    with open('features.train', 'wb') as f:
        torch.save((train_codes, train_segments), f)
    with open('features.test', 'wb') as f:
        torch.save((test_codes, test_segments), f)

    return (train_codes, train_segments), (test_codes, test_segments)


if __name__ == '__main__':
    pickle_data()       # Should only be done once
    split_testsets()    # Should only be done once

