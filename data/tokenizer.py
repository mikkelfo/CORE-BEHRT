import torch
from transformers import BatchEncoding


class EHRTokenizer():
    def __init__(self, config, vocabulary=None):
        self.config = config
        if vocabulary is None:
            self.new_vocab = True
            self.vocabulary = {
                '[PAD]': 0,
                '[CLS]': 1, 
                '[SEP]': 2,
                '[UNK]': 3,
                '[MASK]': 4,
            }
        else:
            self.new_vocab = False
            self.vocabulary = vocabulary

    def __call__(self, seqs: dict(str, list), padding=True, truncation=768):
        return self.batch_encode(seqs, padding, truncation)

    def batch_encode(self, features: dict(str, list(list)), padding=True, truncation=768):
        if padding or truncation:
            max_len = truncation

        data = {key: [] for key in features.keys()}
        data['attention_mask'] = []

        N_features = len(features['concept'])
        for i in range(N_features):
            seq = {key: value[i] for key, value in features.items()}
            seq = self.insert_special_tokens(seq)

            # Truncation
            if truncation and len(seq['concept']) > truncation:
                seq = self.truncate(seq, max_len=max_len)
            
            # Created after truncation for efficiency
            seq['attention_mask'] = [1] * len(seq['concept'])

            # Padding
            if padding and len(seq) < max_len:
                seq = self.pad(seq, max_len=max_len)

            # Encode concepts
            seq['concept'] = self.encode(seq['concept'])

            for key, value in seq.items():
                data[key].append(value)

        return BatchEncoding(data, tensor_type="pt")

    def encode(self, seq):
        if self.new_vocab:
            for concept in seq:
                if concept not in self.vocabulary:
                    self.vocabulary[concept] = len(self.vocabulary)

        return [self.vocabulary[concept] for concept in seq]

    def truncate(self, seq, max_len):
        truncation_length = max_len - 1 - self.config.background_length
        
        if seq['concept'][-truncation_length] == '[SEP]':
            truncation_length += 1

        for key, value in seq.items():
            seq[key] = value[1+self.config.background_length] + value[-(truncation_length):]    # Keep CLS, background sentence and newest information

        return seq

    def pad(self, seq, max_len):
        difference = max_len - len(seq)
        for key, value in seq.items():
            token = '[PAD]' if key == 'concept' else 0
            seq[key] = value + [token] * difference
        return seq

    def insert_special_tokens(self, seq: dict(str, list)):
        
        # Insert SEP tokens
        if self.config.sep_tokens and 'segment' in seq:     # Check if SEP tokens are enabled
            seq = self.insert_sep_tokens(seq)

        # Insert CLS token
        if self.config.cls_token:                           # Check if CLS tokens are enabled
            seq = self.insert_cls_token(seq)
        
        return seq

    def insert_sep_tokens(self, seq):
        new_seq = {key: [] for key in seq.keys()}
        for i in range(len(seq['concept'])):                # Iterate over each entry
            for key, value in seq.items():                  # Iterate over all features
                new_seq[key].append(value[i])               # Copy the seq
                
                # Insert SEP token if segment changes
                if seq['segment'][i] != seq['segment'][i]:  # If segment changes
                    if key == 'concept':
                        new_seq[key].append('[SEP]')        # Insert SEP token
                    else:
                        new_seq[key].append(value[i])       # Copy feature value
        return new_seq

    def insert_cls_token(self, seq):
        for key, value in seq.items():
            token = '[CLS]' if key == 'concept' else 0      # Determine token value (CLS for concepts, 0 for rest)
            seq[key] = [token] + value
        return seq

    def save_vocab(self, dest):
        with open(dest, 'wb') as f:
            torch.save(self.vocabulary, f)

