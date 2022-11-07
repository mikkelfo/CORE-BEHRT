import torch
from transformers import BatchEncoding


class EHRTokenizer():
    def __init__(self, vocabulary=None):
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

    def __call__(self, seq, padding=True, truncation=768):
        return self.batch_encode(seq, padding, truncation)

    def batch_encode(self, seqs, padding=True, truncation=768) -> BatchEncoding:
        if padding:
            max_len = max([len(sum(seq, [])) for seq in seqs]) + 2

        data = {
            'input_ids': [],
            'visit_segments': [],
            'attention_mask': [],
        }

        for seq in seqs:
            tokenized_seq = [self.vocabulary['[CLS]']]
            visit_segments = [0]

            # Tokenize each visit
            for i, codes in enumerate(seq):
                tokenized_seq += self.encode(codes)     # Encode codes          (input_ids)
                visit_segments += [i] * len(codes)      # Create visit segments (token_type_ids)
            tokenized_seq.append(self.vocabulary['[SEP]'])
            attention_mask = [1] * len(tokenized_seq)   # Create mask           (attention_mask) 
            visit_segments.append(i)

            # Padding
            if padding:
                difference = max_len - len(tokenized_seq)
                tokenized_seq += [self.vocabulary['[PAD]']] * difference
                visit_segments += [0] * difference
                attention_mask += [0] * difference

            # Truncating
            if truncation:
                tokenized_seq = tokenized_seq[:truncation]
                visit_segments = visit_segments[:truncation]
                attention_mask = attention_mask[:truncation]

            data['input_ids'].append(tokenized_seq)
            data['visit_segments'].append(visit_segments)
            data['attention_mask'].append(attention_mask)


        return BatchEncoding(data, tensor_type="pt")

    def encode(self, seq):
        if self.new_vocab:
            for code in seq:
                if code not in self.vocabulary:
                    self.vocabulary[code] = len(self.vocabulary)

        return [self.vocabulary[code] for code in seq]

    def save_vocab(self, dest):
        with open(dest, 'wb') as f:
            torch.save(self.vocabulary, f)

