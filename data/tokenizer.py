import torch


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

    def batch_encode(self, seqs, padding=True, truncation=768):
        if padding:
            max_len = truncation

        data = {
            'events': [],
            'attention_mask': []
        }

        for seq in seqs:
            seq = self.insert_special_tokens(seq)

            # Truncation
            if truncation and len(seq) > truncation:
                seq = [seq[0]] + seq[-(truncation-1):]    # Keep CLS and newest events

            # Created after truncation for effiency
            attention_mask = [1] * len(seq)

            # Padding
            if padding:
                difference = max_len - len(seq)
                seq += [('[PAD]', 0, 0, 0)] * difference
                attention_mask += [0] * difference

            # Tokenization
            tokenized_seq = self.encode(seq)

            data['events'].append(torch.tensor(tokenized_seq, dtype=torch.int))
            data['attention_mask'].append(torch.tensor(attention_mask, dtype=torch.int))

        return data

    def encode(self, seq):
        if self.new_vocab:
            for event in seq:
                concept = event[0]
                if concept not in self.vocabulary:
                    self.vocabulary[concept] = len(self.vocabulary)

        return [(self.vocabulary[concept], age, abspos, segment) for concept, age, abspos, segment in seq]

    def save_vocab(self, dest):
        with open(dest, 'wb') as f:
            torch.save(self.vocabulary, f)

    def insert_special_tokens(self, seq):
        new_seq = [('[CLS]', 0, 0, 0)]
        for i, (token, age, abspos, segment) in enumerate(seq):
            new_seq.append((token, age, abspos, segment))
            #     End of seq     OR   age[i]  !=  age[i+1]      -> Insert SEP token
            if i == len(seq) - 1 or seq[i][1] != seq[i+1][1]:
                new_seq.append(('[SEP]', age, abspos, segment))

        return new_seq

