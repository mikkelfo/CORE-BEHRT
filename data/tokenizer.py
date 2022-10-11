import torch


class EHRTokenizer():
    def __init__(self, vocabulary=None):
        if vocabulary is None:
            self.new_vocab = True
            self.vocabulary = {'pad_token': 0}
        else:
            self.new_vocab = False
            self.vocabulary = vocabulary

    def __call__(self, seq):
        return self.batch_encode(seq)

    def batch_encode(self, seqs, padding=True, truncation=None):
        max_len = max([len(seq) for seq in seqs])
        
        output_seqs = []

        for seq in seqs:
            # Tokenizing
            tokenized_seq = self.encode(seq)
            # Padding
            if padding:
                difference = max_len - len(tokenized_seq)
                padded_seq = tokenized_seq + [0] * difference
            # Truncating
            truncated_seq = padded_seq[:truncation]

            output_seqs.append(truncated_seq)

        return output_seqs

    def encode(self, seq):
        if self.new_vocab:
            for code in seq:
                if code not in self.vocabulary:
                    self.vocabulary[code] = len(self.vocabulary)

        return [self.vocabulary[code] for code in seq]

    def save_vocab(self, dest):
        with open(dest, 'wb') as f:
            torch.save(self.vocabulary, f)

