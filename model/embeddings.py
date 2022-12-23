import torch.nn as nn


class EhrEmbeddings(nn.Module):
    def __init__(self, config):
        super(EhrEmbeddings, self).__init__()
        self.config = config
        self.concept_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        self.age_embeddings = nn.Embedding(100, config.hidden_size) # Swap for Time2Vec
        self.abspos_embeddings = nn.Embedding(10000, config.hidden_size) # Swap for Time2Vec
        
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        concepts,
        ages=None,
        abspos=None,
        segments=None,
    ):
        concepts_embedded = self.concept_embeddings(concepts)

        if ages is None:
            ages_embedded = 0
        else:
            ages_embedded = self.age_embeddings(ages)

        if abspos is None:
            abspos_embedded = 0
        else:
            abspos_embedded = self.abspos_embeddings(abspos)

        if segments is None:
            segments_embedded = 0
        else:
            segments_embedded = self.segment_embeddings(abspos)
        
        embeddings = concepts_embedded + ages_embedded + abspos_embedded + segments_embedded

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

