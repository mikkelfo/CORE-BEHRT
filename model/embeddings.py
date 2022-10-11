import torch
import torch.nn as nn


class EhrEmbeddings(nn.Module):
    def __init__(self, config):
        super(EhrEmbeddings, self).__init__()
        self.config = config
        self.code_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        code_ids,
        position_ids=None,
        segment_ids=None,
    ):
        code_embedded = self.code_embeddings(code_ids)

        # Position ids are constructed if not present
        if position_ids is None:
            position_ids = torch.arange(self.config.max_position_embeddings).expand((1, -1))
        position_embedded = self.position_embeddings(position_ids)

        # If no segment_ids are provided, all code_ids are assumed to belong to a single visit
        if segment_ids is None:
            segment_ids = torch.zeros_like(code_ids)
        segment_embedded = self.segment_embeddings(segment_ids)
        
        embeddings = code_embedded + position_embedded + segment_embedded

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

