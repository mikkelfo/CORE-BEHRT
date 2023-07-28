import torch.nn as nn
from embeddings.time2vec import Time2Vec
import torch


class EhrEmbeddings(nn.Module):
    """
        EHR Embeddings

        Forward inputs:
            input_ids: torch.LongTensor             - (batch_size, sequence_length)
            token_type_ids: torch.LongTensor        - (batch_size, sequence_length)
            position_ids: dict(str, torch.Tensor)   - (batch_size, sequence_length)

        Config:
            vocab_size: int                         - size of the vocabulary
            hidden_size: int                        - size of the hidden layer
            type_vocab_size: int                    - size of max segments
            layer_norm_eps: float                   - epsilon for layer normalization
            hidden_dropout_prob: float              - dropout probability
            linear: bool                            - whether to linearly scale embeddings (a: concept, b: age, c: abspos, d: segment)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
       
        self.concept_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.age_embeddings = Time2Vec(1, config.hidden_size)
        self.abspos_embeddings = Time2Vec(1, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.config.to_dict().get('linear', False):
            self.a = nn.Parameter(torch.ones(1))
            self.b = nn.Parameter(torch.zeros(1))
            self.c = nn.Parameter(torch.zeros(1))
            self.d = nn.Parameter(torch.zeros(1))
        else:
            self.a = self.b = self.c = self.d = 1

    def forward(
        self,
        input_ids: torch.LongTensor,                  # concepts
        token_type_ids: torch.LongTensor = None,      # segments
        position_ids: dict = None, # age and abspos
        inputs_embeds: torch.Tensor = None,
        **kwargs
    ):
        if inputs_embeds is not None:
            return inputs_embeds

        embeddings = self.a * self.concept_embeddings(input_ids)
        
        if token_type_ids is not None:
            segments_embedded = self.segment_embeddings(token_type_ids)
            embeddings += self.b * segments_embedded

        if position_ids is not None:
            if 'age' in position_ids:
                ages_embedded = self.age_embeddings(position_ids['age'])
                embeddings += self.c * ages_embedded
            if 'abspos' in position_ids:
                abspos_embedded = self.abspos_embeddings(position_ids['abspos'])
                embeddings += self.d * abspos_embedded
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

