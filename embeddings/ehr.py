import torch.nn as nn
from embeddings.time2vec import Time2Vec
import torch


class EhrEmbeddings(nn.Module):
    def __init__(self, config):
        super(EhrEmbeddings, self).__init__()
        self.config = config
        
        self.concept_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.age_embeddings = Time2Vec(config.max_position_embeddings, config.hidden_size)
        self.abspos_embeddings = Time2Vec(config.max_position_embeddings, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.LongTensor,                  # concepts
        token_type_ids: torch.LongTensor = None,      # segments
        position_ids: dict[str, torch.Tensor] = None, # age and abspos
        inputs_embeds: torch.Tensor = None,
        **kwargs
    ):
        if inputs_embeds is not None:
            return inputs_embeds

        embeddings = self.concept_embeddings(input_ids)
        
        if token_type_ids is not None:
            segments_embedded = self.segment_embeddings(token_type_ids)
            embeddings += segments_embedded

        if position_ids is not None:
            if 'age' in position_ids:
                ages_embedded = self.age_embeddings(position_ids['age'])
                embeddings += ages_embedded
            if 'abspos' in position_ids:
                abspos_embedded = self.abspos_embeddings(position_ids['abspos'])
                embeddings += abspos_embedded
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

