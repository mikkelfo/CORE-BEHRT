import torch
import torch.nn as nn
from typing import Dict
from transformers import BertConfig

from corebehrt.embeddings.time2vec import Time2Vec

# Constants (data dependent) for Time2Vec
TIME2VEC_AGE_MULTIPLIER = 1e-2
TIME2VEC_ABSPOS_MULTIPLIER = 1e-4
TIME2VEC_MIN_CLIP = -100
TIME2VEC_MAX_CLIP = 100

class EhrEmbeddings(nn.Module):
    """
        Forward inputs:
            input_ids: torch.LongTensor             - (batch_size, sequence_length)
            token_type_ids: torch.LongTensor        - (batch_size, sequence_length)
            position_ids: dict(str, torch.Tensor)   - (batch_size, sequence_length)
                We abuse huggingface's standard position_ids to pass additional information (age, abspos)
                This makes BertModel's forward method compatible with our EhrEmbeddings

        Config:
            vocab_size: int                         - size of the vocabulary
            hidden_size: int                        - size of the hidden layer
            type_vocab_size: int                    - size of max segments
            layer_norm_eps: float                   - epsilon for layer normalization
            hidden_dropout_prob: float              - dropout probability
    """
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.to_dict().get('layer_norm_eps', 1e-12))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initalize embeddings
        self.concept_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.age_embeddings = Time2Vec(1, config.hidden_size, init_scale=TIME2VEC_AGE_MULTIPLIER, clip_min=TIME2VEC_MIN_CLIP, clip_max=TIME2VEC_MAX_CLIP)
        self.abspos_embeddings = Time2Vec(1, config.hidden_size, init_scale=TIME2VEC_ABSPOS_MULTIPLIER, clip_min=TIME2VEC_MIN_CLIP, clip_max=TIME2VEC_MAX_CLIP)
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
    def forward(
        self,
        input_ids: torch.LongTensor,                  # concepts
        token_type_ids: torch.LongTensor = None,      # segments
        position_ids: Dict[str, torch.Tensor] = None, # age and abspos
        inputs_embeds: torch.Tensor = None,
        **kwargs
    )->torch.Tensor:
        if inputs_embeds is not None:
            return inputs_embeds
        
        embeddings = self.concept_embeddings(input_ids)
        
        embeddings += self.segment_embeddings(token_type_ids)
        embeddings += self.age_embeddings(position_ids['age'])
        embeddings += self.abspos_embeddings(position_ids['abspos'])
                    
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
