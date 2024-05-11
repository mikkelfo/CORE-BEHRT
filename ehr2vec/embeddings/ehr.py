import logging
from typing import Dict

import torch
import torch.nn as nn
from transformers import BertConfig

from ehr2vec.embeddings.time2vec import Time2Vec

logger = logging.getLogger(__name__)  # Get the logger for this module

TIME2VEC_AGE_MULTIPLIER = 1e-2
TIME2VEC_ABSPOS_MULTIPLIER = 1e-4
TIME2VEC_MIN_CLIP = -100
TIME2VEC_MAX_CLIP = 100
class BaseEmbeddings(nn.Module):
    """Base Embeddings class with shared methods"""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.LayerNorm = nn.LayerNorm(config.hidden_size, 
                                      eps=config.to_dict().get('layer_norm_eps', 1e-12))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def apply_layer_norm_and_dropout(self, embeddings: torch.Tensor)->torch.Tensor:
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)

    def initialize_linear_params(self, config)->None:
        if config.to_dict().get('linear', False):
            self.a = nn.Parameter(torch.ones(1))
            self.b = nn.Parameter(torch.zeros(1))
            self.c = nn.Parameter(torch.zeros(1))
            self.d = nn.Parameter(torch.zeros(1))
        else:
            self.a = self.b = self.c = self.d = 1

class EhrEmbeddings(BaseEmbeddings):
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
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.initialize_embeddings(config)
        self.initialize_linear_params(config)

    def initialize_embeddings(self, config: BertConfig)->None:
        logger.info("Initialize Concept/Segment/Age embeddings.")
        self.concept_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.age_embeddings = Time2Vec(1, config.hidden_size, init_scale=TIME2VEC_AGE_MULTIPLIER, clip_min=TIME2VEC_MIN_CLIP, clip_max=TIME2VEC_MAX_CLIP)
        if config.to_dict().get('abspos_embeddings', True):
            logger.info("Initialize time2vec(abspos) embeddings.")
            self.abspos_embeddings = Time2Vec(1, config.hidden_size, init_scale=TIME2VEC_ABSPOS_MULTIPLIER, clip_min=TIME2VEC_MIN_CLIP, clip_max=TIME2VEC_MAX_CLIP)
        else:
            self.abspos_embeddings = None
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
    def forward(
        self,
        input_ids: torch.LongTensor,                  # concepts
        token_type_ids: torch.LongTensor = None,      # segments
        position_ids: Dict[str, torch.Tensor] = None, # age and abspos
        inputs_embeds: torch.Tensor = None,
        **kwargs
    )->torch.Tensor:
        
        if inputs_embeds is None:
            embeddings = self.a * self.concept_embeddings(input_ids)
            
            if token_type_ids is not None:
                segments_embedded = self.segment_embeddings(token_type_ids)
                embeddings += self.b * segments_embedded

            if position_ids is not None:
                if 'age' in position_ids:
                    ages_embedded = self.age_embeddings(position_ids['age'])
                    embeddings += self.c * ages_embedded
                if 'abspos' in position_ids:
                    if self.abspos_embeddings is not None:
                        abspos_embedded = self.abspos_embeddings(position_ids['abspos'])
                        embeddings += self.d * abspos_embedded
                    
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)

            return embeddings
        else:
            return inputs_embeds
