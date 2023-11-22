from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from embeddings.time2vec import Time2Vec
from transformers import BertConfig

TIME2VEC_AGE_MULTIPLIER = None
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
        self.concept_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.age_embeddings = Time2Vec(1, config.hidden_size, init_scale=TIME2VEC_AGE_MULTIPLIER,
                                       clip_min=TIME2VEC_MIN_CLIP, clip_max=TIME2VEC_MAX_CLIP)
        self.abspos_embeddings = Time2Vec(1, config.hidden_size, init_scale=TIME2VEC_ABSPOS_MULTIPLIER,
                                          clip_min=TIME2VEC_MIN_CLIP, clip_max=TIME2VEC_MAX_CLIP)
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
    

class BehrtEmbeddings(BaseEmbeddings):
    """
    Construct the embeddings from word, segment, age and position
    """
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.initialize_embeddings(config)
        self.initialize_linear_params(config)

    def initialize_embeddings(self, config: BertConfig)->None:
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # !tempory solution for compatibility with old models
        cfg = config.to_dict()
        max_segment = cfg.get('max_segment_embeddings', None)
        self.segment_embeddings = nn.Embedding(max_segment if max_segment is not None else cfg.get('type_vocab_size', None) , 
                                               config.hidden_size)
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

    def forward(self, input_ids:torch.Tensor, 
                token_type_ids:torch.Tensor=None, 
                position_ids:Dict[str, torch.Tensor]=None, 
                inputs_embeds:torch.Tensor = None, **kwargs)->torch.Tensor:
        if inputs_embeds is not None:
            return inputs_embeds
        embeddings = self.a * self.word_embeddings(input_ids)
        
        if position_ids is not None:
            if 'age' in position_ids:
                age_embed = self.age_embeddings(position_ids['age'])
                embeddings += self.c * age_embed
            if 'position_ids' in position_ids:
                posi_embed = self.posi_embeddings(position_ids['position_ids'])
                embeddings += self.d * posi_embed
        segment_embed = self.b * self.segment_embeddings(token_type_ids)
        embeddings += segment_embed
                
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def _init_posi_embedding(self, max_position_embedding: int, hidden_size:int)->torch.Tensor:
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)



