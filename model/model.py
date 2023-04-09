from transformers import BertForMaskedLM
from embeddings.ehr import EhrEmbeddings
from torch import softmax
import torch
from typing import List, Tuple
from loss import loss

class BertEHRModel(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert.embeddings = EhrEmbeddings(config)

    def forward(self, 
        input_ids, 
        token_type_ids=None, 
        attention_mask=None, 
        position_ids=None, 
        inputs_embeds=None, 
        target=None,    # Gets assigned to labels for CE loss
    ):                                                                                                  # notice this
        return super().forward(input_ids, token_type_ids, attention_mask, position_ids, inputs_embeds, labels=target)


class HierarchicalBertEHRModel(BertEHRModel):
    def __init__(self, config, leaf_nodes: List[Tuple], trainable_loss_weights=True):
        super().__init__(config)
        self.h_loss = loss.CE_FlatSoftmax_MOP(leaf_nodes, trainable_loss_weights, )
        
    def forward(self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        target=None,    # Use this instead of labels (avoids CE loss)
    ):                                                                                                      # notice this
        outputs = super().forward(input_ids, token_type_ids, attention_mask, position_ids, inputs_embeds, labels=None) # we should output hidden states, so we don't compute CE loss inside
        outputs.loss = self.h_loss(outputs.logits, target)

        return outputs

    