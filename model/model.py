from typing import List, Tuple

from embeddings.ehr import EhrEmbeddings
from loss import loss
from transformers import BertForMaskedLM


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
    ):                     
                                                                              # notice this
        output = super().forward(input_ids, inputs_embeds=inputs_embeds, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=target)                                                                                   

        return output


class HierarchicalBertEHRModel(BertEHRModel):
    def __init__(self, config, leaf_nodes: List[Tuple], trainable_loss_weights=True):
        super().__init__(config)
        config.vocab_size = config.emb_vocab_size
        # self.bert.embeddings = EhrEmbeddings(config)
        self.h_loss = loss.CE_FlatSoftmax_MOP(leaf_nodes, trainable_loss_weights, )
        # we need to add loss params to model params
    def forward(self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        target=None,    # Use this instead of labels (avoids CE loss)
    ):                                                                                                      # notice this
        outputs = super().forward(input_ids, inputs_embeds=inputs_embeds, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, target=None) # we should output hidden states, so we don't compute CE loss inside        
        if target is not None:
            outputs['loss'] = self.h_loss(outputs.logits, target)

        return outputs

    