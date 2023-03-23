from transformers import BertForMaskedLM
from embeddings.ehr import EhrEmbeddings


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
    def __init__(self, config):
        super().__init__(config)

    def forward(self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        target=None,    # Use this instead of labels (avoids CE loss)
    ):                                                                                                      # notice this
        outputs = super().forward(input_ids, token_type_ids, attention_mask, position_ids, inputs_embeds, labels=None)
        outputs.loss = self.get_loss(outputs.logits, target)

        return outputs

    # Implemenet h_loss here
    # You can take more __init__ parameters in if you need it for loss computations
    def get_loss(self, logits, target):
        pass
