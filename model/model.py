from transformers import BertModel
from embeddings.ehr import EhrEmbeddings
import torch.nn as nn
from heads import MLMHead, FineTuneHead


class BertEHRModel(BertModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert.embeddings = EhrEmbeddings(config)
        self.loss_fct = nn.CrossEntropyLoss()
        
        self.cls = MLMHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]    # Last hidden state
        logits = self.cls(sequence_output)

        if labels is not None:
            outputs.loss = self.get_loss(logits, labels)

        return outputs

    def get_loss(self, logits, labels):
        return self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))


class BertForFineTuning(BertEHRModel):
    def __init__(self, config):
        super().__init__(config)

        self.loss_fct = nn.BCEWithLogitsLoss()
        self.cls = FineTuneHead(config)

    def get_loss(self, logits, labels):
        return self.loss_fct(logits.view(-1), labels.view(-1))

