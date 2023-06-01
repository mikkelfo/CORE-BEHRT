from transformers import BertModel
from embeddings.ehr import EhrEmbeddings
import torch.nn as nn
from model.heads import MLMHead, FineTuneHead
import torch


class BertEHRModel(BertModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = EhrEmbeddings(config)
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
        labels_mask=None,
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
        outputs.logits = logits

        if labels is not None:
            outputs.loss = self.get_loss(logits, labels, labels_mask)

        return outputs

    def get_loss(self, logits, labels, labels_mask=None):
        return self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))


class BertForFineTuning(BertEHRModel):
    def __init__(self, config):
        super().__init__(config)
        if config.pos_weight:
            pos_weight = torch.tensor(config.pos_weight)
        else:
            pos_weight = None

        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.cls = FineTuneHead(config)

    def get_loss(self, hidden_states, labels, labels_mask=None):    
        return self.loss_fct(hidden_states.view(-1), labels.view(-1))


class HierarchicalBertForPretraining(BertEHRModel):
    def __init__(self, config):
        super().__init__(config)

        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

        # TODO: Make this configurable
        self.linear_combination = torch.arange(config.levels, 0, -1) / config.levels

    def get_loss(self, logits, labels, labels_mask):
        logits = logits.view(-1, self.config.levels, self.config.vocab_size)    # Reshape to (batch_size*seq_len, levels, vocab_size) for masking
        logits = logits[labels_mask.bool().view(-1)]                            # Remove where original target was -100
        loss = self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1, self.config.vocab_size))  # Reshape both to (batch_size*seq_len*levels, vocab_size)
        loss = loss.view(-1, self.config.levels) * self.linear_combination  # Reshape to (batch_size*seq_len, levels) and apply linear combination

        return loss.mean()
