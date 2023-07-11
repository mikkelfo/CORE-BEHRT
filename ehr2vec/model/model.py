import torch
import torch.nn as nn
from embeddings.ehr import EhrEmbeddings
from model.heads import FineTuneHead, HMLMHead, MLMHead
from transformers import BertModel
from tree.tree import TreeBuilder


class BertEHRModel(BertModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = EhrEmbeddings(config)
        self.loss_fct = nn.CrossEntropyLoss()
        
        self.cls = MLMHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None, # It's used for both attention and labels mask
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
        outputs.logits = logits

        if labels is not None:
            outputs.loss = self.get_loss(logits, labels, attention_mask)

        return outputs

    def get_loss(self, logits, labels, attention_mask=None):
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
    def __init__(self, config, tree, tree_matrix=None):
        super().__init__(config)

        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        self.cls = HMLMHead(config)

        # TODO: Make this configurable
        self.linear_combination = torch.arange(config.levels, 0, -1) / config.levels
        
        if not tree_matrix:
            tree_matrix = tree.get_tree_matrix()

        self.tree_matrix = tree_matrix
        self.tree_matrix_sparse = self.tree_matrix.to_sparse()
        self.tree_mask = self.tree_matrix.any(2)

    def get_loss(self, logits, labels, labels_mask):
        levels = 5
        batch_size, seq_len, leaves = logits.shape
        logits = logits.permute(2, 0, 1).view(leaves, -1)  # (leaves, batch_size*seq_len)

        # Calculate level predictions (one level each)
        acc_loss = 0
        for i in range(levels):
            if levels-1 == i:
                level_predictions = logits
            else:
                level_predictions = torch.matmul(self.tree_matrix[i, self.tree_mask[i]], logits)    # (leaves, leaves) @ (leaves, batch_size*seq_len) -> (leaves, batch_size*seq_len)
            level_predictions = level_predictions.transpose(1, 0)
            level_predictions = level_predictions[labels_mask.view(-1)]
            level_labels = labels[:, i, self.tree_mask[i]]
            loss = self.loss_fct(level_predictions, level_labels)
            acc_loss += (loss * self.linear_combination[i]).mean()
        
        return acc_loss / levels

