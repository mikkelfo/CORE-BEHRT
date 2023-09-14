import torch
import torch.nn as nn
from transformers import BertModel
from src.embeddings.ehr import EhrEmbeddings
from src.model.heads import MLMHead, FineTuneHead, HMLMHead
import src.common.loading as loading


class BertEHRModel(BertModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = EhrEmbeddings(config)
        self.loss_fct = nn.CrossEntropyLoss()

        self.cls = MLMHead(config)

    def forward(self, batch):
        # Unpack batch
        input_ids = batch["concept"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["segment"] if "segment" in batch else None
        position_ids = {
            "age": batch["age"] if "age" in batch else None,
            "abspos": batch["abspos"] if "abspos" in batch else None,
        }
        labels = batch["target"] if "target" in batch else None
        labels_mask = batch["target_mask"] if "target_mask" in batch else None
        inputs_embeds = batch["inputs_embeds"] if "inputs_embeds" in batch else None

        # Forward pass
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]  # Last hidden state
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

        self.loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.cls = HMLMHead(config)

        tree = loading.tree(config)
        self.tree_matrix = tree.get_tree_matrix()
        self.tree_mask = self.tree_matrix.any(2)
        self.num_levels = self.tree_matrix.shape[0]

        if config.linear_combination is None:
            self.linear_combination = (
                torch.arange(self.num_levels, 0, -1) / self.num_levels
            )
        else:
            self.linear_combination = torch.tensor(config.linear_combination)

    def get_loss(self, logits, labels, labels_mask):
        batch_size, seq_len, leaves = logits.shape
        logits = logits.permute(2, 0, 1).view(
            leaves, -1
        )  # (leaves, batch_size*seq_len)

        # Calculate level predictions (one level each)
        acc_loss = 0
        for i in range(self.num_levels):
            if self.num_levels - 1 == i:
                level_predictions = logits
            else:
                level_predictions = torch.matmul(
                    self.tree_matrix[i, self.tree_mask[i]], logits
                )  # (leaves, leaves) @ (leaves, batch_size*seq_len) -> (leaves, batch_size*seq_len)
            level_predictions = level_predictions.transpose(1, 0)
            level_predictions = level_predictions[labels_mask.view(-1)]
            level_labels = labels[:, i, self.tree_mask[i]]
            loss = self.loss_fct(level_predictions, level_labels)
            acc_loss += (loss * self.linear_combination[i]).mean()

        return acc_loss / self.num_levels
