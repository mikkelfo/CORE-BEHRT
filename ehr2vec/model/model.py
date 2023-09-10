import torch
import torch.nn as nn
from embeddings.ehr import EhrEmbeddings
from model.heads import FineTuneHead, HMLMHead, MLMHead
from transformers import BertModel
from common.config import instantiate

class BertEHREncoder(BertModel):
    def __init__(self, config):
        super().__init__(config)
        if not config.to_dict().get('embedding', None):
            self.embeddings = EhrEmbeddings(config)
        else:
            self.embeddings = instantiate(config.embedding, config)
    def forward(
        self,
        batch: dict,
    ):
        position_ids={
            'age': batch.get('age', None),
            'abspos': batch.get('abspos', None),
            'dosage': batch.get('dosage', None),
            'unit': batch.get('unit', None)
        },
        outputs = super().forward(
            input_ids=batch['concept'],
            attention_mask=batch.get('attention_mask', None),
            token_type_ids=batch.get('segment', None),
            position_ids=position_ids,
            inputs_embeds=batch.get('embeddings', None),
        )
        return outputs

class BertEHRModel(BertEHREncoder):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = nn.CrossEntropyLoss()
        
        self.cls = MLMHead(config)

    def forward(
        self,
        batch: dict,
    ):
        outputs = super().forward(batch)

        sequence_output = outputs[0]    # Last hidden state
        logits = self.cls(sequence_output)
        outputs.logits = logits

        if batch.get('target', None) is not None:
            outputs.loss = self.get_loss(logits, batch['target'])

        return outputs

    def get_loss(self, logits, labels):
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
    def __init__(self, config, tree=None, tree_matrix=None):
        super().__init__(config)

        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        self.cls = HMLMHead(config)

        self.levels = config.levels
        
        if getattr(config, "trainable_level_weights"):
            self.trainable_level_weights = config.trainable_level_weights
            self.linear_combination = nn.Parameter(torch.arange(self.levels-1, 0, -1).float() / self.levels)
        else:
            self.trainable_level_weights = False
            self.linear_combination = torch.arange(self.levels-1, 0, -1) / self.levels
        
        if not isinstance(tree_matrix, torch.Tensor):
            assert tree is not None, "Either tree or tree_matrix must be provided"
            tree_matrix = tree.get_tree_matrix()
        self.register_tree_matrix(tree_matrix) 
        

    def register_tree_matrix(self, tree_matrix):
        tree_mask = tree_matrix.any(2)
        for i in range(self.levels):
            self.register_buffer(f'tree_matrix_{i}', tree_matrix[i, tree_mask[i]])
        self.register_buffer('tree_mask', tree_mask)

    def forward(
        self,
        batch: dict,
    ):
        target = batch.pop('target', None)
        outputs = super().forward(batch)


        sequence_output = outputs[0]    # Last hidden state
        logits = self.cls(sequence_output)
        outputs.logits = logits

        if target is not None:
            outputs.loss = self.get_loss(logits, target, batch['attention_mask'])

        return outputs

    def get_loss(self, logits, labels, target_mask):
        levels = labels.shape[1]
        batch_size, seq_len, leaves = logits.shape
        logits = logits.permute(2, 0, 1).view(leaves, -1)  # (leaves, batch_size*seq_len)

        # Calculate level predictions (one level each)
        acc_loss = 0
        for i in range(levels):
            if levels-1 == i:
                level_predictions = logits
            else:
                level_predictions = torch.matmul(getattr(self, f"tree_matrix_{i}"), logits)    # (leaves, leaves) @ (leaves, batch_size*seq_len) -> (leaves, batch_size*seq_len)
            level_predictions = level_predictions.transpose(1, 0)
            level_predictions = level_predictions[target_mask.view(-1)]
            level_labels = labels[:, i, self.tree_mask[i]]
            
            loss = self.loss_fct(level_predictions, level_labels)
            if i == 0:
                acc_loss = loss.mean()
            else:
                acc_loss += loss.mean() * self.linear_combination[i-1]
                if self.trainable_level_weights:
                    acc_loss -= torch.log(self.linear_combination[i-1])
        
        return acc_loss / levels

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Returns a dictionary containing a whole state of the module.
        This is an override of the default state_dict function in nn.Module
        It excludes the tree_matrix_buffer and tree_mask_buffer from the state_dict.
        """
        result = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for i in range(self.levels):
            result.pop(f'tree_matrix_{i}', None)
        result.pop('tree_mask', None)
        return result
