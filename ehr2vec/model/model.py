import torch
import torch.nn as nn
from transformers import BertModel, RoFormerModel

from ehr2vec.embeddings.ehr import BehrtEmbeddings, EhrEmbeddings
from ehr2vec.model.heads import FineTuneHead, HMLMHead, MLMHead
from ehr2vec.model.activations import SwiGLU

import logging
logger = logging.getLogger(__name__)
class BertEHREncoder(BertModel):
    def __init__(self, config):
        super().__init__(config)
        if not config.to_dict().get('embedding', None):
            logger.info("No embedding type specified. Using default EHR embedding.")
            self.embeddings = EhrEmbeddings(config)
        elif config.embedding == 'original_behrt':
            logger.info("Using original Behrt embedding.")
            self.embeddings = BehrtEmbeddings(config)
        else:
            raise ValueError(f"Unknown embedding type: {config.embedding}")

        # Activate transformer++ recipe
        if config.to_dict().get('plusplus'):
            logger.info("Using Transformer++ recipe.")
            config.embedding_size = config.hidden_size
            config.rotary_value = False
            self.encoder = RoFormerModel(config).encoder

            for layer in self.encoder.layer:
                layer.intermediate.intermediate_act_fn = SwiGLU(config)
                # layer.output.LayerNorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps) # We dont use RMSNorm (only speedup, no performance gain)

    def forward(
        self,
        batch: dict,
    ):
        present_keys = [k for k in ['age', 'abspos', 'position_ids', 'dosage', 'unit'] if k in batch]
        position_ids = {key: batch.get(key) for key in  present_keys}
        outputs = super().forward(
            input_ids=batch['concept'],
            attention_mask=batch.get('attention_mask', None),
            token_type_ids=batch.get('segment', None),
            position_ids=position_ids,
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

class BertForRegression(BertEHRModel):
    """Regression model for fine-tuning. Uses MSE loss"""
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = nn.MSELoss()
        self.cls = FineTuneHead(config)

    def get_loss(self, hidden_states, labels, labels_mask=None):    
        return self.loss_fct(hidden_states.view(-1), labels.view(-1))

class HierarchicalBertForPretraining(BertEHRModel):
    def __init__(self, config, tree=None, tree_matrix=None):
        super().__init__(config)

        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        self.cls = HMLMHead(config)

        self.levels = config.levels
        
        self.initialize_level_weights(config)
        self.validate_and_register_tree_matrix(tree, tree_matrix)
        
    def initialize_level_weights(self, config):
        """
        Initializes level weights, either as a fixed tensor or as a trainable parameter.
        First level is weighted with 1. Other weights can be set.
        """
        if hasattr(config, 'level_weights'):
            self.linear_combination = torch.tensor(config.level_weights).float()
        else:
            self.linear_combination = torch.arange(self.levels-1, 0, -1).float() / self.levels

        if len(self.linear_combination) != (self.levels-1):
            raise ValueError(f"level_weights {len(self.linear_combination)} must have the same length as levels-1 {self.levels-1}")

        self.trainable_level_weights = getattr(config, "trainable_level_weights", False)
        if self.trainable_level_weights:
            self.linear_combination = nn.Parameter(self.linear_combination)

    def validate_and_register_tree_matrix(self, tree, tree_matrix):
        if not isinstance(tree_matrix, torch.Tensor):
            if tree is None:
                raise ValueError("Either tree or tree_matrix must be provided")
            tree_matrix = tree.get_tree_matrix()
        self.register_tree_matrix(tree_matrix) 

    def register_tree_matrix(self, tree_matrix: torch.Tensor):
        """Register tree matrix as buffer and create a mask for each level."""
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











