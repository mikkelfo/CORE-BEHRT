import logging

import torch
import torch.nn as nn
from transformers import BertModel
from transformers.models.roformer.modeling_roformer import RoFormerEncoder

from ehr2vec.embeddings.ehr import EhrEmbeddings
from ehr2vec.model.activations import SwiGLU
from ehr2vec.model.heads import FineTuneHead, MLMHead


logger = logging.getLogger(__name__)
class BertEHREncoder(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = EhrEmbeddings(config)
        
        # Activate transformer++ recipe
        if config.to_dict().get('plusplus'):
            logger.info("Using Transformer++ recipe.")
            config.rotary_value = False
            self.encoder = RoFormerEncoder(config)

            for layer in self.encoder.layer:
                layer.intermediate.intermediate_act_fn = SwiGLU(config)
                # layer.output.LayerNorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps) # We dont use RMSNorm (only speedup, no performance gain)

    def forward(self, batch: dict):
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
        self.forward = self.forward_mlm
            
    def forward_mlm(self, batch: dict):
        outputs = super().forward(batch)

        sequence_output = outputs[0]    # Last hidden state
        logits = self.cls(sequence_output, batch['attention_mask'])
        outputs.logits = logits

        if batch.get('target', None) is not None:
            outputs.loss = self.get_loss(logits, batch['target'])
            outputs.mlm_loss = outputs.loss.detach().item()

        return outputs

    def get_loss(self, logits, labels):
        """Calculate loss for masked language model."""
        return self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

    def get_classification_loss(self, logits, labels):
        """Calculate loss for prolonged length of stay classification if applicable."""
        return self.plos_fct(logits.view(-1), labels.view(-1))

class BertForFineTuning(BertEHRModel):
    def __init__(self, config):
        super().__init__(config)
        if config.pos_weight:
            pos_weight = torch.tensor(config.pos_weight)
        else:
            pos_weight = None

        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.cls = FineTuneHead(config)
        logger.info(f"Using {self.cls.__class__.__name__} as classifier.")
        
    def get_loss(self, hidden_states, labels, labels_mask=None):    
        return self.loss_fct(hidden_states.view(-1), labels.view(-1))

