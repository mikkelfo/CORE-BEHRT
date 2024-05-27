import torch.nn as nn
from transformers import BertModel
from transformers.models.roformer.modeling_roformer import RoFormerEncoder

from corebehrt.embeddings.ehr import EhrEmbeddings
from corebehrt.model.activations import SwiGLU
from corebehrt.model.heads import FineTuneHead, MLMHead


class BertEHREncoder(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = EhrEmbeddings(config)
        
        # Activate transformer++ recipe
        config.rotary_value = False
        self.encoder = RoFormerEncoder(config)
        for layer in self.encoder.layer:
            layer.intermediate.intermediate_act_fn = SwiGLU(config)

    def forward(self, batch: dict):
        position_ids = {key: batch[key] for key in ['age', 'abspos']}
        outputs = super().forward(
            input_ids=batch['concept'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['segment'],
            position_ids=position_ids,
        )
        return outputs

class BertEHRModel(BertEHREncoder):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.cls = MLMHead(config)
            
    def forward(self, batch: dict):
        outputs = super().forward(batch)

        sequence_output = outputs[0]    # Last hidden state
        logits = self.cls(sequence_output, batch['attention_mask'])
        outputs.logits = logits

        if batch.get('target') is not None:
            outputs.loss = self.get_loss(logits, batch['target'])

        return outputs

    def get_loss(self, logits, labels):
        """Calculate loss for masked language model."""
        return self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))


class BertForFineTuning(BertEHRModel):
    def __init__(self, config):
        super().__init__(config)

        self.loss_fct = nn.BCEWithLogitsLoss()
        self.cls = FineTuneHead(config)
        
    def get_loss(self, hidden_states, labels):    
        return self.loss_fct(hidden_states.view(-1), labels.view(-1))

