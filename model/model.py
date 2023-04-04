from transformers import BertForMaskedLM
from embeddings.ehr import EhrEmbeddings
import torch.nn as nn


class BertEHRModel(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.bert.embeddings = EhrEmbeddings(config)

    def add_new_binary_head(self):
        self.cls = nn.Linear(self.config.hidden_size, self.config.num_labels)

