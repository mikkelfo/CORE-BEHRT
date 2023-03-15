from transformers import BertForMaskedLM
from embeddings.ehr import EhrEmbeddings


class BertEHRModel(BertForMaskedLM):
    def __init__(self, config):
        super(BertEHRModel, self).__init__(config)
        self.config = config

        self.bert.embeddings = EhrEmbeddings(config)

