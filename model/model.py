from transformers import BertConfig, BertForMaskedLM
from embeddings.ehr import EhrEmbeddings


class BertEHRModel(BertForMaskedLM):
    def __init__(self, config):
        super(BertEHRModel, self).__init__(config)
        self.config = config

        self.bert.embeddings = EhrEmbeddings(config)


if __name__ == '__main__':
    config = BertConfig(
        # vocab_size=None,              
        # max_position_embeddings=None, 
        # type_vocab_size=None
    )
    model = BertEHRModel(config)