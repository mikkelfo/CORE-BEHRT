import torch
from transformers import BertConfig
from torch.utils.data import DataLoader

from data.preprocess import split_testsets
from dataset.MLM import MLMDataset
from model.model import BertEHRModel


def MLM_pretraining():
    train, test = split_testsets()
    train_codes, train_segments = train
    test_codes, test_segments = test

    with open('vocabulary.pt', 'rb') as f:
        vocabulary = torch.load(f)

    # Find max segments
    max_segments = max(max([max(segment) for segment in train_segments]), max([max(segment) for segment in test_segments]))

    config = BertConfig(
        vocab_size=len(vocabulary),              
        max_position_embeddings=512,    # Change?
        type_vocab_size=max_segments    # Should be smarter
    )

    model = BertEHRModel(config)
    

    train_dataset = MLMDataset(train_codes, train_segments, vocabulary)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)


    for batch in train_dataloader:
        codes, segments, masked = batch
        output = model(input_ids=masked, token_type_ids=segments)
