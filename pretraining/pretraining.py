import torch
from transformers import BertConfig

from torch.utils.data import DataLoader
from model.model import BertEHRModel
from utils.utils import load_features, load_vocabulary
from utils.args import setup_training

import matplotlib.pyplot as plt


def MLM_pretraining(args):
    train_set, test_set = load_features(train_file=args.train_set, test_file=args.test_set)
    vocabulary = load_vocabulary(args.vocabulary)

    # Find max segments
    max_segments = max(train_set.get_max_segments(), test_set.get_max_segments())

    config = BertConfig(
        vocab_size=len(vocabulary),              
        max_position_embeddings=args.max_tokens,
        type_vocab_size=max_segments
    )

    model = BertEHRModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Prepare train_set for training
    train_set.setup_mlm(vocabulary)

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size)

    torch.save(args, 'args.pt')

    all_loss = []
    for epoch in range(args.epochs):
        epoch_loss = 0
        for idx, batch in enumerate(train_dataloader):
            (codes, segments, attention_mask), (masked_seq, target) = batch
            output = model(input_ids=masked_seq, attention_mask=attention_mask, token_type_ids=segments, labels=target)

            epoch_loss += output.loss.item()

            output.loss.backward()
            optimizer.step()
            
        optimizer.step()
        all_loss.append(epoch_loss / len(train_dataloader))
        print(all_loss)
        
        torch.save(model.state_dict(), f'model_{epoch}.pt')
    plt.plot(all_loss)
    plt.savefig('loss.png')
    plt.show()


if __name__ == '__main__':
    args = setup_training()
    MLM_pretraining(args)
