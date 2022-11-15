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

    # Prepare datasets for training
    train_set.setup_mlm(vocabulary)
    test_set.setup_mlm(vocabulary)

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size)

    torch.save(args, 'args.pt')

    all_train_loss = []
    all_test_loss = []
    for epoch in range(args.epochs):

        # Training
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            (codes, segments, attention_mask), (masked_seq, target) = batch
            output = model(input_ids=masked_seq, attention_mask=attention_mask, token_type_ids=segments, labels=target)

            train_loss += output.loss.item()
            output.loss.backward()
            optimizer.step()

        all_train_loss.append(train_loss / len(train_dataloader))
        print(f'Train loss {epoch}: {train_loss / len(train_dataloader)}')

        # Testing
        model.eval()
        test_loss = 0
        for batch in test_dataloader:
            (codes, segments, attention_mask), (masked_seq, target) = batch
            test_loss += model(input_ids=masked_seq, attention_mask=attention_mask, token_type_ids=segments, labels=target).loss.item()

        all_test_loss.append(test_loss / len(test_dataloader))
        print(f'Test loss {epoch}:', test_loss / len(test_dataloader))
        
        torch.save(model.state_dict(), f'model_{epoch}.pt')

    print(all_train_loss)
    print(all_test_loss)
    plt.plot(all_train_loss)
    plt.savefig('train_loss.png')
    plt.show()

    plt.plot(all_test_loss)
    plt.savefig('test_loss.png')
    plt.show()


if __name__ == '__main__':
    args = setup_training()
    MLM_pretraining(args)
