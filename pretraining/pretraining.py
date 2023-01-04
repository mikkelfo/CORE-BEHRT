import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import BertConfig
from torch.utils.data import DataLoader

from model.model import BertEHRModel
from utils.utils import load_features, load_vocabulary, to_device
from utils.args import setup_training

# Specific for computerome
import matplotlib
matplotlib.use('TkAgg')


def MLM_pretraining(args):
    train_set, test_set = load_features(train_file=args.train_set, test_file=args.test_set)
    vocabulary = load_vocabulary(args.vocabulary)

    # Find max segments
    max_segments = max(train_set.get_max_segments(), test_set.get_max_segments()) + 1

    config = BertConfig(
        vocab_size=len(vocabulary),              
        max_position_embeddings=args.max_tokens,
        type_vocab_size=max_segments
    )

    model = BertEHRModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Prepare datasets for training
    train_set.setup_mlm(args)
    test_set.setup_mlm(args)

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    # Create seperate folder for output
    os.mkdir(args.experiment_name)

    torch.save(args, f'{args.experiment_name}/args.pt')
    torch.save(config, f'{args.experiment_name}/config.pt')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    all_train_loss = []
    all_test_loss = []
    for epoch in range(args.epochs):

        # Training
        model.train()
        train_loss = 0
        for i, batch in tqdm(enumerate(train_dataloader)):
            events, attention_mask, masked_events, target = batch
            to_device(*batch, device=device)

            masked_concepts, age, abspos, segments = masked_events.permute((2, 0, 1))
            output = model(input_ids=masked_concepts, attention_mask=attention_mask, labels=target)

            train_loss += output.loss.item()
            if (i+1) % args.grad_accumulation == 0:
                print(f'  Epoch {epoch}.{i}: {output.loss.item()}')
                output.loss.backward()
                optimizer.step()

        all_train_loss.append(train_loss / len(train_dataloader))
        print(f'Train loss {epoch}: {train_loss / len(train_dataloader)}')
        print()

        # Testing
        model.eval()
        test_loss = 0
        for batch in test_dataloader:
            events, attention_mask, masked_events, target = batch
            to_device(*batch, device=device)

            masked_concepts, age, abspos, segments = masked_events.permute((2, 0, 1))
            test_loss += model(input_ids=masked_concepts, attention_mask=attention_mask, labels=target).loss.item()

        all_test_loss.append(test_loss / len(test_dataloader))
        print(f'Test loss {epoch}:', test_loss / len(test_dataloader))
        
        torch.save(model.state_dict(), f'{args.experiment_name}/model_{epoch}.pt')

    print(all_train_loss)
    print(all_test_loss)
    plt.plot(all_train_loss)
    plt.savefig(f'{args.experiment_name}/train_loss.png')
    plt.show()

    plt.plot(all_test_loss)
    plt.savefig(f'{args.experiment_name}/test_loss.png')
    plt.show()


if __name__ == '__main__':
    args = setup_training()
    MLM_pretraining(args)
