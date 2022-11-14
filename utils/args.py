import argparse


def setup_preprocess():
    parser = argparse.ArgumentParser(description='Preprocessing of EHR data')

    parser.add_argument('-f', '--data_file', type=str)
    parser.add_argument('-v', '--vocabulary', type=str, default=None)

    parser.add_argument('-i', '--identifiers', type=str, default=['Key', 'Code', 'NOTED_DATE'], nargs=3, help="Identifiers order by 'Key', 'Code', 'Time stamp'")
    
    parser.add_argument('-v', '--vocabulary_file', type=str, default="vocabulary.pt")
    parser.add_argument('--tokenized_file', type=str, default="tokenized_output.pt")

    parser.add_argument('-r', '--test_ratio', type=float, default=0.2)
    parser.add_argument('--train_file', type=str, default="dataset.train")
    parser.add_argument('--test_file', type=str, default="dataset.test")

    args = parser.parse_args()

    return args


def setup_training():
    parser = argparse.ArgumentParser(description='Training script for EHR data')

    # General
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=256)
    parser.add_argument('-mt', '--max_tokens', type=int, default=768)

    parser.add_argument('-v', '--vocabulary', type=str, default="vocabulary.pt")
    parser.add_argument('--train_set', type=str, default="dataset.train")
    parser.add_argument('--test_set', type=str, default="dataset.test")

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.01)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args1 = setup_preprocess()
    args2 = setup_training()
    print(args1)
    print(args2)