import argparse


def setup_preprocess():
    parser = argparse.ArgumentParser(description='Preprocessing of EHR data')

    # For preprocessing
    parser.add_argument('-f', '--data_file', type=str, default="data.pt")

    # For tokenization
    parser.add_argument('-v', '--vocabulary', type=str, default=None)
    parser.add_argument('--vocabulary_file', type=str, default="vocabulary.pt")
    parser.add_argument('--tokenized_file', type=str, default="tokenized_output.pt")

    # For splitting
    parser.add_argument('-r', '--test_ratio', type=float, default=0.2)
    parser.add_argument('--train_file', type=str, default="dataset.train")
    parser.add_argument('--test_file', type=str, default="dataset.test")

    args = parser.parse_args()

    return args


def setup_training():
    parser = argparse.ArgumentParser(description='Training script for EHR data')

    # General
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('-mt', '--max_tokens', type=int, default=768)
    parser.add_argument('-ga', '--grad_accumulation', type=int, default=1)

    parser.add_argument('-v', '--vocabulary', type=str, default="vocabulary.pt")
    parser.add_argument('--train_set', type=str, default="dataset.train")
    parser.add_argument('--test_set', type=str, default="dataset.test")

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.01)

    parser.add_argument('-mr', '--masked_ratio', type=float, default=0.3)

    parser.add_argument('-name', '--experiment_name', type=str, default='unnamed_experiment')

    args = parser.parse_args()

    return args

