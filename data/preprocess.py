import pandas as pd
import torch
import hydra

from tokenizer import EHRTokenizer
from dataset.EHR import EHRDataset
from utils.args import setup_preprocess
from data.utils import create_concept_features, create_age_features, create_abspos_features, create_segment_features, create_demographics, save_to_features

@hydra.main(config_name='data_config.yaml', config_path='.', version_base='1.3')
def process(cfg, args):
    features = {}
    # Get concept features (mandatory)
    concepts = create_concept_features(cfg.concepts)
    features['concept'] = []

    # Get age features (optional)
    if 'ages' in cfg:
        ages = create_age_features(cfg.ages, concepts)
        concepts['age'] = ages
        features['age'] = []

    # Get abspos features (optional)
    if 'abspos' in cfg:
        abspos = create_abspos_features(cfg.abspos, concepts)
        concepts['abspos'] = abspos
        features['abspos'] = []
    
    if 'segments' in cfg:
        segments = create_segment_features(cfg.segments, concepts)
        concepts['segment'] = segments
        features['segment']  = []

    if 'demographics' in cfg:
        demographics = create_demographics(cfg.demographics, ages='ages' in cfg, abspos='abspos' in cfg, segments='segments' in cfg)
        concepts = pd.concat((demographics, concepts))

    concepts = concepts.sort_values('TIMESTAMP')
    concepts = concepts[features.keys()]    # Keep only relevant features

    # Updates features dict inplace (mutable)
    concepts.groupby('Key.Patient', sort=False).apply(lambda patient: save_to_features(features, patient))

    with open(args.data_file, 'wb') as f:
        torch.save(features, f)

    return features


def tokenize_data(args):
    with open(args.data_file, 'rb') as f:
        features = torch.load(f)

    # Tokenize and save
    tokenizer = EHRTokenizer(args.vocabulary)
    outputs = tokenizer(features)

    tokenizer.save_vocab(args.vocabulary_file)
    with open(args.tokenized_file, 'wb') as f:
        torch.save(outputs, f)


def split_dataset(args):
    with open(args.tokenized_file, 'rb') as f:
        inputs = torch.load(f)

    dataset = EHRDataset(inputs)
    train_set, test_set = dataset.split(args.test_ratio)

    with open(args.train_file, 'wb') as f:
        torch.save(train_set, f)
    with open(args.test_file, 'wb') as f:
        torch.save(test_set, f)


if __name__ == '__main__':
    # Only needs to be run once
    args = setup_preprocess()
    process(args)
    tokenize_data(args)
    split_dataset(args)

