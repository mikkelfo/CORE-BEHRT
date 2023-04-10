import torch
from hydra import initialize, compose
from data.tokenizer import H_EHRTokenizer
from data.split import Splitter
from data.dataset import H_MLMDataset
from data.concept_loader import ConceptLoader
from data_fixes.infer import Inferrer
from data.featuremaker import FeatureMaker
from data_fixes.handle import Handler

def main():
    with initialize(config_path='configs'):
        features_config = compose(config_name='featuremaker.yaml')
        tokenizer_config = compose(config_name='tokenizer.yaml')

    # Load concepts
    # concepts, patients_info = ConceptLoader()(features_config.data_dir)

    # Infer missing values
    # concepts = Inferrer()(concepts)

    # Create feature sequences
    # features = FeatureMaker(features_config)(concepts, patients_info)

    # Overwrite nans and incorrect values
    # features = Handler()(features)
    # torch.save(features, 'features.pt')

    print("Load sequences")
    features = torch.load("C:\\Users\\fjn197\\PhD\\projects\\PHAIR\\pipelines\\patbert\\data\\sequence\\synthetic\\synthetic.pt")
    print("Split")


    train, test, val = Splitter()(features)
    print("Tokenize")
    tokenizer = H_EHRTokenizer(config=tokenizer_config)
    encoded_train = tokenizer(train, padding=tokenizer_config.padding, truncation=tokenizer_config.truncation)
    print("Save vocab")
    tokenizer.save_vocab('data/tokenized/hierarchical/test')
    tokenizer.freeze_vocabulary() 
    print("Tokenize test and val")
    # finds closest ancestor in the hierarchy for unknown tokens
    encoded_test = tokenizer(test, padding=tokenizer_config.padding, truncation=tokenizer_config.truncation)
    encoded_val = tokenizer(val, padding=tokenizer_config.padding, truncation=tokenizer_config.truncation)

    print("Create datasets")
    # To dataset
    train_dataset = H_MLMDataset(encoded_train, vocabulary=tokenizer.vocabulary)
    test_dataset = H_MLMDataset(encoded_test, vocabulary=tokenizer.vocabulary)
    val_dataset = H_MLMDataset(encoded_val, vocabulary=tokenizer.vocabulary)

    print("Save datasets")
    torch.save(train_dataset, 'data/tokenized/hierarchical/test/dataset.train')
    torch.save(test_dataset, 'data/tokenized/hierarchical/test/dataset.test')
    torch.save(val_dataset, 'data/tokenized/hierarchical/test/dataset.val')

if __name__ == '__main__':
    main()