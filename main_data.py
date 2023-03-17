import torch
from hydra import initialize, compose
from data.concept_loader import ConceptLoader
from data_fixes.infer import Inferrer
from data.featuremaker import FeatureMaker
from data_fixes.handle import Handler
from data.tokenizer import EHRTokenizer
from data.dataset import MLMDataset
from data.split import Splitter

def main():
    with initialize(config_path='configs'):
        cfg = compose(config_name='featuremaker.yaml')
    """
        Loads
        Infers nans
        Creates features
        Overwrite nans
        Tokenize
        To dataset
    """
    # Load concepts
    concepts, patients_info = ConceptLoader()(cfg.data_dir)

    # Infer missing values
    concepts = Inferrer()(concepts)

    # Create feature sequences
    features = FeatureMaker(cfg)(concepts, patients_info)

    # Overwrite nans and incorrect values
    features = Handler()(features)

    # Split
    train, test, val = Splitter()(features)

    # Tokenize
    tokenizer = EHRTokenizer({'sep_tokens': True, 'cls_token': True})
    encoded_train = tokenizer(train, padding=False)
    tokenizer.lock_vocabulary()
    encoded_test = tokenizer(test, padding=False)
    encoded_val = tokenizer(val, padding=False)

    # To dataset
    train_dataset = MLMDataset(encoded_train, vocabulary=tokenizer.vocabulary)
    test_dataset = MLMDataset(encoded_test, vocabulary=tokenizer.vocabulary)
    val_dataset = MLMDataset(encoded_val, vocabulary=tokenizer.vocabulary)
    torch.save(train_dataset, 'dataset.train')
    torch.save(test_dataset, 'dataset.test')
    torch.save(val_dataset, 'dataset.val')
    

if __name__ == '__main__':
    main()

