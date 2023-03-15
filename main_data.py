import torch
from hydra import initialize, compose
from data.concept_loader import ConceptLoader
from data_fixes.infer import Inferrer
from data.featuremaker import FeatureMaker
from data_fixes.overwrite import Overwriter
from data.tokenizer import EHRTokenizer
from data.dataset import MLMDataset

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
    features = Overwriter()(features)

    # Tokenize
    tokenizer = EHRTokenizer({'sep_tokens': True, 'cls_token': True})
    encoded = tokenizer(features, padding=False)

    # To dataset
    dataset = MLMDataset(encoded, vocabulary=tokenizer.vocabulary)
    train, val, test = dataset.split()
    torch.save(train, 'dataset.train')
    torch.save(val, 'dataset.val')
    torch.save(test, 'dataset.test')
    

if __name__ == '__main__':
    main()

