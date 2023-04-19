import torch
from hydra import initialize, compose
from data.concept_loader import ConceptLoader
from data_fixes.infer import Inferrer
from data.featuremaker import FeatureMaker
from data_fixes.handle import Handler
from data_fixes.exclude import Excluder
from data.tokenizer import EHRTokenizer
from data.split import Splitter
from downstream_tasks.outcomes import OutcomeMaker

def main_data():
    with initialize(config_path='configs'):
        data_config = compose(config_name='data.yaml')
    """
        Loads
        Infers nans
        Creates features
        Overwrite nans
        Tokenize
        To dataset
    """
    # Load concepts
    concepts, patients_info = ConceptLoader()(**data_config.loader)

    # Infer missing values
    concepts = Inferrer()(concepts)

    # Make outcomes
    patients_info = OutcomeMaker(data_config.outcomes)(concepts, patients_info)

    # Create feature sequences and outcomes
    features, outcomes = FeatureMaker(data_config)(concepts, patients_info)
    
    # Overwrite nans and other incorrect values
    features = Handler()(features)

    # Exclude patients with <k concepts
    features, outcomes = Excluder()(features, outcomes)

    # Save final features and outcomes
    torch.save(features, 'features.pt')
    torch.save(outcomes, 'outcomes.pt')

    # Split
    train_features, test_features, val_features = Splitter()(features)
    train_outcomes, test_outcomes, val_outcomes = Splitter()(outcomes)

    # Tokenize
    tokenizer = EHRTokenizer(data_config.tokenizer)
    train_encoded = tokenizer(train_features)
    tokenizer.freeze_vocabulary()
    test_encoded = tokenizer(test_features)
    val_encoded = tokenizer(val_features)

    # Save features and outcomes
    torch.save(train_encoded, 'train_encoded.pt')
    torch.save(train_outcomes, 'train_outcomes.pt')
    torch.save(test_encoded, 'test_encoded.pt')
    torch.save(test_outcomes, 'test_outcomes.pt')
    torch.save(val_encoded, 'val_encoded.pt')
    torch.save(val_outcomes, 'val_outcomes.pt')
    

if __name__ == '__main__':
    main_data()

