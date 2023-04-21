import torch
import hydra
from data.concept_loader import ConceptLoader
from data_fixes.infer import Inferrer
from data.featuremaker import FeatureMaker
from data_fixes.handle import Handler
from data_fixes.exclude import Excluder
from data.tokenizer import EHRTokenizer
from data.split import Splitter
from downstream_tasks.outcomes import OutcomeMaker

@hydra.main(version_base=None, config_path="configs/data", config_name="data")
def main_data(cfg):
    """
        Loads data
        Infers nans
        Finds outcomes
        Creates features
        Handles wrong data
        Excludes patients with <k concepts
        Splits data
        Tokenizes
        Saves
    """
    # Load concepts
    concepts, patients_info = ConceptLoader()(**cfg.loader)

    # Infer missing values
    concepts = Inferrer()(concepts)

    # Make outcomes
    patients_info = OutcomeMaker(cfg)(concepts, patients_info)

    # Create feature sequences and outcomes
    features, outcomes = FeatureMaker(cfg)(concepts, patients_info)
    
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
    tokenizer = EHRTokenizer(cfg.tokenizer)
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

