import torch
from hydra import initialize, compose
from data.tokenizer import H_EHRTokenizer
from data.split import Splitter
from data.dataset import H_MLMDataset
from data.medical import Tree
from data.concept_loader import ConceptLoader
from data_fixes.infer import Inferrer
from data.featuremaker import FeatureMaker
from data_fixes.handle import Handler
from tqdm import tqdm
from os.path import join
import typer

def main(test: bool = typer.Option(False)):
    with initialize(config_path='configs'):
        features_config = compose(config_name='featuremaker.yaml')
        tokenizer_config = compose(config_name='tokenizer.yaml')
        dataset_config = compose(config_name='dataset.yaml')
    data_dir = "data\\features\\hierarchical\\test"
    if not test:
        # Load concepts
        concepts, patients_info = ConceptLoader()(features_config.data_dir)
        # Infer missing values
        concepts = Inferrer()(concepts)

        # Create feature sequences
        features = FeatureMaker(features_config)(concepts, patients_info)

        # Overwrite nans and incorrect values
        features = Handler()(features)
        torch.save(features, 'features.pt')
    else:
        print("Load sequences")
        features = torch.load("C:\\Users\\fjn197\\PhD\\projects\\PHAIR\\pipelines\\ehr2vec\\data\\sequence\\synthetic\\synthetic.pt")
    
    print("Split")
    train, test, val = Splitter()(features)
    print("Tokenize")
    tokenizer = H_EHRTokenizer(config=tokenizer_config, test=test)
    encoded_train = tokenizer(train, padding=tokenizer_config.padding, truncation=tokenizer_config.truncation)
    
    tokenizer.freeze_vocabulary() 
    print("Tokenize test and val")
    # finds closest ancestor in the hierarchy for unknown tokens
    encoded_test = tokenizer(test, padding=tokenizer_config.padding, truncation=tokenizer_config.truncation)
    encoded_val = tokenizer(val, padding=tokenizer_config.padding, truncation=tokenizer_config.truncation)
    print("Save vocab")
    tokenizer.save_vocab(data_dir)

    print("Populate tree")
    tree = Tree(tokenizer.df_sks_names)
    tree(train['concept'])

    print("Create datasets")
    # To dataset
    dataset_kwargs = {'vocabulary': tokenizer.vocabulary, 'h_vocabulary': tokenizer.h_vocabulary,
            'leaf_nodes':tokenizer.get_leaf_nodes(), 'base_leaf_probs':tree.leaf_probabilities}
    dataset_kwargs.update(dataset_config)
    train_dataset = H_MLMDataset(encoded_train, **dataset_kwargs)
    test_dataset = H_MLMDataset(encoded_test, **dataset_kwargs)
    val_dataset = H_MLMDataset(encoded_val, **dataset_kwargs)
    
    print("Save datasets")
    torch.save(train_dataset, join(data_dir,'dataset.train'))
    torch.save(test_dataset, join(data_dir,'dataset.test'))
    torch.save(val_dataset, join(data_dir,'dataset.val'))

if __name__ == '__main__':
    main()