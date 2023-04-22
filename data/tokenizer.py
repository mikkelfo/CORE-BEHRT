from os.path import join
from typing import Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BatchEncoding

from data import medical


class EHRTokenizer():
    def __init__(self, config, vocabulary=None):
        self.config = config
        if vocabulary is None:
            self.new_vocab = True
            self.vocabulary = {
                '[PAD]': 0,
                '[CLS]': 1, 
                '[SEP]': 2,
                '[UNK]': 3,
                '[MASK]': 4,
            }
        else:
            self.new_vocab = False
            self.vocabulary = vocabulary

    def __call__(self, features: dict, padding=True, truncation=512):
        return self.batch_encode(features, padding, truncation)

    def batch_encode(self, features: dict, padding=True, truncation=512):
        data = {key: [] for key in features}
        data['attention_mask'] = []

        for patient in self._patient_iterator(features):
            patient = self.insert_special_tokens(patient)                   # Insert SEP and CLS tokens

            if truncation and len(patient['concept']) > truncation:
                patient = self.truncate(patient, max_len=truncation)        # Truncate patient to max_len
            
            # Created after truncation for efficiency
            patient['attention_mask'] = [1] * len(patient['concept'])       # Initialize attention mask

            patient['concept'] = self.encode(patient['concept'])            # Encode concepts

            for key, value in patient.items():
                data[key].append(value)

        if padding:
            longest_seq = max([len(s) for s in data['concept']])            # Find longest sequence
            data = self.pad(data, max_len=longest_seq)                      # Pad sequences to max_len

        
        return BatchEncoding(data, tensor_type='pt' if padding else None)

    def encode(self, concepts: list):
        if self.new_vocab:
            for concept in concepts:
                if concept not in self.vocabulary:
                    self.vocabulary[concept] = len(self.vocabulary)

        return [self.vocabulary.get(concept, self.vocabulary['[UNK]']) for concept in concepts]

    def truncate(self, patient: dict, max_len: int):
        # Find length of background sentence (+2 to include CLS token and SEP token)
        additional_tokens = int(self.config['cls_token']) + int(self.config['sep_tokens'])
        background_length = len([x for x in patient.get('concept', []) if x[:3] == "BG_"]) + additional_tokens
        truncation_length = max_len - background_length
        
        # Do not start seq with SEP token (SEP token is included in background sentence)
        if patient['concept'][-truncation_length] == '[SEP]':
            truncation_length -= 1

        for key, value in patient.items():
            patient[key] = value[:background_length] + value[-truncation_length:]    # Keep background sentence + newest information

        return patient

    def pad(self, features: dict,  max_len: int):
        padded_data = {key: [] for key in features}
        for patient in self._patient_iterator(features):
            difference = max_len - len(patient['concept'])

            for key, values in patient.items():
                token = self.vocabulary['[PAD]'] if key == 'concept' else 0
                padded_data[key].append(values + [token] * difference)

        return padded_data

    def insert_special_tokens(self, patient: dict):
        if self.config['sep_tokens']:
            if 'segment' not in patient:
                raise Exception('Cannot insert [SEP] tokens without segment information')
            patient = self.insert_sep_tokens(patient)

        if self.config['cls_token']:
            patient = self.insert_cls_token(patient)
        
        return patient

    def insert_sep_tokens(self, patient: dict):
        padded_segment = patient['segment'] + [None]                # Add None to last entry to avoid index out of range

        for key, values in patient.items():
            new_seq = []
            for i, val in enumerate(values):
                new_seq.append(val)

                if padded_segment[i] != padded_segment[i+1]:
                    token = '[SEP]' if key == 'concept' else val
                    new_seq.append(token)

            patient[key] = new_seq

        return patient

    def insert_cls_token(self, patient: dict):
        for key, values in patient.items():
            token = '[CLS]' if key == 'concept' else 0          # Determine token value (CLS for concepts, 0 for rest)
            patient[key] = [token] + values
        return patient

    def save_vocab(self, dest: str):
        torch.save(self.vocabulary, dest)

    def _patient_iterator(self, features: dict):
        for i in range(len(features['concept'])):
            yield {key: values[i] for key, values in features.items()}

    def freeze_vocabulary(self):
        self.new_vocab = False
        self.save_vocab('vocabulary.pt')



class H_EHRTokenizer(EHRTokenizer):
    """
    In addition to integer encoding, also encode the hierarchy of the concepts as tuples
    and return hierarchical vocabulary. Also constructs a dataframe with the SKS names and coresponding tuples.
    """
    def __init__(self, config:Dict, vocabulary:Dict=None, code_types=['D','M'], full_sks_vocab_ls:List[Dict]=None, names_df:pd.DataFrame=None, tuples_df:pd.DataFrame=None, test=False):
        super(H_EHRTokenizer, self).__init__(config, vocabulary)
        self.h_vocabulary = {}
        if isinstance(full_sks_vocab_ls, type(None)) or isinstance(names_df, type(None)) or isinstance(tuples_df, type(None)):
            constructor = medical.TableConstructor(main_vocab=self.vocabulary, code_types=code_types, test=test)
            self.full_sks_vocab_ls, self.df_sks_names, self.df_sks_tuples = constructor()
        else:
            self.full_sks_vocab_ls = full_sks_vocab_ls
            self.df_sks_names = names_df
            self.df_sks_tuples = tuples_df
    def __call__(self, features: Dict[str,List[List]], padding:bool=True, truncation:int=512):
        data = self.batch_encode(features, padding, truncation)
        self.inv_vocab = {v: k for k, v in self.vocabulary.items()}
        data['target'] = self.batch_encode_hierarchy(data)
        return data

    def batch_encode_hierarchy(self, data:Dict[str, List[List[int]]])->List[List[tuple]]:
        """Encode the hierarchy of the concepts as tuple"""
        targets = [] # list of lists of tuples
        for patient_concept_tok in tqdm(data['concept'], 'h_encode patients'): # these are encoded as integers
            # we can skip the padding and truncation here, because we already did it in the batch_encode function
            pat_h_enc_concepts = self.h_encode_patient(patient_concept_tok)
            targets.append(pat_h_enc_concepts)
        return targets

    def h_encode_patient(self, patient_concept_tok: List[int])->List[tuple]:
        """Encode concepts hierarchically as tuples for one patient"""
        pat_targets = [] # list of tuples
        for concept_tok in patient_concept_tok:
            concept = self.inv_vocab[int(concept_tok)] # instead we map the integers to the corresponding concepts
            # We don't need to take care of freezing the vocabulary here, because we already did it in the batch_encode function
            # all concepts that show up here are in the main vocabulary
            # TODO: think whether we should try partial matching here e.g. D10.011 not in vocabulary, but D10.01 is
            if concept not in self.h_vocabulary:
                if concept not in self.df_sks_names.stack().unique(): # check if it's in the database
                    print(concept, 'not in the database')
                    self.add_unknown_concept_to_hierarchy(concept)
                self.h_vocabulary[concept] = self.get_lowest_level_node(concept)
            pat_targets.append(self.h_vocabulary[concept])
            
        return pat_targets

    def encode(self, concepts: list):
        """Overwrite the encode function of base class to add the concept to the vocabulary if it's not there yet"""
        if self.new_vocab:
            for concept in concepts:
                if concept not in self.vocabulary:
                    self.vocabulary[concept] = len(self.vocabulary)

        return [self.vocabulary.get(concept, self.check_for_ancestor(concept)) for concept in concepts]

    def check_for_ancestor(self, concept):
        """Check if the concept has an ancestor in the vocabulary
        If yes, return the tuple of the closest ancestor"""
        if concept not in self.df_sks_names.stack().unique():
            return self.vocabulary['[UNK]']
        
        index = self.df_sks_names.where(self.df_sks_names == concept).stack().index
        branch = index[-1][0]
        level = index[-1][1]
        while (concept not in self.vocabulary) and (level>0): # go up the hierarchy until you find a concept in the vocabulary
            level -= 1 # go up the hierarchy
            concept = self.df_sks_names.iloc[branch, level] # get parent
        return self.vocabulary.get(concept, self.vocabulary['[UNK]'])

    def add_unknown_concept_to_hierarchy(self, concept:str)->None:
        """Add a new concept to the hierarchy."""
        max_node_lvl0 = self.df_sks_tuples.iloc[:,0].max() # at the first level, get maximum integer value (num root nodes) 
        new_max_node_lvl0 = (max_node_lvl0[0]+1,) + max_node_lvl0[1:] # increment by one (new root node)
        new_hierarchy_row = self.extend_node_to_lower_levels(new_max_node_lvl0) # extend to bottom level e.g. (n, 0 ,0) -> [(n, 0, 0), (n, 1, 0), (n, 1, 1)]

        self.df_sks_tuples.loc[len(self.df_sks_tuples)] = new_hierarchy_row # add the list of nodes at the end of the df
        self.df_sks_names.loc[len(self.df_sks_names)] = [concept] * len(new_hierarchy_row) # add at the end of the df
        for node, sks_voc in zip(new_hierarchy_row, self.full_sks_vocab_ls):
            sks_voc[concept] = node

    def get_lowest_level_node(self, name:str)->Tuple[int]:
        """Get the lowest level tuple of a concept
        E.g. the [SEP] token exists on all levels, but we want to get the lowest level tuple, so we would get (i, 1,1,1,..,1)"""
        indices = self.df_sks_names[self.df_sks_names==name].stack().index
        max_id = indices.max()
        return self.df_sks_tuples.iloc[max_id[0], max_id[1]]
 
    @staticmethod
    def get_parent(node:Tuple[int])->Tuple[int]:
        """Get parent node of a node defined by a tuple."""
        idx = next((i for i, x in enumerate(node) if x == 0), None)
        if idx is not None:
            return node[:idx - 1] + (0,) * (len(node) - idx + 1)
        else:
            return node[:-1] + (0,)
    
    @staticmethod
    def extend_node_to_lower_levels(node:Tuple)->List[Tuple]: # we need to extend to both levels below
        """Given a tuple, extend it to the lowest level, by replicating and adding 1s to the end of the tuple"""
        row_ls = []
        row_ls.append(node)
        for i in range(node.__len__()-1):
            new_node = node[:i+1] + (1,) + node[i+2:]
            node = new_node
            row_ls.append(new_node)
        return row_ls
    
    def save_vocab(self, dest: str):
        torch.save(self.vocabulary, join(dest, "vocabulary.pt"))
        torch.save(self.h_vocabulary, join(dest, "h_vocabulary.pt"))
        torch.save(self.get_leaf_nodes(), join(dest, "leaf_nodes.pt"))
        self.df_sks_names.to_csv(join(dest, "sks_names.csv"), index=None)
        self.df_sks_tuples.to_csv(join(dest, "sks_tuples.csv"), index=None)


    def freeze_vocabulary(self):
        self.new_vocab = False
    
    def get_leaf_nodes(self):
        """Get all leaf nodes of the hierarchy"""
        return torch.tensor(self.df_sks_tuples.iloc[:,-1].unique().tolist(), dtype=torch.long)



