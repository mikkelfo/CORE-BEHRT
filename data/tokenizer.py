import torch
from transformers import BatchEncoding
from os.path import join
from data.medical import SKSVocabConstructor
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import random

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
        background_length = len([x for x in patient.get('concept', []) if x[:3] == "BG_"]) + 2
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
    """In addition to integer encoding, also encode the hierarchy of the concepts as tuple
    and return hierarchical vocabulary. In addition, constructs a dataframe with the SKS names and tuples.
    """
    def __init__(self, config, vocabulary=None, test=False, data=None):
        super().__init__(config, vocabulary)
        self.h_vocabulary = {}
        _, sks_vocab_tup = SKSVocabConstructor(main_vocab=self.vocabulary)() 
        if test:
            rand_keys = random.sample(sorted(sks_vocab_tup), 10)
            data_flat = [item for sublist in data['concept'] for item in sublist]
            # sks_vocab_tup = {k: sks_vocab_tup[k] for k in data_flat if k in sks_vocab_tup}
            # sks_vocab_tup = {k: sks_vocab_tup[k] for k in rand_keys}
            sks_vocab_tup = {'A':(1,0,0), 'B':(2,0,0), 'Aa':(1,1,0), 'Ab':(1,2,0), 'Ba':(2,1,0), 'Bb':(2,2,0), 'Ca':(3,1,0)}
        
        self.extended_sks_vocab_ls = self.extend_leafs(sks_vocab_tup) # extend leaf nodes to bottom level
        self.full_sks_vocab_ls = self.fill_parents(self.extended_sks_vocab_ls) # fill parents to top level
        self.df_sks_names, self.df_sks_tuples = self.construct_h_table_from_dics(self.full_sks_vocab_ls) # full table of the SKS vocab tree

    def __call__(self, features: dict, padding=True, truncation=512):
        data = self.batch_encode(features, padding, truncation)
        data['h_concept'] = self.batch_encode_hierarchy(data)
        return data

    def batch_encode_hierarchy(self, data:List[List[str]])->List[List[tuple]]:
        """Encode the hierarchy of the concepts as tuple"""
        h_concepts = [] # list of lists of tuples
        for patient_concept_ls in data['concept']:
            h_concepts.append(self.h_encode_patient(patient_concept_ls))
        return h_concepts

    def h_encode_patient(self, patient_concept_ls: List[str]):
        pat_h_concepts = [] # list of tuples
        for concept in patient_concept_ls:
            if concept not in self.h_vocabulary:
                if concept not in self.df_sks_names.stack().unique(): # check if it's in the database
                    self.add_unknown_concept_to_hierarchy(concept)
                self.h_vocabulary[concept] = self.get_lowest_level_node(concept)
            pat_h_concepts.append(self.h_vocabulary[concept])
        return pat_h_concepts
    
    def add_unknown_concept_to_hierarchy(self, concept:str)->None:
        """Add a new concept to the hierarchy. The concept is a tuple of the form (name, parent, level)"""
        max_node_lvl0 = self.df_sks_tuples.iloc[:,0].max()
        new_max_node_lvl0 = (max_node_lvl0[0]+1,) + max_node_lvl0[1:] # increment
        new_hierarchy_row = self.extend_node_to_lower_levels(new_max_node_lvl0)

        self.df_sks_tuples.loc[len(self.df_sks_tuples)] = new_hierarchy_row # add at the end of the df
        self.df_sks_names.loc[len(self.df_sks_names)] = [concept] * len(new_hierarchy_row) # add at the end of the df

    def construct_h_table_from_dics(self, tree:List[Dict[str, tuple]])->tuple[pd.DataFrame, pd.DataFrame]:
        """From a list of dictionaries construct two pands dataframes, where each dictionary represents a column
        The relationship of the rows is defined by the tuples in the dictionaries"""
        
        synchronized_ls = self.synchronize_levels(tree)
        
        inv_tree = [self.invert_dic(dic) for dic in tree]
        df_sks_tuples= pd.DataFrame(synchronized_ls).T
        df_sks_names = df_sks_tuples.copy()
        # map onto names
        for i, col in enumerate(df_sks_tuples.columns):
            df_sks_names[col] = df_sks_tuples[col].map(lambda x: inv_tree[i][x])
        
        return df_sks_names, df_sks_tuples

    @staticmethod
    def get_sks_vocab_ls(sks_vocab_tup:Dict[str, Tuple])->List[Dict[str, Tuple]]:
        """Convert tuple dict to a list of dicts, one for each level"""
        num_levels = len(sks_vocab_tup[list(sks_vocab_tup.keys())[0]])
        vocab_ls = [dict() for _ in range(num_levels)]
        for node_key, node in sks_vocab_tup.items():
            if 0 in node:
                level = node.index(0)
            else:
                level = -1
            vocab_ls[level-1][node_key] = node
        return vocab_ls

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
    
    def save_vocab(self, dest: str):
        torch.save(self.vocabulary, join(dest, "vocabulary.pt"))
        torch.save(self.h_vocabulary, join(dest, "h_vocabulary.pt"))
        self.df_sks_names.to_csv(join(dest, "sks_names.csv"), index=None)
        self.df_sks_tuples.to_csv(join(dest, "sks_tuples.csv"), index=None)

    # additional funcs, can be part of vocab constructor
    @staticmethod
    def extend_one_level(nodes0:Dict[str, Tuple], nodes1:Dict[str, Tuple], nodes1_lvl:int) -> Dict[str, Tuple]:
        """Takes a two dictionaries on two adjacent levels and extends the leafs of the first to the second one. 
        dic0: dictionary on level i
        dic1: dictionary on level i+1
        dic1_level: level i+1"""
        for node0_key, node0 in tqdm(nodes0.items(), desc='extending level'):
            flag = False
            for _, node1 in nodes1.items():
                if (node0[:nodes1_lvl]==node1[:nodes1_lvl]):
                    flag = True
                    break

            if not flag:
                nodes1[node0_key] = node0[:nodes1_lvl] + (1,) + node0[nodes1_lvl+1:]
        return nodes1

    def extend_leafs(self, h_dic:Dict[str, Tuple])->List[Dict]:
        """Takes a list of ordered dictionaries, where each dictionary represents nodes on one hierarchy level by tuples
        and extends leafs that are not on the lowest level"""
        tree = self.get_sks_vocab_ls(h_dic) # turn dict of tuples into list of dicts, one for each level
        for level in tqdm(range(len(tree)-1), desc='extending leafs'):
            tree[level+1] = self.extend_one_level(tree[level], tree[level+1], level+1)
        return tree

    def fill_parents(self, tree:List[Dict[str, Tuple]])->List[Dict]:
        """Takes a list of ordered dictionaries, where each dictionary represents nodes on one hierarchy level by tuples
        and fills in missing parents"""
        n_levels = len(tree)
        for level in tqdm(range(len(tree)-2, -1, -1), desc='filling parents'): # start from bottom level, and go to the top
            tree[level] = self.fill_parents_one_level(tree[level], tree[level+1], level+1, n_levels)
        return tree

    @staticmethod
    def fill_parents_one_level(node_dic0:Dict[str, Tuple], node_dic1:Dict[str, Tuple], node_dic1_level:int, n_levels:int):
        """Takes two dictionaries on two adjacent levels and fills in missing parents."""
        for node1_key, node1 in node_dic1.items():
            parent_node = node1[:node_dic1_level] + (0,)*(n_levels-node_dic1_level)# fill with zeros to the end of the tuple
            if parent_node not in node_dic0.values():
                node_dic0[node1_key] = parent_node
        return node_dic0

    @staticmethod
    def replicate_nodes_to_match_lower_level(nodes0: List[tuple], nodes1:List[tuple], nodes1_level:int)->List[tuple]:
        """Given two lists of nodes on two adjacent levels, replicate nodes of dic0 to match dic1."""
        new_nodes0 = []
        for node1 in nodes1:
            for node0 in nodes0:
                if node0[:nodes1_level] == node1[:nodes1_level]:
                    new_nodes0.append(node0)
        return new_nodes0

    def synchronize_levels(self, tree:List[Dict[str, tuple]])->List[List]:
        """Takes a list of ordered dictionaries, where each dictionary represents nodes on one hierarchy level by tuples
        and replicates nodes on one level to match the level below"""
        tree = tree[::-1] # we invert the list to go from the bottom to the top

        dic_bottom = tree[0] # lowest level
        ls_bottom = sorted([v for v in dic_bottom.values()])

        tree_depth = ls_bottom[0].__len__()

        ls_ls_tup = [] 
        ls_ls_tup.append(ls_bottom) # we can append the lowest level as it is
        
        for top_level in range(1, len(tree)): #start from second entry
            dic_top = tree[top_level]
            ls_top = sorted([v for v in dic_top.values()])
            ls_bottom = ls_ls_tup[top_level-1]
            ls_top_new = self.replicate_nodes_to_match_lower_level(ls_top, ls_bottom, tree_depth-top_level)
            ls_ls_tup.append(ls_top_new)
        
        ls_ls_tup = ls_ls_tup[::-1] # we invert the list to go from the bottom to the top
        return ls_ls_tup
    @staticmethod
    def invert_dic(dic:Dict)->Dict:
            return {v:k for k,v in dic.items()}

    @staticmethod
    def extend_node_to_lower_levels(node:Tuple)->List[Tuple]:
        """Given a tuple, extend it to the lowest level, by replicating and adding 1s to the end of the tuple"""
        row_ls = []
        row_ls.append(node)
        for i in range(node.__len__()-1):
            new_node = node[:i+1] + (1,) + node[i+2:]
            node = new_node
            row_ls.append(new_node)
        return row_ls
    
    # obsolete?
    def get_leaf_nodes(self, vocab):
        """
            Get the leaf nodes of a tree defined by a dictionary of tuples.
            Parameters: 
                tuple_dic: A dictionary of tuples, where the keys are the codes and the values are the tuples.
        """
        # Step 1: Create a set of parent nodes
        parent_nodes = set(self.get_parent(node_tuple) for node_tuple in vocab.values())
        # Step 2: Identify leaf nodes
        leaf_nodes = {code: node_tuple for code, node_tuple in vocab.items() if node_tuple not in parent_nodes}
        return leaf_nodes
