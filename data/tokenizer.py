import torch
from transformers import BatchEncoding
from os.path import join
from medical import SKSVocabConstructor
from typing import List, Dict
import pandas as pd

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
    def __init__(self, config, vocabulary=None):
        super().__init__(config, vocabulary)
        self.h_vocabulary = {}
        sks_vocab_ls = SKSVocabConstructor(main_vocab=vocabulary)
        self.extended_sks_vocab_ls = self.extend_leafs(sks_vocab_ls) # extend leaf nodes to bottom level
        self.df_sks_names, self.df_sks_tuples = self.construct_h_table_from_dics(self.extended_sks_vocab_ls) # full table of the SKS vocab tree

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
                self.h_vocabulary[concept] = self.get_lowest_level_tuple(concept)
            pat_h_concepts.append(self.h_vocabulary[concept])
        return pat_h_concepts
    
    def add_unknown_concept_to_hierarchy(self, concept):
        """Add a new concept to the hierarchy. The concept is a tuple of the form (name, parent, level)"""
        max_tup_lvl0 = self.df_sks_tuples.iloc[:,0].max()
        new_max_tup_lvl0 = (max_tup_lvl0[0]+1,) + max_tup_lvl0[1:] # increment
        new_hierarchy_row = self.extend_tuple_to_lower_levels(new_max_tup_lvl0)
        self.df_sks_tuples.loc[len(self.df_sks_tuples)] = new_hierarchy_row
        self.df_sks_names.loc[len(self.df_sks_names)] = [concept] * len(new_hierarchy_row)

    def construct_h_table_from_dics(self, ls_dic:List[Dict[str, tuple]])->tuple[pd.DataFrame, pd.DataFrame]:
        """From a list of dictionaries construct two pands dataframes, where each dictionary represents a column
        The relationship of the rows is defined by the tuples in the dictionaries"""
        synchronized_ls = self.synchronize_levels(ls_dic)
        
        inv_ls_dic = [self.invert_dic(dic) for dic in ls_dic]
        df_tup = pd.DataFrame(synchronized_ls).T
        df_names = df_tup.copy()
        # map onto names
        for i, col in enumerate(df_tup.columns):
            df_names[col] = df_tup[col].map(lambda x: inv_ls_dic[i][x])
        
        return df_names, df_tup

    def get_lowest_level_tuple(self, name):
        """Get the lowest level tuple of a concept
        E.g. the [SEP] token exists on all levels, but we want to get the lowest level tuple, so we would get (i, 1,1,1,..,1)"""
        indices = self.df_names[self.df_names==name].stack().index
        max_id = indices.max()
        return self.df_tuples.iloc[max_id[0], max_id[1]]

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

    @staticmethod
    def get_parent(node_tuple):
        """Get parent node of a node defined by a tuple."""
        idx = next((i for i, x in enumerate(node_tuple) if x == 0), None)
        if idx is not None:
            return node_tuple[:idx - 1] + (0,) * (len(node_tuple) - idx + 1)
        else:
            return node_tuple[:-1] + (0,)
    
    def save_vocab(self, dest: str):
        torch.save(self.vocabulary, join(dest, "vocabulary.pt"))
        torch.save(self.h_vocabulary, join(dest, "h_vocabulary.pt"))
        self.df_sks_names.to_csv(join(dest, "sks_names.csv"), index=None)
        self.df_sks_tuples.to_csv(join(dest, "sks_tuples.csv"), index=None)

    # additional funcs, can be part of vocab constructor
    @staticmethod
    def extend_one_level(dic0:Dict, dic1:Dict, dic1_level:int) -> Dict:
        """Takes a two dictionaries on two adjacent levels and extends the leafs of the first to the second one. 
        dic0: dictionary on level i
        dic1: dictionary on level i+1
        dic1_level: level i+1"""
        for k0, t0 in dic0.items():
            flag = False
            # print('k0',k)
            for _, t1 in dic1.items():
                if (t0[:dic1_level]==t1[:dic1_level]):
                    # print('k1',k2)
                    flag = True
                    break
            if not flag:
                dic1[k0] = t0[:dic1_level] + (1,) + t0[dic1_level+1:]
        return dic1

    def extend_leafs(self, ls_dic:List[Dict[str, tuple]])->List[Dict]:
        """Takes a list of ordered dictionaries, where each dictionary represents nodes on one hierarchy level by tuples
        and extends leafs that are not on the lowest level"""
        for i in range(len(ls_dic)-1):
            ls_dic[i+1] = self.extend_one_level(ls_dic[i], ls_dic[i+1], i+1)
        return ls_dic

    @staticmethod
    def replicate_nodes_to_match_lower_level(ls0: Dict[str, tuple], ls1:Dict[str, tuple], ls1_level:int)->List[tuple]:
        """Given two dictionaries on two adjacent levels, replicate nodes of dic0 to match dic1."""
        new_ls0 = []
        for t1 in ls1:
            for t0 in ls0:
                if t0[:ls1_level] == t1[:ls1_level]:
                    new_ls0.append(t0)
        return new_ls0

    def synchronize_levels(self, ls_dic:List[Dict[str, tuple]])->List[List]:
        """Takes a list of ordered dictionaries, where each dictionary represents nodes on one hierarchy level by tuples
        and replicates nodes on one level to match the level below"""
        ls_dic = ls_dic[::-1] # we invert the list to go from the bottom to the top

        dic_bottom = ls_dic[0] # lowest level
        ls_bottom = sorted([v for v in dic_bottom.values()])

        tree_depth = ls_bottom[0].__len__()

        ls_ls_tup = [] 
        ls_ls_tup.append(ls_bottom) # we can append the lowest level as it is
        
        for top_level in range(1, len(ls_dic)): #start from second entry
            dic_top = ls_dic[top_level]
            ls_top = sorted([v for v in dic_top.values()])
            ls_bottom = ls_ls_tup[top_level-1]
            ls_top_new = self, self.replicate_nodes_to_match_lower_level(ls_top, ls_bottom, tree_depth-top_level)
            ls_ls_tup.append(ls_top_new)
        
        ls_ls_tup = ls_ls_tup[::-1] # we invert the list to go from the bottom to the top
        return ls_ls_tup
    @staticmethod
    def invert_dic(dic:Dict)->Dict:
            return {v:k for k,v in dic.items()}

    @staticmethod
    def extend_tuple_to_lower_levels(t):
        """Given a tuple, extend it to the lowest level, by adding 1s to the end of the tuple"""
        row_ls = []
        row_ls.append(t)
        for i in range(t.__len__()-1):
            new_t = t[:i+1] + (1,) + t[i+2:]
            t = new_t
            row_ls.append(new_t)
        return row_ls