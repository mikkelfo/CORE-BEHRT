import torch
from transformers import BatchEncoding
from os.path import join
from medical import SKSVocabConstructor

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
    and return hierarchical vocabulary
    The method get_leaf_nodes takes the hierarchical dictionary (with tuples as values) and returns the leaf nodes.
    It is recommendes to use sks_vocab to get leafs.
    This is because the SKS database is more complete.
    If during testing, a concept is encountered that does not exactly match to the training data, 
    we can still compute it's probability.
    """
    def __init__(self, config, vocabulary=None):
        super().__init__(config, vocabulary)
        self.h_vocabulary = {}
        self.sks_vocab = SKSVocabConstructor(main_vocab=vocabulary)

    def __call__(self, features: dict, padding=True, truncation=512):
        data = self.batch_encode(features, padding, truncation)
        data['h_concept'] = self.batch_encode_hierarchy(data)
        return data

    def batch_encode_hierarchy(self, data):
        """Encode the hierarchy of the concepts as tuple"""
        h_concepts = [] # list of lists of tuples
        for patient_ls in data['concept']:
            h_concepts.append(self.h_encode_patient(patient_ls))
        return h_concepts

    def h_encode_patient(self, patient_ls: dict):
        pat_h_concepts = [] # list of tuples
        for concept in patient_ls:
            if concept not in self.h_vocabulary:
                if concept not in self.sks_vocab:
                    self.h_vocabulary[concept] = self.sks_vocab['[UNK]'] # we might introduce a new UNK token for the hierarchy
                    pat_h_concepts.append(self.h_vocabulary['[UNK]'])
                else:
                    self.h_vocabulary[concept] = self.sks_vocab[concept]
                    pat_h_concepts.append(self.h_vocabulary[concept])
            else:
                pat_h_concepts.append(self.h_vocabulary[concept]) # increment count for this concept
        return pat_h_concepts
    
    def get_leaf_node(self, h_vocab):
        """
            Get the leaf nodes of a tree defined by a dictionary of tuples.
            Parameters: 
                tuple_dic: A dictionary of tuples, where the keys are the codes and the values are the tuples.
        """
        # Step 1: Create a set of parent nodes
        parent_nodes = set(self.get_parent(node_tuple) for node_tuple in h_vocab.values())
        # Step 2: Identify leaf nodes
        leaf_nodes = {code: node_tuple for code, node_tuple in h_vocab.items() if node_tuple not in parent_nodes}
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