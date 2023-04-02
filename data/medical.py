import pickle as pkl
import random
import string
from os.path import dirname, join
from typing import Dict, List, Tuple
import torch
import pandas as pd
from tqdm import tqdm

"""  
("UA","UA"), # Abdominal Circumference [cm]
("UB","UB"), # Orificiums opening into [cm]
("UH","UH"), # Head circumference [cm]
("UP","UP"), # Fetus
("UT","UT"), # Mother
("V00","V99"), # Weight of the placenta [g]
("VA","VA"), # Undisclosed
("VRA","VRA"), # regarding weight, height 
("VRB","VRB"), # tobacco and alcohol
("VRK","VRK"), # interoperative bleeding
"""
# convert raw data to working format
def sks_codes_to_list()->None:
    """Convert the SKScomplete list of codes to a list of codes in pickles format."""
    codes = []
    with open(join(dirname(__file__), "SKScomplete.txt")) as f:
        for line in f:
            codes.append(line.split(' ')[0])
    codes = set(codes)
    with open("SKScodes.pkl", "wb") as f:
        pkl.dump(codes, f)

def npu_codes_to_list()->None:
    """Convert the NPUlistEN221222.csv to a list of codes in pickles format."""
    df = pd.read_csv(join(dirname(__file__),'NPUlistEN221222.csv'), 
        encoding='latin-1', delimiter=';', usecols=['NPU code'])
    codes = df['NPU code'].unique().tolist()
    # replace DNK stands for local danish NPU codes
    codes = [c.replace('DNK', 'labL').replace('NPU', 'labL') for c in codes]
    with open( "NPUcodes.pkl", "wb") as f:
        pkl.dump(codes, f)

class MedicalCodes():
    """
    Loads medical codes from SKS database and NPU (lab codes).
    codes can be accessed by prefix (D, L, M) or signature (lab, dia, ...).
    """

    def __init__(self):
        with open(join(dirname(__file__), "SKScodes.pkl"), "rb") as f:
            self.sks_codes = list(pkl.load(f))
        with open(join(dirname(__file__), "NPUcodes.pkl"), "rb") as f:
            self.npu_codes = pkl.load(f) 
        self.all_codes = self.npu_codes + self.sks_codes
        self.prefix_dic ={
            'L':'lab',
            'D':'icd',
            'M':'atc',
            'A':'adm',
            'O':'operations',
            'P':'procedures',
            'T':'special_codes',
            'U':'ext_injuries',
            'V':'studies',
        }
    def get_codes_by_prefix(self, prefix):
        return getattr(self, 'get_'+self.prefix_dic[prefix])()

    def get_codes_type(self, signature, min_len=2):
        codes =[c.strip(signature) for c in self.all_codes if c.startswith(signature)]
        return [c for c in codes if len(c)>=min_len]

    def get_lab(self):
        return sorted(self.get_codes_type('lab'))
    def get_icd(self):
        return sorted(self.get_codes_type('dia', min_len=4))
    def get_atc(self):
        codes = self.get_codes_type('atc', min_len=4)
        codes[codes.index('N05CC10')] = 'MZ99' # thalidomid, wrongly generated code will be assigned a new code
        return sorted(codes)
    def get_adm(self):
        return sorted(self.get_codes_type('adm'))
    def get_operations(self):
        return sorted(self.get_codes_type('opr'))
    def get_procedures(self):
        return sorted(self.get_codes_type('pro'))
    def get_special_codes(self):
        return sorted(self.get_codes_type('til'))
    def get_ext_injuries(self):
        return sorted(self.get_codes_type('uly'))
    def get_studies(self):
        return sorted(self.get_codes_type('und'))

#TODO: Implement subtopics for ICD codes
#TODO: Add option to reduce level, and flatten nodes from levels below
class SKSVocabConstructor():
    """
    Construct two vocabularies for medical codes, mapping to integers and tuples (nodes).
    Every integer of the tuple specifies a branch on a level. Integer 0 is reserved for empty node.
    Currently we have the hierarchy for medication and diagnosis implemented. 
    """
    def __init__(self, main_vocab=None, code_types=['D', 'M'], num_levels=8):
        """
        main_vocab: initial vocabulary, if None, create a new one
        code_types: list of code types to include in the vocabulary (by prefix D, M, L)
        num_levels: number of levels in the hierarchy, don't change this if you don't know what you are doing.
        """
        
        self.code_types = code_types
        for code_type in self.code_types:
            if code_type not in ['D', 'M']:
                raise ValueError(f'Hierarchy for type {code_type} not implemented yet.') 

        self.num_levels=num_levels
        self.vocabs = [{'[ZERO]':0} for _ in range(num_levels)] # these vocabularies will contain the tree

        self.medcodes = MedicalCodes()

        # take care of special codes
        if isinstance(main_vocab, type(None)):
            self.special_codes = ['[CLS]', '[PAD]', '[SEP]', '[MASK]', '[UNK]', '[BG_Mand]', '[BG_Kvinde]']
            self.main_vocab = {token: idx for idx,
                               token in enumerate(self.special_codes)}
        else:
            self.main_vocab = main_vocab
            self.special_codes = [k for k in main_vocab if k.startswith('[')]

        self.init_icd_helpers()
        self.atc_topic_ls = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
        self.atc_topic_dic = {topic:(i+1) for i, topic in enumerate(self.atc_topic_ls)}
        self.topics = {'D':self.icd_topic_options, 'M':self.atc_topic_ls}

        self.init_temp_vocabs()


    def __call__(self, check_for_uniqueness=True)->Tuple[Dict[str, int], Dict[str, Tuple[int]]]:
        """Return vocab, mapping concepts to tuples, where each tuple element is a code on a level
        The dictionares contain concept present in the SKS code and the ones inmain vocab.
        types contains the types of codes to be included in the vocabulary, e.g. ['D', 'M', 'L']"""
        
        self.handle_level_one() # add special tokens/code types to the first level
       
        for code_type in self.code_types: # Loop over disease, medication, lab_codes,...
            self.handle_codetype(code_type)
        # for codes which exist only in the top n levels (e.g. type codes, topics codes), 
        # we add zeros to lower levels
        self.fill_with_zeros() 
       
        tree = {} # this is the dictionary of tuples (first vocab contains all the concepts)
        for concept in self.vocabs[0]:
            tree[concept] = self.map_concept_to_tuple(concept)
        
        if check_for_uniqueness:
            if not self.unique_nodes(tree):
                raise ValueError('Not all nodes are unique')
        del tree['[ZERO]']

        tree = self.extend_leafs_to_bottom(tree)

        return self.main_vocab, tree

    def extend_leafs_to_bottom(self, tree:Dict[str, Tuple]):
        """Leafs that are not on the bottom level are filled with 1s"""
        tree_tensor = torch.tensor([v for v in tree.values()], dtype=torch.long)
        for row_id in tqdm(range(len(tree_tensor)), 'extend_leafs_to_bottom'):        
            if (tree_tensor[row_id]!=0).all(): # already at bottom level
                continue
            leaf_level = (tree_tensor[row_id]!=0).nonzero()[-1]+1 
            children_mask = (tree_tensor[row_id, :leaf_level] == tree_tensor[:, :leaf_level]).all(dim=1) 
            if children_mask.sum()<=1: # if only equal to itself, no children 
                tree_tensor[row_id, leaf_level:] = 1
            # insert back into vocab
        tree = {k:tuple(tree_tensor[row_id].tolist()) for row_id, k in enumerate(tree.keys())}
        return tree

    def handle_level_one(self):
        """Add special tokens and code types to the first level"""
        for special_code in self.special_codes:
            self.vocabs[0][special_code] = len(self.vocabs[0]) 
        for code_type in self.code_types:
            self.vocabs[0][code_type] = len(self.vocabs[0]) 

        for code_type in self.code_types:
            codes = self.medcodes.get_codes_by_prefix(code_type)
            for code in codes:
                self.vocabs[0][code] = self.vocabs[0][code_type]
       

    def handle_codetype(self, code_type)->None:
        """Here we go through the different EHR code types and add them to the hierarchy"""
        if code_type == 'D':
            print('Adding diagnosis codes')
            self.add_icd_vocabs()
        elif code_type == 'M':
            print('Adding medication codes')
            self.add_atc_vocabs()
        else:
            print(f"Codes of type {code_type} are not yet implemented")
        return None

    def fill_with_zeros(self):
        """Starting from top level, if concept not present in level below, fill with zeros"""
        for i, vocab in enumerate(self.vocabs[:-1]):
            for concept in vocab:
                if concept not in self.vocabs[i+1]:
                    self.vocabs[i+1][concept] = 0

    def map_concept_to_tuple(self, concept:str)->Tuple[int]:
        """Using the list of vocabs, map a concept to a tuple of integers"""
        tuple_of_integers = []
        for vocabulary in self.vocabs:
            if concept in vocabulary:
                tuple_of_integers.append(vocabulary[concept])
            else:
                tuple_of_integers.append(vocabulary["[UNK]"])
        return tuple(tuple_of_integers)

    def add_icd_vocabs(self)->None:
        """Add ICD codes to the vocabulary"""
        for topic in self.icd_topic_options:
            self.add_topic(topic, 'D')
        for code in self.medcodes.get_codes_by_prefix('D'):
            self.vocabs[1][code] = self.get_topic(code)

        self.add_category('D') # A00, A01, ..., Z99
        # this also takes care of the special codes
        # think whether it can be separated and moved to add_lower_levels_icd
        self.add_lower_levels_icd() 
        return None

    def add_lower_levels_icd(self):
        for code in self.medcodes.get_codes_by_prefix('D'):
            if code.startswith(('DU', 'DV')): # filled at topic level already
                continue
            for i, vocab in enumerate(self.vocabs[3:]): # 0/1/2 are type/topic/category
                if len(code)>i+4:
                    vocab[code] = self.alphanumeric_vocab[code[i+4]]

    def add_atc_vocabs(self)->None:
        """Add ATC codes to the vocabulary"""
        for topic in self.atc_topic_ls:
            self.add_topic(topic, 'M')
        self.add_category('M') # A, B, C, ..., V
        self.add_lower_level_atc()
        return None

    def add_lower_level_atc(self):
        for code in self.medcodes.get_codes_by_prefix('M'):
            for i, vocab in enumerate(self.vocabs[3:]): # check this
                if len(code)>i+4:
                    vocab[code] = self.alphanumeric_vocab[code[i+4]]

    # Topics
    def add_topic(self, topic:Tuple[str, str], code_type:str):
        """Add all codes from a topic to the vocabulary"""
        type_voc, topic_voc = self.vocabs[:2]

        for topic in self.topics[code_type]:
            if code_type == 'D':
                topic_name = f"D{topic[0]}-D{topic[1]}"
            elif code_type == 'M':
                topic_name = f"M{topic}"
            type_voc[topic_name] = type_voc[code_type]
            topic_voc[topic_name] = self.get_topic(topic_name)
        

    def get_topic(self, code):
        if code.startswith('D'):
            return self.get_ICD_topic(code)
        elif code.startswith('M'):
            return self.get_ATC_topic(code)
        else:
            print(f"Code type starting with {code[0]} not implemented yet")
    
    def get_ICD_topic(self, code):
        for i, option in enumerate(self.icd_topic_options):
            if option[0] <= code[1:4] <= option[1]:
                return i+1
            elif code.startswith("DU"): # special codes (childbirth and pregnancy)
                return len(self.icd_topic_options)+2
            elif code.startswith("DV"): #weight, height and various other codes
                return len(self.icd_topic_options)+3
        return len(self.icd_topic_options)+4

    def get_ATC_topic(self, code):
        """Returns the topic of an ATC code, here it's simply the first letter of the code'"""
        if code[1] in self.atc_topic_dic:
            return self.atc_topic_dic[code[1]]
        else:
            return len(self.atc_topic_dic) + 2 #we start at 1, so we need to add 2

    # TODO: implement subtopics for ICD codes
    def add_category(self, type:str):
        """Add ICD/ATC categories to the vocabulary"""
        for code in self.medcodes.get_codes_by_prefix(type):
            if code.startswith(('DU', 'DV')):
                self.handle_special_icd_codes(code) # here we handle all levels of the special codes
                continue
            category = code[:4] # e.g. DA00, DA01, ..., DZ99 / MA00, MA01, ..., MZ99
            if category not in self.vocabs[2]: # we first create the category keys, and then use it to fill the topic and type keys
                if type == 'D':
                    cat_int = self.icd_category_vocab[category]
                elif type == 'M':
                    cat_int = self.two_digit_vocab[category[2:]]
                else:
                    print(f"Code type starting with {type[0]} not implemented yet")
                self.vocabs[2][category] = cat_int
                self.vocabs[1][category] = self.get_topic(category)
                self.vocabs[0][category] = self.vocabs[0][type]
            self.vocabs[2][code] = self.vocabs[2][category]
            self.vocabs[1][code] = self.vocabs[1][category]

    def handle_special_icd_codes(self, code):
        """Handle special codes DU, DV"""
        cat_vocab = self.vocabs[2]

        def handle_DU_codes():
            """Related to pregnancy and childbirth"""
            if self.all_digits(code[2:4]):
                handle_DUiiixxx()
            elif code[2] in ['A', 'B' ,'U', 'H', 'P', 'T']:
                handle_DUX()
            else:
                print(f"category for code {code} not implemented yet")

        def handle_DUiiixxx():
            """Special code, U followed by three digits/ 2 digits, then sometimes D (for days), 
                stands for durations of pregnancy"""
            if "DUiiixxx" not in self.vocabs[2]:
                    cat_vocab['DUiiixxx'] = 1
            cat_vocab[code] = cat_vocab['DUiiixxx']
            self.vocabs[3][code] = self.DUiii_voc[code]

        def handle_DUX():
            """Here X stands for A, B, H, P, T"""
            cat_vocab[code] = string.ascii_uppercase.index(code[2])+2 # skip 0 and DUiiixxx 
            self.vocabs[3][code] = self.abc123_voc[code[3:5]]# next two integers stand for a measure e.g. DUA10 is 10 cm circumference
            # these codes end here

        def handle_DV_codes():
            if self.all_digits(code[2:]):
                handle_DViiii()
            if code[2] == 'A':
                handle_DVA()
            if code[2] == 'R':
                handle_DVR()
            
        def handle_DViiii():
            """Special code, V followed by four digits, stands for weight, height, etc."""
            if "DViiii" not in self.vocabs[2]:
                    cat_vocab['DViiii'] = 1
            cat_vocab[code] = cat_vocab['DViiii']
            self.vocabs[3][code] = self.DV4int_voc[code]

        def handle_DVA():
            """Special code, VA followed by two digits, stands for weight, height, etc."""
            if "DVA" not in self.vocabs[2]:
                cat_vocab['DVA'] = 2
            cat_vocab[code] = cat_vocab['DVA']
            self.vocabs[3][code] = self.abc123_voc[code[3:5]]

        def handle_DVR():
            if "DVR" not in self.vocabs[2]:
                cat_vocab['DVR'] = 3
            cat_vocab[code] = cat_vocab['DVR']

            if code[3] == 'A':
                handle_DVRA()
            elif code[3] == 'B':
                handle_DVRB()
            elif code[3] == 'K':
                handle_DVRK()
            else:
                print(f"Subcategories for {code} not implemented yet")

        def handle_DVRA():
            if "DVRA" not in self.vocabs[2]:
                self.vocabs[3]['DVRA'] = 1
            self.vocabs[3][code] = self.vocabs[3]['DVRA']
            self.vocabs[4][code] = self.two_digit_vocab[code[4:6]] 
            if len(code)>6:
                self.vocabs[5][code] = self.alphanumeric_vocab[code[-1]] # 'DVRA02A'
    
        def handle_DVRB():
            if "DVRB" not in self.vocabs[2]:
                self.vocabs[3]['DVRB'] = 2
            self.vocabs[3][code] = self.vocabs[3]['DVRB']
            self.vocabs[4][code] = self.two_digit_vocab[code[4:6]] # last level

        def handle_DVRK():
            if "DVRK" not in self.vocabs[2]:
                self.vocabs[3]['DVRK'] = 3
            self.vocabs[3][code] = self.vocabs[3]['DVRK'] 
            self.vocabs[4][code] = 1 # only one branch
            
        
        if code.startswith('DU'):
            handle_DU_codes()
        if code.startswith('DV'):
            handle_DV_codes()
        self.vocabs[2] = cat_vocab

    @staticmethod
    def all_digits(codes):
        """Check if a string only contains digits"""
        return all([c.isdigit() for c in codes])

    def get_DU_two_digit_codes_voc(self)->Dict[str, int]:
        """Get a vocabulary for the two digit codes in the DUiii branch """
        duii_ls = list(set([code for code in self.medcodes.get_icd() if code.startswith('DU') and self.all_digits(code[2:4])]))
        return {code: idx+1 for idx, code in enumerate(duii_ls)}

    def get_DV_four_digit_codes_voc(self)->Dict[str, int]:
        """Get a vocabulary for the four digit codes in the DV branch """
        dv4int_ls = list(set([code for code in self.medcodes.get_icd() if code.startswith('DV') and self.all_digits(code[2:6])]))
        return {code: idx+1 for idx, code in enumerate(dv4int_ls)}

    def init_icd_helpers(self):
        self.icd_topic_options = [
            ("A00","B99"), # Certain Infectious and Parasitic Diseases
            ("C00","D48"), # Neoplasms
            ("D50","D89"), # Blood, Blood-Forming Organs, and Certain Disorders Involving the Immune Mechanism
            ("E00","E90"), # Endocrine, Nutritional, and Metabolic Diseases, and Immunity Disorders
            ("F00","F99"), # Mental, Behavioral, and Neurodevelopmental Disorders
            ("G00","G99"), # Diseases of the Nervous System
            ("H00","H59"), # Diseases of the Eye and Adnexa
            ("H60","H95"), # Diseases of the Ear and Mastoid Process
            ("I00","I99"), # Diseases of the Circulatory System
            ("J00","J99"), # Diseases of the Respiratory System
            ("K00","K93"), # Diseases of the Digestive System
            ("L00","L99"), # Diseases of the Skin and Subcutaneous Tissue
            ("M00","M99"), # Diseases of the Musculoskeletal System and Connective Tissue
            ("N00","N99"), # Diseases of the Genitourinary System
            ("O00","O99"), # Pregnancy, Childbirth, and the Puerperium
            ("P00","P96"), # Certain Conditions Originating in the Perinatal Period
            ("Q00","Q99"), # Congenital Malformations, Deformations, and Chromosomal Abnormalities
            ("R00","R99"), # Symptoms, Signs, and Ill-Defined Conditions
            ("S00","T98"), # Injury, Poisoning, and Certain Other Consequences of External Causes
            ("X60","Y09"), # External Causes of Injury
            ("Z00","Z99"), # Factors Influencing Health Status and Contact with Health Services
            ("U00","UZZ"), # Supplementary Classification of Factors Influencing Health Status and Contact with Health Services
            ("V00","VZZ"), # Supplementary Classification of External Causes of Injury and Poisoning
        ]  
        self.DUiii_voc = self.get_DU_two_digit_codes_voc()
        self.DV4int_voc = self.get_DV_four_digit_codes_voc()

    def init_temp_vocabs(self):
        self.alphanumeric_vocab = self.add_alphanumeric_vocab()
        self.two_digit_vocab = self.add_two_digit_vocab()
        self.icd_category_vocab = self.add_icd_category_vocab()
        self.abc123_voc = self.add_two_letter_vocab(self.add_two_digit_vocab(self.alphanumeric_vocab)) # complete vocabulary with 1 and 2 letter codes

    @staticmethod
    def add_two_digit_vocab(temp_vocab:Dict[str, int]={'[ZERO]':0})->Dict[str, int]:
        """Takes vocabulary and expands with two digit codes 00, 01, 02, ...
        Useful for ICD codes, where the first two digits are the category"""
        for i in range(10):
                for j in range(10):
                    temp_vocab[str(i)+str(j)] = len(temp_vocab)
        return temp_vocab
        
    @staticmethod
    def add_alphanumeric_vocab(temp_vocab:Dict[str, int]={'[ZERO]':0})->Dict[str, int]:
        """Takes a vocabulary, and extends it with the alphanumeric characters len(vocab)+1:0, len(vocab)+2:1, ...)"""
        for i in range(10):
            temp_vocab[str(i)] = len(temp_vocab)
        for a in string.ascii_uppercase:
            temp_vocab[a] = len(temp_vocab)
        return temp_vocab
    @staticmethod
    def add_two_letter_vocab(temp_vocab:Dict[str, int]={'[ZERO]':0})->Dict[str, int]:
        """Takes a vocabulary, and extends it with the two letter codes len(vocab)+1:0, len(vocab)+2:1, ...)"""
        for a in string.ascii_uppercase:
            for b in string.ascii_uppercase:
                temp_vocab[a+b] = len(temp_vocab)
        return temp_vocab

    @staticmethod
    def add_icd_category_vocab(temp_vocab:Dict[str, int]={'[ZERO]':0})->Dict[str, int]:
        """Takes a vocabulary, and extends it with the two letter codes len(vocab)+1:0, len(vocab)+2:1, ...)"""
        for X in string.ascii_uppercase:
            for i in range(10):
                for j in range(10):
                    temp_vocab['D'+X+ str(i)+str(j)] = len(temp_vocab)
        return temp_vocab

    @staticmethod
    def unique_nodes(h_vocab: Dict[str, Tuple])->bool:
        """Check that nodes are unique."""
        nodes = []
        for k, v in tqdm(h_vocab.items(), 'Checking nodes for uniqueness'):
            if v in nodes:
                print(k)
                print(f'Node {v} is not unique!')
                print({k:v2 for k, v2 in h_vocab.items() if v2==v})
                return False
            nodes.append(v)
        return True


class TableConstructor():
    """Producing tables with names and tuples of the entire tree."""
    def __init__(self, sks_tree=None, main_vocab=None, code_types=['D', 'M'], num_levels=8) -> None:
        if isinstance(sks_tree, type(None)):
            print("Constructing SKS tree...")
            _, self.sks_tree = SKSVocabConstructor(main_vocab, code_types, num_levels)()
        else:
            self.sks_tree = sks_tree

    def __call__(self)->Tuple[pd.DataFrame, pd.DataFrame]:
        # split the levels into several dicts, to avoid issues with repeating keys
        self.tree_ls = self.get_sks_tree_ls(self.sks_tree)
        # fill parents to top level, e.g.: SEP:(1,1,1) ->  SEP:(1,1,0), SEP:(1,0,0), in separate dicts
        self.extended_tree_ls = self.fill_parents(self.tree_ls) 
        names, tuples = self.construct_h_table_from_dics(self.extended_tree_ls) # full table of the SKS vocab tree
        return self.extended_tree_ls, names, tuples
    
    def construct_h_table_from_dics(self, tree:List[Dict[str, tuple]])->tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main function to construct the SKS vocab tree.
        From a list of dictionaries construct two pands dataframes, where each dictionary represents a column
        The relationship of the rows is defined by the tuples in the dictionaries
        """
        synchronized_ls = self.synchronize_levels(tree) # list of dictionaries, one for each level
        
        df_sks_tuples= pd.DataFrame(synchronized_ls).T
        df_sks_names = df_sks_tuples.copy()

        inv_tree = [self.invert_dic(dic) for dic in tree] 
        # map onto names
        for i, col in enumerate(df_sks_tuples.columns):
            df_sks_names[col] = df_sks_tuples[col].map(lambda x: inv_tree[i][x])
        
        return df_sks_names, df_sks_tuples    

    def fill_parents(self, tree:List[Dict[str, Tuple]])->List[Dict]:
        """Takes a list of ordered dictionaries, where each dictionary represents nodes on one hierarchy level by tuples
        and fills in missing parents"""
        n_levels = len(tree)
        for level in tqdm(range(len(tree)-2, -1, -1), desc='filling parents'): # start from bottom level, and go to the top
            tree[level] = self.fill_parents_one_level(tree[level], tree[level+1], level+1, n_levels)
        return tree

    @staticmethod
    def fill_parents_one_level(parent_dic:Dict[str, Tuple], child_dic:Dict[str, Tuple], child_dic_level:int, n_levels:int):
        """Takes two dictionaries on two adjacent levels and fills in missing parents."""
        for child_name, child_node in child_dic.items():
            parent_node = child_node[:child_dic_level] + (0,)*(n_levels-child_dic_level)# fill with zeros to the end of the tuple
            if parent_node not in parent_dic.values():
                parent_dic[child_name] = parent_node
        return parent_dic
    

    @staticmethod
    def get_sks_tree_ls(sks_tree_tup:Dict[str, Tuple])->List[Dict[str, Tuple]]:
        """Convert tuple dict to a list of dicts, one for each level"""
        num_levels = len(sks_tree_tup[list(sks_tree_tup.keys())[0]])
        vocab_ls = [dict() for _ in range(num_levels)]
        for node_key, node in sks_tree_tup.items():
            if 0 in node:
                level = node.index(0)
            else:
                level = num_levels 
            vocab_ls[level-1][node_key] = node
        return vocab_ls
    
    def synchronize_levels(self, tree:List[Dict[str, Tuple]])->List[List[Tuple]]:
        """Takes a list of ordered dictionaries, where each dictionary represents nodes on one hierarchy level by tuples
        and replicates nodes on one level to match the level below"""
        tree = tree[::-1] # we invert the list to go from the bottom to the top
        dic_bottom = tree[0] # lowest level
        ls_bottom = sorted([v for v in dic_bottom.values()])

        tree_depth = ls_bottom[0].__len__()

        ls_ls_tup = [] 
        ls_ls_tup.append(ls_bottom) # we can append the lowest level as it is
        
        for top_level in tqdm(range(1, len(tree)), desc='Syncrhonize'): #start from second entry
            dic_top = tree[top_level]
            ls_top = sorted([v for v in dic_top.values()])
            ls_bottom = ls_ls_tup[top_level-1]
            ls_top_new = self.replicate_nodes_to_match_lower_level(ls_top, ls_bottom, tree_depth-top_level)
            ls_ls_tup.append(ls_top_new)
        
        ls_ls_tup = ls_ls_tup[::-1] # we invert the list to go from the bottom to the top
        return ls_ls_tup

    @staticmethod
    def replicate_nodes_to_match_lower_level(top_nodes: List[tuple], bottom_nodes:List[tuple], bottom_level:int)->List[tuple]:
        """Given two lists of nodes on two adjacent levels, replicate nodes of dic0 to match dic1."""
        new_top_nodes = []
        for bottom_node in tqdm(bottom_nodes, desc='replicate node'):
            for top_node in top_nodes:
                if top_node[:bottom_level] == bottom_node[:bottom_level]:
                    new_top_nodes.append(top_node)
        return new_top_nodes

    @staticmethod
    def invert_dic(dic:Dict)->Dict:
            return {v:k for k,v in dic.items()}



