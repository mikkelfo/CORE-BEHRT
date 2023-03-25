import pickle as pkl
import random
import string
from os.path import dirname, join
from typing import Dict, List, Tuple

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
    def get_codes_by_prefix(self, prefix, min_len=2):
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
class SKSVocabConstructor():
    """
    Construct two vocabularies for medical codes, mapping to integers and tuples (nodes).
    Every integer of the tuple specifies a branch on a level. Integer 0 is reserved for empty node.
    Currently we have the hierarchy for medication and diagnosis implemented. 
    """
    def __init__(self, main_vocab=None, code_types=None, num_levels=6):
        """main_vocab: initial vocabulary, if None, create a new one
        code_types: list of code types to include in the vocabulary (by prefix D, M, L)
        num_levels: number of levels in the hierarchy, don't change this if you don't know what you are doing.
        """
        self.num_levels=num_levels
        self.medcodes = MedicalCodes()

        if isinstance(main_vocab, type(None)):
            self.special_tokens = ['[CLS]', '[PAD]', '[SEP]', '[MASK]', '[UNK]', '[BG_Mand]', '[BG_Kvinde]']
            self.main_vocab = {token: idx for idx,
                               token in enumerate(self.special_tokens)}
        else:
            self.main_vocab = main_vocab
            self.special_tokens = [k for k in main_vocab if k.startswith('[')]
            
        self.vocabs = []
        self.alphanumeric_vocab = self.add_alphanumeric_vocab()
        self.two_digit_vocab = self.add_two_digit_vocab()
        if isinstance(code_types, type(None)):
            self.code_types=['D', 'M'] # diagnosis and medication by default
        else:
            self.code_types = code_types
    

        for code_type in self.code_types:
            if code_type not in ['D', 'M', 'L']:
                raise ValueError('Type not implemented yet.') # TODO: Instead add the codes on level 1


    def __call__(self)->Tuple[Dict[str, int], Dict[str, Tuple[int]]]:
        """Return vocab, mapping concepts to tuples, where each tuple element is a code on a level
        The dictionares contain concept present in the SKS code and the ones inmain vocab.
        types contains the types of codes to be included in the vocabulary, e.g. ['D', 'M', 'L']"""
        h_vocab = {}
        for level in range(self.num_levels+1):
            self.vocabs.append(self.construct_vocab_dic(level))
        for concept in self.vocabs[0]:
            h_vocab[concept] = self.map_concept_to_tuple(concept)
        # if not self.unique_nodes(h_vocab):
            # raise ValueError('Not all nodes are unique')
        return self.main_vocab, h_vocab

    def map_concept_to_tuple(self, concept:str)->Tuple[int]:
        """Using the list of vocabs, map a concept to a tuple of integers"""
        tuple_of_integers = []
        for vocabulary in self.vocabs:
            if concept in vocabulary:
                tuple_of_integers.append(vocabulary[concept])
            else:
                tuple_of_integers.append(vocabulary["[UNK]"])
        return tuple(tuple_of_integers)

    def construct_vocab_dic(self, level:int)->Dict[str, int]:
        """Construct a vocabulary dictionary for a given level
        level 0: separated by types (lab, medication, diagnosis, procedures, [SEP], [MASK], [UNK],..)
        level 1: topic
        level 2: category
        ...
        """
        # if not 0<=level<=6:
            # raise ValueError("Level must be between 0 and 5")
        if level==0: # separated by types   
            # TODO: include level 0
            all_codes = []
            for prefix in self.code_types:
                all_codes += self.medcodes.get_codes_by_prefix(prefix)
            
            vocab = self.get_type_vocab(all_codes)
        elif level==1: # categories, lab tests, birthmonths, birthyears ...
            vocab = self.get_first_level_vocab()
        else: # assign 0s to special tokens and construct hiearachy for icd and atc
            vocab = self.get_lower_level_vocab(level)
        return vocab
    
    def get_type_vocab(self, all_codes):
        """Uses the temporary vocabulary to assign a category to each code."""
        vocab = {'[ZERO]':0}
        temp_vocab = self.get_temp_vocab_type()
        all_codes += self.special_tokens
        for code in all_codes:
            if code[0] in self.code_types:
                vocab[code] = temp_vocab[code[0]]
            else:
                # special tokens
                vocab[code] = temp_vocab[code.split(']')[0]+']']
        return vocab

    def get_first_level_vocab(self):
        vocab = {'[ZERO]':0}
        all_codes = []
        for prefix in self.code_types:
            all_codes += self.medcodes.get_codes_by_prefix(prefix)
        i = 0
        for code in all_codes:
            if code.startswith('D') or code.startswith('M'):
                vocab[code] = self.topic(code) # only icd and atc codes so far    
            if code.startswith('L'):
                vocab[code] = i +1
                i += 1
        for code in self.special_tokens: # we loop twice through birthyear and birthmonth
            vocab[code] = 0
        return vocab
    
    def get_lower_level_vocab(self, level):
        # Looks good so far
        vocab = {'[ZERO]':0}
        for code_type in self.code_types:
            if code_type=='L':
                for code in self.medcodes.get_lab():
                    vocab[code] = 0 # we placed the labtests on level 1
            prefix = self.medcodes.prefix_dic[code_type]
            getter_func = getattr(self, 'add_'+ prefix +'_to_vocab')
            vocab = getter_func(vocab, level)

        vocab = self.add_special_to_vocab(vocab)
        # TODO: add adm, opr, pro, til, uly, und, lab
        return vocab 

    def get_temp_vocab_type(self):
        """Get a temporary vocab for types of codes e.g. [CLS], [SEX], Diagnoses"""
        temp_keys = [code.split(']')[0]+']' for code in self.special_tokens]
        temp_keys += self.code_types 
        temp_vocab = {token:idx+1 for idx, token in enumerate(temp_keys)}
        return temp_vocab

    def add_icd_to_vocab(self, vocab, level):
        """Add disease codes to vocabulary levels lower than 1"""
        temp_vocab = self.get_temp_vocab_icd(level)
        
        for code in self.medcodes.get_icd():
            if code.startswith(('DU', 'DV')):
                vocab = self.handle_special_disease_codes(code, level, vocab, temp_vocab)
            else:
                if level==2:
                    vocab[code] = temp_vocab[code[:4]]
                else:
                    vocab = self.insert_code(vocab, code, temp_vocab, level+1)
        return vocab

    def add_atc_to_vocab(self, vocab, level):
        """Add medication codes at levels lower than 1"""
        temp_vocab = self.get_temp_vocab_atc(level)
        for code in self.medcodes.get_atc():
            if level==2:
                vocab[code] = temp_vocab[code[2:4]]
            elif level==3 or level==4:
                vocab = self.insert_code(vocab, code, temp_vocab, level+1)
            else:
                vocab = self.insert_code(vocab, code, temp_vocab, [level+1, level+3])
        return vocab

    def add_special_to_vocab(self, vocab):
        """Add special tokens to vocab, at levels lower than 1, we append zeros 
        which will be turned into zero vectors"""
        special_codes = [k for k in self.main_vocab if k.startswith('[')]
        for code in special_codes:
            vocab[code] = 0
        return vocab

    @staticmethod
    def insert_code(vocab, code, temp_vocab, ids):
        """Insert part of the code into the vocabulary"""
        if isinstance(ids, int):
            ids = [ids, ids+1]
        if len(code)>=(ids[1]):
            vocab[code] = temp_vocab[code[ids[0]:ids[1]]]
        else:
            vocab[code] = 0
        return vocab
        
    def handle_special_disease_codes(self, code, level, vocab, temp_vocab):
        """Handle special codes DU, DV"""
        def handle_two_digit_code():
            """Handle special codes followed by (at least) two digits"""
            if level==2: 
                vocab[code] =  self.two_digit_vocab[code[2:4]]
            else:
                if level>=len(code):
                    vocab[code] = 0 # code ends here (leaf node), fill with zeros
                else:
                    vocab[code] = self.alphanumeric_vocab[code[level]] # we fill all level below with zero
        def handle_DUA_DUB_DUH():
            if level==2:
                vocab[code] = self.alphanumeric_vocab[code[2]]
            elif level==3: # next two integers stand for a measure e.g. DUA10 is 10 cm circumference
                vocab[code] = self.two_digit_vocab[code[3:5]]
            else: #DVRA, DVRB, DVRK
                if level>=len(code)-1: # we skip the last digit since level 3 covered two digits
                    vocab[code] = 0 # code ends here (leaf node), fill with zeros
                else:
                    vocab[code] = self.alphanumeric_vocab[code[level+1]] # we fill all level below with zero
        def handle_DVRK01():
            if level==2:
                vocab[code] = self.alphanumeric_vocab[code[3]]
            else:
                vocab[code] = 0
        def handle_DUP_DUT_DVA():
            if level==2:
                vocab[code] = self.alphanumeric_vocab[code[2]]
            elif level==3:
                if self.all_digits(code[3:5]):
                    vocab[code] = self.two_digit_vocab[code[3:5]]
                else:
                    vocab[code] = self.alphanumeric_vocab[code[3]]
            else:
                vocab[code] = 0
        def handle_DVRA_DVRB():
            if level==2:
                vocab[code] = self.alphanumeric_vocab[code[3]] # determine if it is A or B
            elif level==3: # next two integers stand for a measure e.g. DUA10 is 10 cm circumference
                vocab[code] = self.two_digit_vocab[code[4:6]]
            elif level==4: # sometimes followed by a letter
                if len(code)==7:
                    vocab[code] = self.alphanumeric_vocab[code[6]]
                else:
                    vocab[code] = 0
            else:
                vocab[code] = 0
        
        if self.all_digits(code[2:4]):
            handle_two_digit_code()
        elif code.startswith(('DUA', 'DUB', 'DUH')):
            handle_DUA_DUB_DUH()
        elif code=='DVRK01':
            handle_DVRK01()
        elif code.startswith(('DUP', 'DUT', 'DVA')):
            handle_DUP_DUT_DVA()
        elif code.startswith(('DVRA', 'DVRB')):
            handle_DVRA_DVRB()
        else:
            vocab[code] =  temp_vocab[code[:4]]
        return vocab

    def get_temp_vocab_icd(self, level):
        """Construct a temporary vocabulary for categories for icd codes"""
        temp_vocab = {'[ZERO]':0,'[UNK]':1}                
        special_codes_u = ['DUA', 'DUB', 'DUH', 'DUP', 'DUT'] # different birth-related codes
        special_codes_v = ['DVA', 'DVRA', 'DVRB', 'DVRK01'] # placenta weight, height weight ...
        special_codes = special_codes_u + special_codes_v
    
        if level>=3:
            temp_vocab = self.add_alphanumeric_vocab(temp_vocab)
            temp_vocab = self.add_two_digit_vocab(temp_vocab)
            temp_vocab = self.insert_voc('XX', temp_vocab)
            temp_vocab = self.insert_voc('02A', temp_vocab)
            return temp_vocab
        
        for code in self.medcodes.get_icd():
            if code.startswith('DU') or code.startswith('DV'):
                # special codes
                special_code_bool = [code.startswith(s) for s in special_codes]
                if any(special_code_bool):
                    key = special_codes[special_code_bool.index(True)]
                    if level==2:
                        temp_vocab = self.insert_voc(key, temp_vocab) 
                elif code[3].isdigit(): # duration of pregancy DUwwDdd or size of placenta 
                    if level==2:
                        temp_vocab = self.insert_voc(code[:2], temp_vocab)
                else:
                    if level==2:
                        temp_vocab = self.insert_voc(code[:3], temp_vocab)
            else: 
                if level==2:
                    temp_vocab = self.insert_voc(code[:4], temp_vocab)
        
            
        return temp_vocab

    def get_temp_vocab_atc(self, level):
        """Construct a temporary vocabulary for categories for atc codes"""
        temp_vocab = {'[ZERO]':0,'[UNK]':1}                
        if level==2:
            temp_vocab = self.add_two_digit_vocab(temp_vocab)
        elif level==3 or level==4:
            temp_vocab = self.add_alphanumeric_vocab(temp_vocab)
        else:
            temp_vocab = self.add_two_digit_vocab(temp_vocab)
        return temp_vocab

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
    def insert_voc(code, voc):
        """Insert a code into the vocabulary"""
        if code not in voc:
            voc[code] = len(voc)
        return voc
    @staticmethod
    def same_type(codes):
        """Check if all codes are of the same type (ATC or ICD)"""
        # Get the first character of the first string
        first_char = codes[0][0]
        # Iterate over the strings and compare the first character of each string to the first character of the first string
        for code in codes:
            if code[0] != first_char:
                return False
        return True

    @staticmethod
    def all_digits(codes):
        """Check if a string only contains digits"""
        return all([c.isdigit() for c in codes])

    def topic(self, code):
        if code.startswith('M'):
            return self.ATC_topic(code)
        elif code.startswith('D'):
            return self.ICD_topic(code)
        else:
            print(f"Code type starting with {code[0]} not implemented yet")

    @staticmethod
    def ATC_topic(code):
        assert code[0] == 'M', f"ATC code must start with 'M, code: {code}'"
        atc_topic_ls = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
        atc_topic_dic = {topic:(i+1) for i, topic in enumerate(atc_topic_ls)}
        if code[1] in atc_topic_dic:
            return atc_topic_dic[code[1]]
        else:
            return len(atc_topic_ls)+2 #we start at 1, so we need to add 2
    @staticmethod
    def ICD_topic(code):
        assert code[0] == 'D', f"ICD code must start with 'D', code: {code}"
        options = [
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
        ]   
        for i, option in enumerate(options):
            if option[0] <= code[1:4] <= option[1]:
                return i+1
            elif code.startswith("DU"): # special codes (childbirth and pregnancy)
                return len(options)+2
            elif code.startswith("DV"): #weight, height and various other codes
                return len(options)+3
        return len(options)+4
    @staticmethod
    def unique_nodes(h_vocab: Dict[str, Tuple])->bool:
        """Check that all nodes are unique."""
        nodes = []
        for k, v in h_vocab.items():
            if v in nodes:
                print(f'Node {v} is not unique!')
                print([k for k,v in h_vocab.items() if v == nodes[-1]])
                return False
            nodes.append(v)
        return True


class TreeConstructor():
    """Extending SKSVocab to a full tree structure."""
    def __init__(self, main_vocab=None, code_types=None, num_levels=6, test=False, data=None) -> None:
        _, self.sks_vocab = SKSVocabConstructor(main_vocab, code_types, num_levels)()
        if test:
            data_flat = [item for sublist in data['concept'] for item in sublist]
            self.sks_vocab = {k: self.sks_vocab[k] for k in data_flat if k in self.sks_vocab}
            # rand_keys = random.sample(sorted(self.sks_vocab), 10)
            # self.sks_vocab = {k: self.sks_vocab[k] for k in rand_keys}
            # self.sks_vocab = {'A':(1,0,0), 'X':(4,1,1),'B':(2,0,0), 'Aa':(1,1,0), 'Ab':(1,2,0), 'Ba':(2,1,0), 'Bb':(2,2,0), 'Ca':(3,1,0)}
            self.sks_vocab = {'A':(1,0,0,0), 'Aa':(1,1,0,0), 'Ab':(1,2,0,0), 'Ba':(2,1,0,0), 'Bbaa':(2,2,1,1), 'D':(1,2,0,0)}

    def __call__(self)->Tuple[List[Dict[str, Tuple]], pd.DataFrame, pd.DataFrame]:
        self.extended_sks_vocab_ls = self.extend_leafs(self.sks_vocab) # extend leaf nodes to bottom level
        self.full_sks_vocab_ls = self.fill_parents(self.extended_sks_vocab_ls) # fill parents to top level
        self.df_sks_names, self.df_sks_tuples = self.construct_h_table_from_dics(self.full_sks_vocab_ls) # full table of the SKS vocab tree
        return self.full_sks_vocab_ls, self.df_sks_names, self.df_sks_tuples

    def extend_leafs(self, h_dic:Dict[str, Tuple])->List[Dict]:
        """Takes a list of ordered dictionaries, where each dictionary represents nodes on one hierarchy level by tuples
        and extends leafs that are not on the lowest level"""
        tree = self.get_sks_vocab_ls(h_dic) # turn dict of tuples into list of dicts, one for each level
        for level in tqdm(range(len(tree)-1), desc='extending leafs'):
            tree[level+1] = self.extend_one_level(tree[level], tree[level+1], level+1)
        return tree
        
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
        for node1_key, node1 in tqdm(node_dic1.items(), desc='filling parents'):
            parent_node = node1[:node_dic1_level] + (0,)*(n_levels-node_dic1_level)# fill with zeros to the end of the tuple
            if parent_node not in node_dic0.values():
                node_dic0[node1_key] = parent_node
        return node_dic0
    

    def construct_h_table_from_dics(self, tree:List[Dict[str, tuple]])->tuple[pd.DataFrame, pd.DataFrame]:
        """From a list of dictionaries construct two pands dataframes, where each dictionary represents a column
        The relationship of the rows is defined by the tuples in the dictionaries"""
        
        synchronized_ls = self.synchronize_levels(tree)
        lengths = []
        for ls in synchronized_ls:
            lengths.append(len(ls))
        print(lengths)
        inv_tree = [self.invert_dic(dic) for dic in tree]
        df_sks_tuples= pd.DataFrame(synchronized_ls).T
        df_sks_names = df_sks_tuples.copy()
        return df_sks_names, df_sks_tuples
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
    

    def synchronize_levels(self, tree:List[Dict[str, tuple]])->List[List]:
        """Takes a list of ordered dictionaries, where each dictionary represents nodes on one hierarchy level by tuples
        and replicates nodes on one level to match the level below"""
        tree = tree[::-1] # we invert the list to go from the bottom to the top

        dic_bottom = tree[0] # lowest level
        ls_bottom = sorted([v for v in dic_bottom.values()])

        tree_depth = ls_bottom[0].__len__()

        ls_ls_tup = [] 
        ls_ls_tup.append(ls_bottom) # we can append the lowest level as it is
        
        for top_level in tqdm(range(1, len(tree)), desc='Synchronizing levels'): #start from second entry
            dic_top = tree[top_level]
            ls_top = sorted([v for v in dic_top.values()])
            ls_bottom = ls_ls_tup[top_level-1]
            ls_top_new = self.replicate_nodes_to_match_lower_level(ls_top, ls_bottom, tree_depth-top_level)
            ls_ls_tup.append(ls_top_new)
        
        ls_ls_tup = ls_ls_tup[::-1] # we invert the list to go from the bottom to the top
        return ls_ls_tup

    @staticmethod
    def replicate_nodes_to_match_lower_level(nodes0: List[tuple], nodes1:List[tuple], nodes1_level:int)->List[tuple]:
        """Given two lists of nodes on two adjacent levels, replicate nodes of dic0 to match dic1."""
        new_nodes0 = []
        for node1 in nodes1:
            for node0 in nodes0:
                if node0[:nodes1_level] == node1[:nodes1_level]:
                    new_nodes0.append(node0)
        return new_nodes0

    @staticmethod
    def invert_dic(dic:Dict)->Dict:
            return {v:k for k,v in dic.items()}



