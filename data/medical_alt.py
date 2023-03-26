from os.path import dirname, join
import pickle as pkl
from typing import List, Dict, Tuple
import string

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



class SKSVocabConstructor():
    """
    Construct two vocabularies for medical codes, mapping to integers and tuples (nodes).
    Every integer of the tuple specifies a branch on a level. Integer 0 is reserved for empty node.
    Currently we have the hierarchy for medication and diagnosis implemented. 
    """
    def __init__(self, main_vocab=None, code_types=['D', 'M'], num_levels=7):
        """
        main_vocab: initial vocabulary, if None, create a new one
        code_types: list of code types to include in the vocabulary (by prefix D, M, L)
        num_levels: number of levels in the hierarchy, don't change this if you don't know what you are doing.
        """

        self.code_types = code_types
        for code_type in self.code_types:
            if code_type not in ['D', 'M', 'L']:
                raise ValueError(f'Hierarchy for type {code_type} not implemented yet.') 

        self.num_levels=num_levels
        self.vocabs = [{'[ZERO]':0} for _ in range(num_levels)] # this will be returned

        self.medcodes = MedicalCodes()

        if isinstance(main_vocab, type(None)):
            self.special_codes = ['[CLS]', '[PAD]', '[SEP]', '[MASK]', '[UNK]', '[BG_Mand]', '[BG_Kvinde]']
            self.main_vocab = {token: idx for idx,
                               token in enumerate(self.special_codes)}
        else:
            self.main_vocab = main_vocab
            self.special_codes = [k for k in main_vocab if k.startswith('[')]

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
        self.atc_topic_ls = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']

        self.topics = {'D':self.icd_topic_options, 'M':self.atc_topic_ls}

        self.alphanumeric_vocab = self.add_alphanumeric_vocab()
        self.two_digit_vocab = self.add_two_digit_vocab()


        self.abc123_voc = self.add_two_letter_vocab(self.add_two_digit_vocab(self.alphanumeric_vocab)) # complete vocabulary with 1 and 2 letter codes

    def __call__(self)->Tuple[Dict[str, int], Dict[str, Tuple[int]]]:
        """Return vocab, mapping concepts to tuples, where each tuple element is a code on a level
        The dictionares contain concept present in the SKS code and the ones inmain vocab.
        types contains the types of codes to be included in the vocabulary, e.g. ['D', 'M', 'L']"""
        
        self.handle_level_one() # add special tokens/code types to the first level
       
        for code_type in self.code_types: # Loop over disease, medication, lab_codes,...
            self.handle_codetype(code_type)
        return None, self.vocabs
        # for codes which exist only in the top n levels (e.g. type codes, topics codes), 
        # we add zeros to lower levels
        self.fill_with_zeros() 
        return None, self.vocabs
        h_vocab = {} # this is the dictionary of tuples (first vocab contains all the concepts)
        for concept in self.vocabs[0]:
            h_vocab[concept] = self.map_concept_to_tuple(concept)
        
        # if not self.unique_nodes(h_vocab):
            # raise ValueError('Not all nodes are unique')
        
        return self.main_vocab, h_vocab

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

    def add_icd_vocabs(self)->None:
        """Add ICD codes to the vocabulary"""
        for topic in self.icd_topic_options:
            self.add_topic(topic, 'D')
        for code in self.medcodes.get_codes_by_prefix('D'):
            self.vocabs[1][code] = self.get_topic(code)

        self.add_category('D') # A00, A01, ..., Z99
        self.add_lower_levels_icd() 
        return None

    def add_lower_levels_icd(self):
        for code in self.medcodes.get_codes_by_prefix('D'):
            if code.startswith(('DU', 'DV')):
                pass
                    # vocab = self.handle_special_disease_codes(code, level, vocab, temp_vocab)
    
    def add_atc_vocabs(self)->None:
        """Add ATC codes to the vocabulary"""
        for topic in self.atc_topic_ls:
            self.add_topic(topic, 'M')
        self.add_category('M') # A, B, C, ..., V
        return None

    # Topics
    def add_topic(self, topic:Tuple[str, str], code_type:str):
        """Add all codes from a topic to the vocabulary"""
        level_one_vocab, level_two_vocab = self.vocabs[:2]

        for topic in self.topics[code_type]:
            if code_type == 'D':
                topic_name = f"D{topic[0]}-D{topic[1]}"
            elif code_type == 'M':
                topic_name = f"M{topic}"
            level_one_vocab[topic_name] = level_one_vocab[code_type]
            level_two_vocab[topic_name] = self.get_topic(topic_name)
        self.vocabs[:2] = [level_one_vocab, level_two_vocab]

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
        atc_topic_dic = {topic:(i+1) for i, topic in enumerate(self.atc_topic_ls)}
        if code[1] in atc_topic_dic:
            return atc_topic_dic[code[1]]
        else:
            return len(self.atc_topic_ls) + 2 #we start at 1, so we need to add 2
    
    def add_category(self, type:str):
        """Add ICD categories to the vocabulary"""
        for code in self.medcodes.get_codes_by_prefix(type):
            if code.startswith(('DU', 'DV')):
                self.handle_special_icd_codes(code)
                continue
            category = code[:4] # e.g. DA00, DA01, ..., DZ99 / MA00, MA01, ..., MZ99
            category_ints = code[2:]
            if category not in self.vocabs[2]:
                self.vocabs[2][category] = self.two_digit_vocab[category_ints]
                self.vocabs[1][category] = self.get_topic(category)
                self.vocabs[0][category] = self.vocabs[0][type]

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
            cat_vocab[code] = string.ascii_uppercase.index(code[2])
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
            self.vocabs[4][code] = self.two_digit_vocab[code[4:6]] # last level
    
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
    def insert_code(code, voc):
        """Insert a code into the vocabulary"""
        if code not in voc:
            voc[code] = len(voc)
        return voc
    
    @staticmethod
    def insert_code_part(vocab, code, temp_vocab, ids):
        """Insert part of the code into the vocabulary"""
        if isinstance(ids, int):
            ids = [ids, ids+1]
        if len(code)>=(ids[1]):
            vocab[code] = temp_vocab[code[ids[0]:ids[1]]]
        else:
            vocab[code] = 0
        return vocab

    @staticmethod
    def all_digits(codes):
        """Check if a string only contains digits"""
        return all([c.isdigit() for c in codes])
    @staticmethod
    def get_unused_value(vocab):
        """Get the first unused value in a vocabulary"""
        return max(vocab.values())+1

    def get_DU_two_digit_codes_voc(self)->Dict[str, int]:
        """Get a vocabulary for the two digit codes in the DUiii branch """
        duii_ls = list(set([code for code in self.medcodes.get_icd() if code.startswith('DU') and self.all_digits(code[2:4])]))
        return {code: idx+1 for idx, code in enumerate(duii_ls)}

    def get_DV_four_digit_codes_voc(self)->Dict[str, int]:
        """Get a vocabulary for the four digit codes in the DV branch """
        dv4int_ls = list(set([code for code in self.medcodes.get_icd() if code.startswith('DV') and self.all_digits(code[2:6])]))
        return {code: idx+1 for idx, code in enumerate(dv4int_ls)}
