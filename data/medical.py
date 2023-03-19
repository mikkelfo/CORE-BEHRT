import pickle as pkl
import string

import pandas as pd

"""  ("UA","UA"), # Abdominal Circumference [cm]
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
def sks_codes_to_list():
    """Convert the SKScomplete list of codes to a list of codes in pickles format."""
    codes = []
    with open( "SKScomplete.txt") as f:
        for line in f:
            codes.append(line.split(' ')[0])
    codes = set(codes)
    with open("SKScodes.pkl", "wb") as f:
        pkl.dump(codes, f)

def npu_codes_to_list():
    """Convert the NPUlistEN221222.csv to a list of codes in pickles format."""
    df = pd.read_csv('NPUlistEN221222.csv', 
        encoding='latin-1', delimiter=';', usecols=['NPU code'])
    codes = df['NPU code'].unique().tolist()
    # replace DNK stands for local danish NPU codes
    codes = [c.replace('DNK', 'labL').replace('NPU', 'labL') for c in codes]
    with open( "NPUcodes.pkl", "wb") as f:
        pkl.dump(codes, f)

class MedicalCodes():
    def __init__(self):
        with open("SKScodes.pkl", "rb") as f:
            self.sks_codes = list(pkl.load(f))
        with open("NPUcodes.pkl", "rb") as f:
            self.npu_codes = pkl.load(f) 
        self.codes = self.npu_codes + self.sks_codes

    def get_codes_type(self, signature, min_len=2):
        codes =[c.strip(signature) for c in self.codes if c.startswith(signature)]
        return [c for c in codes if len(c)>=min_len]
    def get_lab(self):
        return sorted(self.get_codes_type('lab'))
    def get_icd(self):
        return sorted(self.get_codes_type('dia', min_len=4))
    def get_atc(self):
        codes = self.get_codes_type('atc', min_len=4)
        codes[codes.index('N05CC10')] = 'MZ99' # thalidomid, wrongly generated code will be assigned a special token
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
    """Construct a vocabulary for SKS codes.
    Returns a dictionary mapping SKS codes to tuples, 
    where every entry in the tuple specifies the branch on a level."""
    def __init__(self, main_vocab=None, additional_types=None, num_levels=6):
        self.medcodes = MedicalCodes()
        self.codes = self.medcodes.codes

        if isinstance(main_vocab, type(None)):
            self.special_tokens = ['[CLS]', '[PAD]', '[SEP]', '[MASK]', '[UNK]', '[BG_Mand]', '[BG_Kvinde]']
            self.main_vocab = {token: idx for idx,
                               token in enumerate(self.special_tokens)}
        else:
            self.main_vocab = main_vocab
            self.special_tokens = [k for k in main_vocab if k.startswith('[')]
            
        self.vocabs = []

        if isinstance(additional_types, type(None)):
            self.additional_types=['D', 'M', 'L']
        self.num_levels = num_levels
    def __call__(self):
        """Return vocab, mapping concepts to tuples, where each tuple element is a code on a level"""
        tuple_vocab = {}
        for level in range(self.num_levels):
            self.vocabs.append(self.construct_vocab_dic(level))
        for concept in self.vocabs[0]:
            tuple_vocab[concept] = self.map_concept_to_tuple(concept)
        return tuple_vocab

    def map_concept_to_tuple(self, concept):
        """Using the list of vocabs, map a concept to a tuple of integers"""
        tuple_of_integers = []
        for vocabulary in self.vocabs:
            if concept in vocabulary:
                tuple_of_integers.append(vocabulary[concept])
            else:
                tuple_of_integers.append(vocabulary["[UNK]"])
        return tuple(tuple_of_integers)
    
    def get_birthmonth(self): # needs to be time2vec later
        return [k for k in self.main_vocab if k.startswith('[BIRTHMONTH]')]
    def get_birthyear(self): # needs to be time2vec later
        return [k for k in self.main_vocab if k.startswith('[BIRTHMONTH]')]

    def construct_vocab_dic(self, level):
        """construct a dictionary of codes and their topics"""
        if not 0<=level<=5:
            raise ValueError("Level must be between 0 and 5")
        if level==0: # separated by types   
            # TODO: include level 0
            all_codes = self.medcodes.get_icd()+self.medcodes.get_atc()\
                +self.medcodes.get_lab()\
                +self.get_birthyear()+self.get_birthmonth()
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
                if code[0] in self.additional_types:
                    vocab[code] = temp_vocab[code[0]]
                else:
                    # special tokens
                    vocab[code] = temp_vocab[code.split(']')[0]+']']
        return vocab

    def get_first_level_vocab(self):
        vocab = {'[ZERO]':0}
        for code in self.medcodes.get_atc()+self.medcodes.get_icd():
            vocab[code] = self.topic(code) # only icd and atc codes so far
        for i, code in enumerate(self.medcodes.get_lab()):
            vocab[code] = i+1
        for code in self.special_tokens: # we loop twice through birthyear and birthmonth
            vocab[code] = 0
        for i, code in enumerate(self.get_birthyear()):
            vocab[code] = i+1
        for i, code in enumerate(self.get_birthmonth()):
            vocab[code] = i+1
        return vocab
    
    def get_lower_level_vocab(self, level):
        # Looks good so far
        vocab = {'[ZERO]':0}
        for code in self.medcodes.get_lab():
            vocab[code] = 0
        vocab = self.add_icd_to_vocab(vocab, level)
        vocab = self.add_atc_to_vocab(vocab, level)
        vocab = self.add_special_to_vocab(vocab)
        # TODO: add adm, opr, pro, til, uly, und, lab
        return vocab 

    def get_temp_vocab_type(self):
        """Get a temporary vocab for types of codes e.g. [CLS], [SEX], Diagnoses"""
        temp_keys = [code.split(']')[0]+']' for code in self.special_tokens]
        temp_keys += self.additional_types 
        temp_vocab = {token:idx for idx, token in enumerate(temp_keys)}
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
        if code[2].isdigit():
            # special code followed by two digits 
            if level==2: 
                vocab[code] =  temp_vocab[code[:2]]
            else:
                vocab[code] = 0 # we fill all level below with zero
        elif code.startswith(('DUA', 'DUB', 'DUH')):
            if level==2:
                vocab[code] = temp_vocab[code[:3]]
            else: #DVRA, DVRB, DVRK
                vocab[code] = 0 # we fill all level below with zeros
        elif code=='DVRK01':
            if level==2:
                vocab[code] = temp_vocab[code]
            else:
                vocab[code] = 0
        elif code.startswith(('DUP', 'DUT', 'DVA')):
            if level==2:
                vocab[code] = temp_vocab[code[:3]]
            elif level==3:
                if code[3].isdigit():
                    vocab[code] = temp_vocab[str(int(code[3:]))] # digits
                else:
                    vocab[code] = temp_vocab[code[3:]]
            else:
                vocab[code] = 0
        elif code.startswith(('DVRA', 'DVRB')):
            if level==2:
                vocab[code] = temp_vocab[code[:4]]
            elif level==3:
                if  self.all_digits(code[4:]):
                    vocab[code] = temp_vocab[str(int(code[4:]))] # digits
                else:
                    vocab[code] = temp_vocab[code[4:]] #TODO: check this
            else:
                vocab[code] = 0
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
            temp_vocab = self.alphanumeric_vocab(temp_vocab)
            temp_vocab = self.two_digit_vocab(temp_vocab)
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
            temp_vocab = self.two_digit_vocab(temp_vocab)
        elif level==3 or level==4:
            temp_vocab = self.alphanumeric_vocab(temp_vocab)
        else:
            temp_vocab = self.two_digit_vocab(temp_vocab)
        return temp_vocab

    @staticmethod
    def two_digit_vocab(temp_vocab):
        for i in range(10):
                for j in range(10):
                    temp_vocab[str(i)+str(j)] = len(temp_vocab)
        return temp_vocab
        
    @staticmethod
    def alphanumeric_vocab(temp_vocab):
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


