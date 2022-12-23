import pandas as pd
import numpy as np
import torch
from datetime import datetime

from tokenizer import EHRTokenizer
from dataset.EHR import EHRDataset
from utils.args import setup_preprocess
from utils.utils import extract_age


# Hardcoded
def extract_features(args):
    """
        ICD and ATC
    """
    icd = pd.read_csv('../../../data/20210526/Aktive_problemliste_diagnoser.csv')
    icd['CONCEPT'] = icd['CURRENT_ICD10_LIST'].map(lambda x: "ICD_" + x[1:])       # [1:] as first letter in code is D (for Diagnosis)
    icd = icd.rename(columns={'Key': 'Key.Patient', 'NOTED_DATE': 'TIMESTAMP'})
    
    atc_admin = pd.read_csv('../../../data/20210526/AdministreretMedicin.csv', encoding="ISO-8859-1")
    atc_admin['CONCEPT'] = atc_admin['ATC'].map(lambda x: "ATCA_" + str(x))
    atc_admin = atc_admin.rename(columns={'TAKEN_TIME': 'TIMESTAMP'})

    atc_ordin = pd.read_csv('../../../data/20210526/OrdineretMedicin.csv', encoding="ISO-8859-1")
    atc_ordin['CONCEPT'] = atc_ordin['ATC'].map(lambda x: "ATCO_" + str(x))
    atc_ordin = atc_ordin.rename(columns={'ORDER_START_TIME': 'TIMESTAMP'})

    combined = pd.concat((
        icd[['Key.Patient', 'CONCEPT', 'TIMESTAMP']], 
        atc_admin[['Key.Patient', 'Key.Indlæggelse', 'CONCEPT', 'TIMESTAMP']], 
        atc_ordin[['Key.Patient', 'CONCEPT', 'TIMESTAMP']]
    ))
    combined = combined[~combined['TIMESTAMP'].isna()]
    combined['TIMESTAMP'] = combined['TIMESTAMP'].map(lambda x: datetime.strptime(x[:10], "%Y-%m-%d"))

    """
        Gender and BMI
    """
    demo = pd.read_csv('../../../data/20210526/Patienter_og_Koen.csv', encoding="ISO-8859-1")
    bmi = demo['Patient.BMI']
    cond = [bmi < 18.5, bmi < 25, bmi < 30, bmi < 35, bmi >= 35, np.isnan(bmi)]
    category = ['underweight', 'normal', 'overweight', 'obese', 'extremely obese', 'UNK']
    demo['Patient.BMI.cat'] = np.select(cond, category)

    demo_lookup = {}
    # Create lookup dict
    for key, patient in demo.groupby('Key.Patient'):
        demo_lookup[key] = patient[['Patient.køn', 'Patient.BMI.cat']].iloc[-1].tolist()

    """
        Age
    """
    age = pd.read_csv('../../../data/20210526/ADT_haendelser.csv', encoding="ISO-8859-1")
    age = age.rename(columns={'Hændelsestidspunkt.Dato.tid': 'TIMESTAMP', 'Patient.alder.ved.Behandlingskontaktens.start': 'AGE'})
    age = age[~age[['TIMESTAMP', 'AGE']].isna().any(1)]
    age['TIMESTAMP'] = age['TIMESTAMP'].map(lambda x: datetime.strptime(x[:10], "%Y-%m-%d"))
    # Mapping timestamp to an age
    age_dict = {}
    for key, patient in age.groupby('Key.Patient'):
        age_dict[key] = (patient['TIMESTAMP'].iloc[0], patient['AGE'].iloc[0])
    # Applying to combined dataframe
    combined['AGE'] = combined.apply(lambda row: extract_age(row, age_dict), axis=1)
    combined = combined[~(combined['AGE'] < 0)]     # Removing wrong ICD/ATC timestamps 


    """
        Start of extraction
    """

    origin_point = datetime(2020, 1, 26)
    patients_info = []
    for key, patient in combined.groupby('Key.Patient'):
        patient = patient.sort_values('TIMESTAMP')

        # Constructing visit segments
        synthetic_counter = 0
        admissions, timestamps = patient['Key.Indlæggelse'].values, patient['TIMESTAMP'].values

        for i in range(len(patient)):
            if pd.isnull(admissions[i]):
                admissions[i] = f"synthetic_visit{synthetic_counter}"
                if i != len(patient)-1 and timestamps[i] != timestamps[i+1]:
                    synthetic_counter += 1
        
        visit_dict = dict((k, i+1) for i, k in enumerate(patient['Key.Indlæggelse'].unique()))

        # Adding background info
        if key in demo_lookup:
            demographic = demo_lookup[key]     # First two tokens are Gender and BMI
        else:
            demographic = ['UNK', 'UNK']

        # Preprending demographic info
        concepts = demographic + patient['CONCEPT'].values.tolist()
        ages = [0, 0] + patient['AGE'].fillna(0).values.tolist()
        abspos = [0, 0] + patient['TIMESTAMP'].map(lambda x: (x - origin_point).days).values.tolist()
        visit_segments = [0, 0] + [visit_dict[adm] for adm in admissions]

        # Packing into event tuples
        events = list(zip(concepts, ages, abspos, visit_segments))

        patients_info.append(events)

    with open(args.data_file, 'wb') as f:
        torch.save(patients_info, f)


def tokenize_data(args):
    with open(args.data_file, 'rb') as f:
        patients_info = torch.load(f)

    """ Tokenize and save """
    tokenizer = EHRTokenizer(args.vocabulary)
    outputs = tokenizer(patients_info)

    tokenizer.save_vocab(args.vocabulary_file)
    with open(args.tokenized_file, 'wb') as f:
        torch.save(outputs, f)


def split_dataset(args):
    with open(args.tokenized_file, 'rb') as f:
        inputs = torch.load(f)

    dataset = EHRDataset(inputs)
    train_set, test_set = dataset.split(args.test_ratio)

    with open(args.train_file, 'wb') as f:
        torch.save(train_set, f)
    with open(args.test_file, 'wb') as f:
        torch.save(test_set, f)


if __name__ == '__main__':
    # Only needs to be run once
    args = setup_preprocess()
    extract_features(args)
    tokenize_data(args)
    split_dataset(args)

