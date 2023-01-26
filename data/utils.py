import pandas as pd
import numpy as np
from hydra.utils import instantiate
from datetime import datetime


def load_dataframe(info: dict):
    # Unpack into dict and instantiate functions
    converters = info.get('converters', None)
    if converters is not None:
        converters = {column: instantiate(func) for column, func in converters.items()}

    # Convert from ListConfig to python list
    date_columns = info.get('date_columns', False)
    if date_columns:
        date_columns = list(date_columns)

    df = pd.read_csv(
        # User defined
        info['filename'],
        converters=converters,
        usecols=info.get('usecols', None),
        names=info.get('names', None),
        parse_dates=date_columns,
        # Defaults
        encoding='ISO-8859-1',
        skiprows=[0],
        header=None,
    )
    return df

def calculate_age(patient, birthdates):
    if patient['Key.Patient'] in birthdates:
        return (patient['TIMESTAMP'] - birthdates[patient['Key.Patient']]['BIRTHDATE']).year
    else:
        return np.nan

def categorize_bmi(bmi):
    try:
        bmi = float(bmi)
    except ValueError:
        return '[UNK]'
    
    if bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'normal'
    elif bmi < 30:
        return 'overweight'
    elif bmi < 35:
        return 'obese'
    else:
        return 'extremely obese'

def construct_segment_dict(key, admission, segment_dict):
    if key not in segment_dict:
        segment_dict[key] = {}
    if admission not in segment_dict[key]:
        segment_dict[key][admission] = len(segment_dict[key]) + 1 

def save_to_features(features, patient):
    for column in patient:
        features[column].append(patient[column].tolist())


def create_concept_features(concept_info: dict):
    concepts = [load_dataframe(info) for info in concept_info.values()]
    concepts = pd.concat(concepts)
    concepts = concepts[~concepts['TIMESTAMP'].isna()]
    concepts = concepts.sort_values('TIMESTAMP')
    return concepts

def create_age_features(age_info: dict, concepts: pd.DataFrame):
    ages = load_dataframe(age_info)
    # Filling in NaN values
    if ages.covid_tests:
        tests = load_dataframe(age_info.covid_tests).drop_duplicates('Key.Admission')
        test_dict = {k: v for k,v in tests.values}  # Create dict of (Key.Admission: timestamp)
        age_notime = ages[ages['TIMESTAMP'].isna()] # We only want to fill in the NaN times
        new_timestamps = age_notime['Key.Patient'].map(lambda key: test_dict.get(key, np.nan))
        ages.loc[new_timestamps, 'TIMESTAMP'] = new_timestamps.values   # Overwrite NaN values with covid test timestamps

    ages = ages.dropna().drop_duplicates('Key.Patient', keep='last')    # Dropping duplicates (we only need one birthdate per patient)
    ages['BIRTHDATE'] = ages['TIMESTAMP'] - ages['AGE'].map(lambda years: pd.Timedelta(years*365.25, 'D'))  # Calculate an approximate birthdate from their current age
    birthdates = ages.set_index('Key.Patient', verify_integrity=False).to_dict('index')     # Create dict of (Key, birthdate)
    return concepts.apply(lambda patient: calculate_age(patient, birthdates), axis=1)

def create_abspos_features(abspos_info: dict, concepts: pd.DataFrame):
    year, month, day = abspos_info['year'], abspos_info['month'], abspos_info['day']
    origin_point = datetime(year, month, day)
    return (concepts['TIMESTAMP'] - origin_point).days

def create_segment_features(segment_info: dict, concepts: pd.DataFrame):
    mask = concepts['Key.Admission'].isna()
    concepts.loc[mask, 'Key.Admission'] = concepts[mask]['TIMESTAMP'].map(lambda timestamp: f'visit_{timestamp}').values

    segment_dict = {}
    concepts.apply(lambda entry: construct_segment_dict(entry['Key.Patient'], entry['Key.Admission'], segment_dict), axis=1)

    return concepts.apply(lambda entry: segment_dict[entry['Key.Patient']][entry['Key.Admission']], axis=1)

def create_demographics(demographics_info: dict, ages=False, abspos=False, segments=False):
    demo = load_dataframe(demographics_info)
    background = {
        'Key.Patient': demo['Key.Patient'].tolist() * 2,
        'CONCEPTS': demo['GENDER'].tolist() + demo['BMI'].tolist()
    }

    # Set optional features to 0 if present
    if ages: background['age'] = 0
    if abspos: background['abspos'] = 0
    if segments: background['segment'] = 0

    demographics = pd.DataFrame(background)
    return demographics