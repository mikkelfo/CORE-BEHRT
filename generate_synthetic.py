from datetime import datetime
from os.path import dirname, join, realpath
from random import sample

import numpy as np
import torch
import typer

from data import medical
from typing import Dict
from collections import defaultdict


class DataGenerator(super):
    def __init__(self, num_patients, min_num_visits, max_num_visits, 
        min_num_codes_per_visit, max_num_codes_per_visit, min_los, 
        max_los, num_atc_codes, num_icd_codes, num_lab_test=500, age_lower_bound=0, seed=42,
        test=True,
        start_date=datetime(2010, 1, 1)):
        """
        Simulates data as lists:
            [pid, los_ls, all_visit_codes, visit_nums]
        min_los, max_los: Length os stay in the hospital,
        num_codes: total number of ICD10 codes to generate
        """
        self.num_patients = num_patients
        self.min_num_codes_per_visit = min_num_codes_per_visit
        self.max_num_codes_per_visit = max_num_codes_per_visit
        self.min_num_visits = min_num_visits
        self.max_num_visits = max_num_visits
        self.min_los = min_los
        self.max_los = max_los
        self.num_icd_codes = num_icd_codes
        self.num_atc_codes = num_atc_codes
        self.age_lower_bound = age_lower_bound
        self.num_lab_tests = num_lab_test
        self.start_date = start_date
        self.rng = np.random.default_rng(seed)
        self.test = test

    def __call__(self) -> Dict:
        """Generates a dictionary which contains concepts, visits, ages and absolute positions"""
        data = defaultdict(list)
        for _ in range(self.num_patients):
            concepts, visits, ages, absolute_position = self.generate_patient_history()
            data['concept'].append(concepts)
            data['segment'].append(visits)
            data['age'].append(ages)
            data['abspos'].append(absolute_position)
        return data

    # generate atc codes
    def generate_patient_history(self):
        """Generates a dictionary which contains sex, ages, length of stay, codes, lab tests, lab tests visits"""
        num_visits = self.rng.integers(self.min_num_visits, self.max_num_visits)
        num_codes_per_visit_ls = self.rng.integers(self.min_num_codes_per_visit, 
            self.max_num_codes_per_visit, 
            size=num_visits) # should icd and atc vectors point in different directions
        los = self.rng.integers(self.min_los, self.max_los, size=num_visits)\
            .tolist()
        los = np.repeat(los, num_codes_per_visit_ls).tolist()
        medcodes = medical.MedicalCodes(test=self.test)
        icd_codes = sample(medcodes.get_codes_by_prefix('D'), self.num_icd_codes)
        atc_codes = sample(medcodes.get_codes_by_prefix('M'), self.num_atc_codes)
        # lab_tests = sample(medcodes.get_lab(), self.num_lab_tests)

        codes = icd_codes + atc_codes #+ lab_tests
        # values = [1]*(len(icd_codes)+len(atc_codes)) + self.rng.normal(size=len(lab_tests)).tolist()
        idx = self.rng.choice(np.arange(len(codes)), np.sum(num_codes_per_visit_ls), replace=True)
        codes = np.array(codes)[idx].tolist()
        # values = np.array(values)[idx].tolist()
        visit_nums = np.arange(1, num_visits+1) # should start with 1!
        visits = np.repeat(visit_nums, num_codes_per_visit_ls).tolist()
        
        birthdate = self.generate_birthdate()
        ages = self.generate_ages(num_visits, birthdate) # pass age as days or rounded years?
        ages = np.repeat(ages, num_codes_per_visit_ls).tolist()
        absolute_position = self.generate_absolute_position(ages, los, birthdate) # in days
        sex_int = self.generate_sex()
        if sex_int == 1:
            sex = 'BG_MALE'
        else:
            sex = 'BG_FEMALE'
        concepts = [sex, '[SEP]'] + codes
        ages = [0,0] + ages
        visits = [0,0] + visits
        absolute_position = [0,0] + absolute_position

        return concepts, visits, ages,  absolute_position


    def generate_ages(self, num_visits, birthdate):
        ages = []
        age_lower_bound = self.age_lower_bound
        age_upper_bound = int(((datetime.today() - birthdate).days)/365)
        random_age = self.rng.integers(age_lower_bound, age_upper_bound)
        for _ in range(num_visits):
            if random_age > age_upper_bound:
                random_age = age_upper_bound
            ages.append(random_age)
            age_lower_bound = random_age 
            random_age = self.rng.poisson(2, 1)[0] + random_age
        return ages
        
    def generate_absolute_position(self, ages, los_ls, birthdate):
        absolute_positions = []
        birthdate_difference = self.start_date - birthdate
        new_age = ages[0]
        days_since_start = new_age*365-birthdate_difference.days
        additional_days = 0
        for age, los in zip(ages, los_ls):
            if age != new_age: 
                days_since_start  = age*365-birthdate_difference.days
                new_age = age
                additional_days = 0
            additional_days += self.rng.poisson(int(los/2),1)[0]
            
            if additional_days > los:
                additional_days = los
            days_since_start = days_since_start + additional_days
            
            absolute_positions.append(days_since_start)
            
        return absolute_positions

    def generate_sex(self):
        return self.rng.binomial(1, 0.5)

    def generate_birthdate(self):
        year = self.rng.integers(1900, 1980)
        month = self.rng.integers(1, 12)
        day = self.rng.integers(1, 28)
        return datetime(year, month, day)

    def simulate_data(self):
        for pid in range(self.num_patients):
            yield self.generate_patient_history('p_'+str(pid))


def main(save_name: str = typer.Option('synthetic', 
        help="name of the file to save the data to, will be saved as pkl"),
        num_patients : int = typer.Option(100), 
        min_num_visits: int = typer.Option(1),
        max_num_visits: int = typer.Option(10),
        min_num_codes_per_visit: int = typer.Option(1),
        max_num_codes_per_visit: int = typer.Option(5),
        min_los: int = typer.Option(1),
        max_los: int = typer.Option(30),
        num_atc_codes: int = typer.Option(1000),
        num_icd_codes: int = typer.Option(1000),
        num_lab_tests: int = typer.Option(1000),
        seed: int = typer.Option(42),
        test: bool = typer.Option(True)):
    generator = DataGenerator(num_patients, min_num_visits, max_num_visits, 
        min_num_codes_per_visit, max_num_codes_per_visit, 
        min_los, max_los, num_atc_codes, num_icd_codes, num_lab_tests, seed=seed, test=test)

    base_dir = dirname(dirname(realpath(__file__)))
    save_path = join(base_dir, 'data', 'sequence', 'synthetic' ,save_name + '.pt')
    torch.save(generator(), save_path)
if __name__ == '__main__':
    typer.run(main)