import torch

def static(data: list):
    padded_data = {
        key: torch.stack([torch.tensor(patient[key]) for patient in data])
        for key in data[0].keys()
    }

    return padded_data


def dynamic_padding(data: list):
    max_len = max([len(patient['concept']) for patient in data])
    for patient in data:
        difference = max_len - len(patient['concept'])
        for key, values in patient.items():
            if key in ['age', 'abspos']:
                dtype = torch.float32
            else:
                dtype = torch.long
            patient[key] = torch.cat((torch.tensor(values, dtype=dtype), torch.zeros(difference, dtype=dtype)), dim=0)
    
    padded_data = {
        key: torch.stack([patient[key] for patient in data])
        for key in data[0].keys()
    }

    return padded_data

