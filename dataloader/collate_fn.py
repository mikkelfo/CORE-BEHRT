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
            if key == 'target':
                if isinstance(values, float):
                    patient[key] = torch.tensor(values)
                    continue
                filler = torch.ones(difference, dtype=values.dtype) * -100
            else:
                filler = torch.zeros(difference, dtype=values.dtype)
            patient[key] = torch.cat((values, filler), dim=0)
    
    padded_data = {
        key: torch.stack([patient[key] for patient in data])
        for key in data[0].keys()
    }

    return padded_data

