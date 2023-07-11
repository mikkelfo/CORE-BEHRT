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
                if isinstance(values, float):           # 0D: For finetuning
                    patient[key] = torch.tensor(values)
                    continue
                elif values.ndim == 1:                  # 1D: For normal pretraining
                    filler = torch.ones(difference, dtype=values.dtype) * -100
                elif values.ndim == 3:                  # 3D: For hierarchical pretraining 
                    filler = torch.ones_like(values) * -100
                    filler = filler[:difference, :, :]
            else:
                filler = torch.zeros(difference, dtype=values.dtype)
            patient[key] = torch.cat((values, filler), dim=0)

    padded_data = {}
    for key in data[0].keys():
        if key == 'target' and data[0]['target'].ndim == 3:                     # For hierarchical pretraining only 
            padded_data[key] = torch.cat([patient[key] for patient in data])    # We give flat labels as we manually mask them in the loss function
        else:
            padded_data[key] = torch.stack([patient[key] for patient in data])

    return padded_data

