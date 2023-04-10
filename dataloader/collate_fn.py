import torch

def static(data: list):
    padded_data = {
        key: torch.stack([torch.tensor(patient[key]) for patient in data])
        for key in data[0].keys()
    }

    return padded_data


def dynamic_padding(data: list, hierarchical: bool = False):
    max_len = max([len(patient['concept']) for patient in data])
    for patient in data:
        difference = max_len - len(patient['concept'])
        for key, values in patient.items():
            if key in ['age', 'abspos']:
                dtype = torch.float32
            else:
                dtype = torch.long
            
            if key != 'target':
                patient[key] = torch.cat((torch.tensor(values, dtype=dtype), torch.zeros(difference, dtype=dtype)), dim=0)
            else:
                targets = torch.tensor(values, dtype=dtype)
                if hierarchical:
                    patient[key] = torch.cat((targets, -100*torch.ones(size=(difference, targets.shape[1]), dtype=dtype)), dim=0)
                else:
                    patient[key] = torch.cat((targets, -100*torch.ones(size=(difference,), dtype=dtype)), dim=0)
    
    padded_data = {
        key: torch.stack([patient[key] for patient in data])
        for key in data[0].keys()
    }

    return padded_data

