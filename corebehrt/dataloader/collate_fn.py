import torch

def dynamic_padding(data: list)->dict:
    max_len = max([len(patient["concept"]) for patient in data])
    for patient in data:
        difference = max_len - len(patient["concept"])
        for key, values in patient.items():
            if key in ["target"]:
                if isinstance(values, float):  # 0D: For finetuning
                    patient[key] = torch.tensor(values)
                    continue
                elif values.ndim == 1:  # 1D: For normal pretraining
                    filler = torch.ones(difference, dtype=values.dtype) * -100
            else:
                filler = torch.zeros(difference, dtype=values.dtype)
            patient[key] = torch.cat((values, filler), dim=0)

    padded_data = {}
    for key in data[0].keys():
        padded_data[key] = torch.stack([patient[key] for patient in data])

    return padded_data

