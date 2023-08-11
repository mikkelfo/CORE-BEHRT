from torch.utils.data import IterableDataset

def check_patient_counts(concepts, patients_info, logger):
    if concepts.PID.nunique() != patients_info.PID.nunique():
            logger.warning(f"patients info contains {patients_info.PID.nunique()} patients != \
                        {concepts.PID.nunique()} unique patients in concepts")

class ConcatIterableDataset(IterableDataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.pids = [pid for dataset in datasets for pid in dataset.pids]
        self.file_ids = [file_id for dataset in datasets for file_id in dataset.file_ids]
    def __iter__(self):
        for dataset in self.datasets:
            yield from dataset
    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])