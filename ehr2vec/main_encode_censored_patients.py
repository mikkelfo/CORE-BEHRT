import os
from os.path import join

from common.config import load_config, Config
from common import azure
from common.setup import prepare_encodings_directory
from common.loader import create_binary_outcome_datasets, load_model

from model.model import BertEHREncoder
from evaluation.encodings import Forwarder
from common.utils import ConcatIterableDataset


config_path = join("configs", "encode_censored_test.yaml")
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

run_name = "encode_censored_patients"

def get_output_path_name(dataset, cfg):
    num_patients = str(int((len(dataset))/1000))+'k'
    if cfg.outcome.censor_type:
        days = True if abs(cfg.outcome.n_hours)>48 else False
        window = int(abs(cfg.outcome.n_hours/24)) if days else abs(cfg.outcome.n_hours)
        days_hours = 'days' if days else 'hours'
        pre_post = 'pre' if cfg.outcome.n_hours<0 else 'post'
        return f"{cfg.outcome.type}_Patients_{num_patients}_Censor_{window}{days_hours}_{pre_post}_{cfg.outcome.censor_type}"
    else:
        if cfg.outcome.type:
            return f"{cfg.outcome.type}_Patients_{num_patients}_Uncensored"
        else:
            return f"Patients_{num_patients}_Uncensored"

def main_encode():
    encodings_file_name = 'encodings.h5'
    cfg = load_config(config_path)
    run = None
    model_path = cfg.paths.model_path
    censored_patients_path = join(model_path, 'encodings', 'censored_patients')

    cfg.output_dir = censored_patients_path
    if cfg.env=='azure':
        run, mount_context = azure.setup_azure(cfg.paths.run_name)
        cfg.paths.data_path = join(mount_context.mount_point, cfg.paths.data_path)
        cfg.paths.model_path = join(mount_context.mount_point, cfg.paths.model_path)
        cfg.paths.outcomes_path = join(mount_context.mount_point, cfg.paths.outcomes_path)
        cfg.output_dir = 'outputs'
    
    output_dir = cfg.output_dir # we will modify cfg. output_dir

    for outcome in cfg.outcomes:
        cfg.outcome = cfg.outcomes[outcome]
        
        train_dataset, val_dataset, _ = create_binary_outcome_datasets(cfg)
        
        if isinstance(train_dataset, type(None)) or isinstance(val_dataset, type(None)):
            dataset = train_dataset if train_dataset is not None else val_dataset
        else:
            dataset = ConcatIterableDataset([train_dataset, val_dataset])
        output_path_name = get_output_path_name(dataset, cfg)
        cfg.output_dir = join(output_dir, output_path_name)
        logger = prepare_encodings_directory(config_path, cfg)
        logger.info(f"Access data from {cfg.paths.data_path}")
        logger.info(f"Access outcomes from {cfg.paths.outcomes_path}")
        logger.info(f'Outcome name: {cfg.outcome.type}')
        logger.info(f'Censor name: {cfg.outcome.censor_type}')
        logger.info(f"Censoring {cfg.outcome.n_hours} hours after censor_outcome")
        logger.info(f"Store in directory with name: {get_output_path_name(dataset, cfg)}")

        logger.info('Initializing model')
        model = load_model(BertEHREncoder, cfg)

        forwarder = Forwarder( 
            model=model, 
            dataset=dataset, 
            run=run,
            logger=logger,
            output_path=join(cfg.output_dir, encodings_file_name),
            **cfg.forwarder_args,
        )
        forwarder.forward_patients()

        if cfg.env=='azure':
            from azure_run import file_dataset_save
            file_dataset_save(local_path='outputs', datastore_name = "workspaceblobstore",
                        remote_path = join("PHAIR", censored_patients_path), name="censored_patients")
    if cfg.env=='azure':
        mount_context.stop()

        
if __name__ == '__main__':
    main_encode()