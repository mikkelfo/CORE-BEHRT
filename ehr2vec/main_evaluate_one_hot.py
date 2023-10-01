import os
from os.path import join

import torch
import pandas as pd
from common.config import load_config, instantiate, get_function
from common.loader import DatasetPreparer
from common.setup import azure_onehot_setup, get_args, setup_logger
from evaluation.utils import get_pos_weight, evaluate_predictions

args = get_args('evaluate_one_hot.yaml')
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)

def main_finetune():
    cfg = load_config(config_path)

    cfg.paths.output_path = cfg.paths.finetune_features_path
    cfg, _, mount_context = azure_onehot_setup(cfg)

    model = instantiate(cfg.model)
    model_name = model.__class__.__name__

    run_folder=join(cfg.paths.output_path, f"{model_name}_{cfg.paths.run_name}")
    os.makedirs(run_folder, exist_ok=True)
    logger = setup_logger(run_folder)
    cfg.save_to_yaml(join(run_folder, 'evaluate_one_hot.yaml'))
    

    X_train, y_train, X_val, y_val = DatasetPreparer(cfg).prepare_onehot_features()
    
    logger.info('Instantiating Model')
    sample_weight = get_pos_weight(cfg, y_train) if cfg.trainer_args.get('sample_weight', False) else None
    
    model.fit(X_train, y_train, sample_weight=sample_weight,)
    pred_probas = model.predict_proba(X_val)[:, 1]
    metrics = [get_function(metric) for metric in cfg.metrics.values()]
    results = evaluate_predictions(y_val, pred_probas, metrics)
    torch.save(model, join(run_folder, 'model.pth'))
    logger.info('Saving results')
    results_df = pd.DataFrame(results, index=[0])
    results_df.to_csv(join(run_folder, 'results.csv'), index=False)
    
    if cfg.env=='azure':
        # from azure_run import file_dataset_save
        #file_dataset_save(local_path=join('outputs', cfg.paths.run_name), datastore_name = "workspaceblobstore",
         #           remote_path = join("PHAIR", cfg.paths.model_path, cfg.paths.run_name))
        mount_context.stop()
    logger.info('Done')


if __name__ == '__main__':
    main_finetune()