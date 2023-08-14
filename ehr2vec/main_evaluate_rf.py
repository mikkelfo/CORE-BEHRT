import os
from os.path import join, split
from shutil import copyfile

import pandas as pd
from common.azure import setup_azure
from common.config import get_function, load_config
from common.io import PatientHDF5Reader
from common.logger import TqdmToLogger
from common.setup import setup_logger, get_args
from evaluation.utils import get_mean_std
from evaluation.optimize import find_best_params
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from common.config import Config

args = get_args("evaluate_rf.yaml")
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)

def azure_mount(cfg):
    if cfg.env=='azure':
        _, mount_context = setup_azure(cfg.run_name)
        mount_dir = mount_context.mount_point
        cfg.model_dir = join(mount_dir, cfg.model_dir) # specify paths here
        cfg.output_dir = join('outputs', cfg.run_name)
    else:
        mount_context = None
    return cfg, mount_context

def prepare_eval_directory(config_path: str, cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    copyfile(config_path, join(cfg.output_dir, split(config_path)[1]))
    return setup_logger(cfg.output_dir)

def main(config_path):
    cfg = load_config(config_path)
    cfg.output_dir = join(cfg.model_dir, 'results', 'RF', cfg.run_name)
    cfg, mount_context = azure_mount(cfg)
    logger = prepare_eval_directory(config_path, cfg)

    metrics = [get_function(metric) for metric in cfg.metrics.values()]
    
    Results_dic = {}
    for task in tqdm(cfg.tasks, desc='Tasks',  file=TqdmToLogger(logger)):
        logger.info(f'Processing task {task}')
        reader = PatientHDF5Reader(join(cfg.model_dir, 'encodings', task, 'encodings.h5'))
        X, y = reader.read_arrays()
        skf = StratifiedKFold(cfg.n_folds)
        results = {metric.__name__:[] for metric in metrics}
        for fold in tqdm(skf.split(X,y), desc='CV',  file=TqdmToLogger(logger)):
            train_ids, val_ids = fold
            X_train, X_val, y_train, y_val = X[train_ids], X[val_ids], y[train_ids], y[val_ids]
            logger.info("Optimize hyperparameters")
            best_params = find_best_params(X_train, y_train, cfg.param_grid)
            logger.info(f"Best params: {best_params}")
            logger.info("Train model")
            clf = RandomForestClassifier(**best_params, random_state=42)
            clf.fit(X_train, y_train)
            logger.info("Predict")
            pred = clf.predict(X_val)
            pred_probas = clf.predict_proba(X_val)[:, 1]
            logger.info("Evaluate")
            for metric in metrics:
                score = metric(y_val, pred_probas if metric.__name__.endswith('auc') else pred)
                logger.info(f"{metric.__name__}: {score}")
                results[metric.__name__].append(score)
        Results_dic[task] = results

    Results_dic = get_mean_std(Results_dic)
    Results_df = pd.DataFrame(Results_dic).T
    Results_df.to_csv(join(cfg.output_dir, 'results.csv'))

    if cfg.env=='azure':
        from azure_run import file_dataset_save
        file_dataset_save(local_path=split(cfg.output_dir)[0], datastore_name = "workspaceblobstore",
                    remote_path = join("PHAIR", cfg.model_dir, 'results', 'RF'))
        mount_context.stop()

if __name__ == '__main__':
    main(config_path)