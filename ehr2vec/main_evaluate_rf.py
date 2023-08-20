import os
from os.path import join, split
from shutil import copyfile

import numpy as np
import pandas as pd
from common.azure import setup_azure
from common.config import Config, get_function, load_config
from common.io import PatientHDF5Reader
from common.logger import TqdmToLogger
from common.setup import get_args, setup_logger
from evaluation.optimize import find_best_params
from evaluation.utils import get_mean_std, Oversampler, sample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

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

def get_dataset(X_unc, pids_unc, reader, shuffle=True):
    X, y = reader.read_arrays()
    y = y.astype(bool)
    mask = y==1
    X, y = X[mask], y[mask]
    pids = reader.read_pids()
    pids = [pid for i, pid in enumerate(pids) if mask[i]]
    indices = [i for i, pid in enumerate(pids_unc) if pid not in pids]
    X_unc = X_unc[indices]
    X, y = np.concatenate([X, X_unc]), np.concatenate([y, np.zeros(len(X_unc), dtype='bool')])
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]
    return X, y

def main(config_path):
    cfg = load_config(config_path)
    cfg.output_dir = join(cfg.model_dir, 'results', 'RF', cfg.run_name)
    cfg, mount_context = azure_mount(cfg)
    logger = prepare_eval_directory(config_path, cfg)
    
    reader = PatientHDF5Reader(join(cfg.model_dir, 'encodings', 'censored_patients', cfg.uncensored, 'encodings.h5'))
    X_uncencored, _ = reader.read_arrays()
    pids_uncensored = reader.read_pids()
    
    metrics = [get_function(metric) for metric in cfg.metrics.values()]
    Results_dic = {}
    for name in tqdm(cfg.tasks, desc='Tasks',  file=TqdmToLogger(logger)):
        task = cfg.tasks[name]
        logger.info(f'Processing task {task}')
        reader = PatientHDF5Reader(join(cfg.model_dir, 'encodings', 'censored_patients', task.folder, 'encodings.h5'))
        X, y = get_dataset(X_uncencored, pids_uncensored, reader)
        logger.info(f"Dataset size: {len(X)}")
        logger.info(f"Perc. positives: {sum(y)/len(y)}")

        if 'sample' in task:
            logger.info(f"Sampling with {task.sample}")
            X, y = sample(X, y, **task.sample)
            logger.info(f"New dataset size: {len(X)}")
            logger.info(f"New Perc. positives: {sum(y)/len(y)}")

        skf = StratifiedKFold(cfg.n_folds, shuffle=True, random_state=42)
        results = {metric.__name__:[] for metric in metrics}
        for fold in tqdm(skf.split(X,y), desc='CV',  file=TqdmToLogger(logger)):
            train_ids, val_ids = fold
            X_train, X_val, y_train, y_val = X[train_ids], X[val_ids], y[train_ids], y[val_ids]

            if "oversampling_ratio" in task and task.get("oversampling_ratio", False):
                logger.info(f"Oversampling with ratio {task.oversampling_ratio}")
                oversampler = Oversampler(task.oversampling_ratio)
                X_train, y_train = oversampler.fit_resample(X_train, y_train)
                logger.info(f"New dataset size: {len(X_train)}")
                logger.info(f"New Perc. positives: {sum(y_train)/len(y_train)}")
            logger.info("Optimize hyperparameters")
            best_params = find_best_params(X_train, y_train, **cfg.optimizer)
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
        Results_dic[name] = results

    Results_dic = get_mean_std(Results_dic)
    Results_df = pd.DataFrame(Results_dic).T
    Results_df.to_csv(join(cfg.output_dir, f'results.csv'), index_label='task')

    if cfg.env=='azure':
        from azure_run import file_dataset_save
        file_dataset_save(local_path=split(cfg.output_dir)[0], datastore_name = "workspaceblobstore",
                    remote_path = join("PHAIR", cfg.model_dir, 'results', 'RF'))
        mount_context.stop()

if __name__ == '__main__':
    main(config_path)