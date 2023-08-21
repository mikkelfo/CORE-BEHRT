import os
from collections import defaultdict
from os.path import join, split
from shutil import copyfile

import numpy as np
import pandas as pd
from common.azure import setup_azure
from common.config import Config, get_function, load_config
from common.io import PatientHDF5Reader
from common.logger import TqdmToLogger
from common.setup import get_args, setup_logger
from evaluation.optimize import find_best_params_RF
from evaluation.utils import Oversampler, get_mean_std, sample
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

args = get_args("evaluate_rf.yaml")
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)


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
        X, y = sample(X, y, task, logger)
        
        logger.info(f"Use {cfg.optimizer.split} of the data to optimize hyperparameters")
        X, X_opt, y, y_opt = train_test_split(X, y, test_size=cfg.optimizer.split, random_state=42)
        best_params = find_best_params_RF(X_opt, y_opt, param_grid=cfg.optimizer.param_grid, **cfg.parallel)
        logger.info(f"Best params: {best_params}")
        logger.info(f"{cfg.n_folds}-fold Cross-validate")
        skf = StratifiedKFold(cfg.n_folds, shuffle=True, random_state=42)
        # Parallelize the cross-validation process
        parallel = Parallel(n_jobs=cfg.parallel.n_jobs, temp_folder=cfg.parallel.temp_folder)
        parallel_results = parallel(delayed(process_fold)(fold, X, y, task, metrics, best_params, logger) for fold in skf.split(X, y))
        # Aggregate results
        Results_dic[name] = get_results_dic(parallel_results)

    Results_dic = get_mean_std(Results_dic)
    Results_df = pd.DataFrame(Results_dic).T
    Results_df.to_csv(join(cfg.output_dir, f'results.csv'), index_label='task')

    if cfg.env=='azure':
        from azure_run import file_dataset_save
        file_dataset_save(local_path=split(cfg.output_dir)[0], datastore_name = "workspaceblobstore",
                    remote_path = join("PHAIR", cfg.model_dir, 'results', 'RF'))
        mount_context.stop()

def process_fold(fold, X, y, task, metrics, best_params, logger):
    train_ids, val_ids = fold
    X_train, X_val, y_train, y_val = X[train_ids], X[val_ids], y[train_ids], y[val_ids]
    
    X_train, y_train = oversample_data(X_train, y_train, task, logger)
    pred, pred_probas = train_and_predict(X_train, y_train, X_val, best_params, logger)
    results = evaluate_predictions(y_val, pred, pred_probas, metrics, logger)
    return results

def train_and_predict(X_train, y_train, X_val, best_params, logger):
    logger.info("Train model")
    clf = RandomForestClassifier(**best_params, random_state=42)
    clf.fit(X_train, y_train)
    
    logger.info("Predict")
    pred = clf.predict(X_val)
    pred_probas = clf.predict_proba(X_val)[:, 1]
    
    return pred, pred_probas

def evaluate_predictions(y_val, pred, pred_probas, metrics, logger):
    results = {}
    logger.info("Evaluate")
    for metric in metrics:
        score = metric(y_val, pred_probas if metric.__name__.endswith('auc') else pred)
        logger.info(f"{metric.__name__}: {score}")
        results[metric.__name__] = score
    return results

def oversample_data(X, y, task, logger):
    if "oversampling_ratio" in task and task.get("oversampling_ratio", False):
        logger.info(f"Oversampling with ratio {task.oversampling_ratio}")
        oversampler = Oversampler(task.oversampling_ratio)
        X, y = oversampler.fit_resample(X, y)
        logger.info(f"New dataset size: {len(X)}")
        logger.info(f"New Perc. positives: {sum(y)/len(y)}")
    return X, y

def get_results_dic(parallel_results):
    results = defaultdict(list)
    for fold_result in parallel_results:
        for key, value in fold_result.items():
            results[key].append(value)
    return results

def sample(X,y, task, logger):
    if 'sample' in task:
        logger.info(f"Sampling with {task.sample}")
        X, y = sample(X, y, **task.sample)
        logger.info(f"New dataset size: {len(X)}")
        logger.info(f"New Perc. positives: {sum(y)/len(y)}")
    return X, y

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


if __name__ == '__main__':
    main(config_path)