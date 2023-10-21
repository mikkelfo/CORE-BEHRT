import os
from os.path import join

import pandas as pd
import torch
from common.azure import save_to_blobstore
from common.config import get_function, instantiate, load_config
from common.loader import DatasetPreparer
from common.setup import azure_onehot_setup, get_args, setup_logger
from evaluation.utils import evaluate_predictions, get_pos_weight
from sklearn.model_selection import train_test_split

CONFIG_NAME = 'evaluate_one_hot.yaml'
DEFAULT_VAL_SPLIT = 0.2
BLOBSTORE = 'PHAIR'

args = get_args(CONFIG_NAME)
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)


def main_finetune():
    cfg = load_config(config_path)
    features_path = cfg.paths.finetune_features_path
    cfg.paths.output_path = features_path
    cfg, _, mount_context = azure_onehot_setup(cfg)

    model = instantiate(cfg.model)
    model_name = model.__class__.__name__
    cfg.paths.run_name = f"{model_name}_{cfg.paths.run_name}"
    run_folder=join(cfg.paths.output_path, cfg.paths.run_name)
     
    os.makedirs(run_folder, exist_ok=True)
    logger = setup_logger(run_folder)
    cfg.save_to_yaml(join(run_folder, 'evaluate_one_hot.yaml'))

    X, y, new_vocab = DatasetPreparer(cfg).prepare_onehot_features()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.trainer_args.get('val_split',DEFAULT_VAL_SPLIT), 
        random_state=cfg.trainer_args.get('seed', 42))
    
    
    logger.info('Instantiating Model')
    sample_weight = get_pos_weight(cfg, y_train) if cfg.trainer_args.get('sample_weight', False) else None
    
    model.fit(X_train, y_train, sample_weight=sample_weight,)
    pred_probas = model.predict_proba(X_val)[:, 1]
    metrics = [get_function(metric) for metric in cfg.metrics.values()]
    results = evaluate_predictions(y_val, pred_probas, metrics)
    torch.save(model, join(run_folder, 'model.pth'))
    torch.save(new_vocab, join(run_folder, 'new_vocab.pt'))
    logger.info('Saving results')
    results_df = pd.DataFrame(results, index=[0])
    results_df.to_csv(join(run_folder, 'results.csv'), index=False)
    try:
        feature_importances = model.feature_importances_
        feature_names = ['age']
        sorted_new_vocab = {k: v for k, v in sorted(new_vocab.items(), key=lambda item: item[1])}
        feature_names = feature_names + [k for k, v in sorted_new_vocab.items()]
        df_feat_imp = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
        df_feat_imp.sort_values('importance', ascending=False).to_csv(join(run_folder, 'feature_importances.csv'), index=False)
    except:
        pass
    if cfg.env=='azure':
        save_to_blobstore(cfg.paths.run_name,
                          join(BLOBSTORE, features_path, cfg.paths.run_name))
        mount_context.stop()
    logger.info('Done')


if __name__ == '__main__':
    main_finetune()