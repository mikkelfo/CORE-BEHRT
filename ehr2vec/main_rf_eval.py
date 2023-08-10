from sklearn.ensemble import RandomForestClassifier
from evaluation.optimize import find_best_params
from evaluation.metrics import get_mean_std
from sklearn.model_selection import StratifiedKFold
from common.io import PatientHDF5Reader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import pandas as pd


tasks = ['censored_patients', 'censored_patients2']
metrics = [roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score]
n_folds = 3
# param_grid = {'n_estimators':[1000],'max_depth':[5,10, None], 'min_samples_split':[2,5 ,10]}
param_grid = {'n_estimators':[1000],'max_depth':[5], 'min_samples_split':[2]}
Results_dic = {}
for task in tasks:
    reader = PatientHDF5Reader(f'outputs/pretraining/test/encodings/{task}/encodings.h5')
    X, y = reader.read_arrays()
    skf = StratifiedKFold(n_folds)
    results = {metric.__name__:[] for metric in metrics}
    for i, fold in tqdm(enumerate(skf.split(X,y)), desc='CV'):
        train_ids, val_ids = fold
        X_train, X_val, y_train, y_val = X[train_ids], X[val_ids], y[train_ids], y[val_ids]
        
        best_params = find_best_params(X_train, y_train, param_grid)

        clf = RandomForestClassifier(**best_params, random_state=42)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_val)
        pred_probas = clf.predict_proba(X_val)[:, 1]

        for metric in metrics:
            score = metric(y_val, pred_probas if metric.__name__.endswith('auc') else pred)
            results[metric.__name__].append(score)
    Results_dic[task] = results

Results_str = get_mean_std(Results_dic)
Results_df = pd.DataFrame(Results_str).T