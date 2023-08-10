from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from common.io import PatientHDF5Reader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

from itertools import product

# Define a custom iterator
def get_all_combinations(param_grid):
    return [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]

def find_best_params(X, y, param_grid):
    all_params = get_all_combinations(param_grid)
    best_score = float('-inf')
    best_params = None
    for params in tqdm(all_params, desc='Grid Search'):
        clf = RandomForestClassifier(oob_score=True, random_state=42, **params)
        clf.fit(X, y)
        oob_score = clf.oob_score_
        if oob_score > best_score:
            best_score = oob_score
            best_params = params

    return best_params


tasks = ['censored_patients', 'censored_patients2']
metrics = [roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score]
n_folds = 2
# param_grid = {'n_estimators':[1000],'max_depth':[5,10, None], 'min_samples_split':[2,5 ,10]}
param_grid = {'n_estimators':[1000],'max_depth':[5, None], 'min_samples_split':[2]}

for task in tasks:
    reader = PatientHDF5Reader(f'outputs/pretraining/test/encodings/{task}/encodings.h5')
    X, y = reader.read_arrays()
    skf = StratifiedKFold(n_folds)

    for i, fold in tqdm(enumerate(skf.split(X,y)), desc='CV'):
        train_ids, val_ids = fold
        X_train, X_val, y_train, y_val = X[train_ids], X[val_ids], y[train_ids], y[val_ids]
        best_params = find_best_params(X_train, y_train, param_grid)

        clf = RandomForestClassifier(**best_params, random_state=42)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_val)
        pred_probas = clf.predict_proba(X_val)[:, 1]

        for metric in metrics:
            if metric.__name__.endswith('auc'):
                print(f'Val {metric.__name__}: {metric(y_val, pred_probas)}')
            else:
                print(f'Val {metric.__name__}: {metric(y_val, pred)}')