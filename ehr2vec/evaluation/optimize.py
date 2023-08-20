from sklearn.ensemble import RandomForestClassifier
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed

def get_all_combinations(param_grid):
    return [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]

def find_best_params_RF(X, y, param_grid, n_jobs=-1, temp_folder=None):
    all_params = get_all_combinations(param_grid)
    # Parallel processing
    results = Parallel(n_jobs=n_jobs, temp_folder=temp_folder)(
        delayed(evaluate_params_RF)(X, y, params) for params in tqdm(all_params, desc='Grid Search')
    )
    # Find the best result
    best_score, best_params = max(results, key=lambda x: x[0])
    
    return best_params

def evaluate_params_RF(X, y, params):
    clf = RandomForestClassifier(oob_score=True, random_state=42, **params)
    clf.fit(X, y)
    oob_score = clf.oob_score_
    return oob_score, params