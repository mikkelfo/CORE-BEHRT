from sklearn.ensemble import RandomForestClassifier
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed

def get_all_combinations(param_grid):
    return [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]

def find_best_params(X, y, param_grid):
    all_params = get_all_combinations(param_grid)
    
    # Parallel processing
    n_jobs = -1  # This means using all processors
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(X, y, params) for params in tqdm(all_params, desc='Grid Search')
    )
    
    # Find the best result
    best_score, best_params = max(results, key=lambda x: x[0])
    
    return best_params

def evaluate_params(X, y, params):
    clf = RandomForestClassifier(oob_score=True, random_state=42, **params)
    clf.fit(X, y)
    oob_score = clf.oob_score_
    return oob_score, params