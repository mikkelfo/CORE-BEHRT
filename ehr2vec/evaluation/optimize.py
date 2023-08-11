from sklearn.ensemble import RandomForestClassifier
from itertools import product
from tqdm import tqdm

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