import numpy as np

def get_mean_std(Results_dic:dict)->dict:
    """Takes a nested dict with methods as outer dict and metrics as inner dict which contains lists os metrics,
    and computes mean \pm std for each list."""
    Results_mean_std = {}

    for method in Results_dic.keys():
        Results_mean_std[method] = {}
        for metric in Results_dic[method].keys():
            mean = np.mean(np.array(Results_dic[method][metric]))
            std = np.std(np.array(Results_dic[method][metric]))
            Results_mean_std[method][metric] = f'{mean:.3f} ± {std:.3f}'
    return Results_mean_std