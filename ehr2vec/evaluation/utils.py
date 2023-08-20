import numpy as np
from sklearn.utils import resample

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


class Oversampler:
    def __init__(self, ratio=1.0, random_state=None):
        """
        Initialize the oversampler.
        
        Parameters:
        - ratio: float (default=1.0)
            Ratio of the number of samples in the minority class after resampling 
            relative to the majority class. Must be between 0 and 1.
        - random_state: int (default=None)
            Random seed for reproducibility.
        """
        assert 0 <= ratio <= 1, "Ratio must be between 0 and 1."
        self.ratio = ratio
        self.random_state = random_state
        
    def fit_resample(self, X, y):
        """
        Oversample the minority class based on the specified ratio.
        """
        # Identify majority and minority classes
        majority_class = np.bincount(y).argmax()
        minority_class = 1 - majority_class
        
        X_majority, X_minority = X[y == majority_class], X[y == minority_class]
        y_majority, y_minority = y[y == majority_class], y[y == minority_class]
        
        # Calculate the number of samples after oversampling
        n_samples = int(X_majority.shape[0] * self.ratio)
        
        # Oversample minority class
        X_minority_oversampled, y_minority_oversampled = resample(
            X_minority, y_minority, 
            replace=True, 
            n_samples=n_samples,
            random_state=self.random_state
        )
        
        # Combine with majority class
        X_resampled = np.vstack((X_majority, X_minority_oversampled))
        y_resampled = np.hstack((y_majority, y_minority_oversampled))
        
        return X_resampled, y_resampled
    
def sample(X, y, n_samples=None, fraction=None):
    n_samples = n_samples if n_samples else int(X.shape[0] * fraction)
    indices = np.random.choice(X.shape[0], size=n_samples, replace=False)
    return X[indices], y[indices]

def validate_outcomes(all_outcomes, cfg):
    for outcome in cfg.outcomes:
        cfg.outcome = cfg.outcomes[outcome]
        if cfg.outcome.type:
            assert cfg.outcome.type in all_outcomes, f"Outcome {cfg.outcome.type} not found in outcomes."
        if cfg.outcome.censor_type:
            assert cfg.outcome.censor_type in all_outcomes, f"Censor type {cfg.outcome.censor_type} not found in outcomes."