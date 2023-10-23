import torch
from functools import lru_cache


def wrapper(f):
    def wrap(self, outputs, batch, **kwargs):
        data = self.base_data(outputs.logits, batch["target"])

        # Apply topk if specified
        if kwargs.get("topk"):
            data = self.topk(kwargs["topk"], **data)

        self.cm = self.confusion_matrix(data["targets"], data["predictions"])

        # Apply function
        result = f(self)

        # Apply rounding if specified
        if kwargs.get("round", self.round):
            result = torch.round(result, decimals=kwargs.get("round", self.round))

        if torch.isnan(result): # NaNs are produced by division by zero
            result = torch.tensor(0)

        return result.item()

    return wrap


class BaseData:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, logits, target):
        return {
            "probas": self._probas(logits),
            "predictions": self._predictions(logits),
            "targets": self._targets(target),
        }

    def _probas(self, logits):
        return torch.sigmoid(logits).cpu().view(-1)

    def _predictions(self, logits):
        probas = self._probas(logits)
        return (probas > self.threshold).long()

    def _targets(self, target):
        return target.view(-1).long().cpu()


class TopkData:
    def __init__(self, topk):
        self.topk = topk

    def __call__(self, **data):
        topk = torch.topk(data["probas"], self.topk, dim=0).indices  # Get topk indices
        return {key: values[topk] for key, values in data.items()}


class Metrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.round = False

    def set_rounding(self, arg: int):  # False or number of decimals
        self.round = arg

    @lru_cache(maxsize=1)
    def base_data(self, outputs, batch):
        return BaseData(self.threshold)(outputs, batch)

    @lru_cache(maxsize=1)
    def confusion_matrix(self, targets, predictions):
        return ConfusionMatrix(targets, predictions)

    @lru_cache(maxsize=1)
    def topk(self, topk, **data):
        return TopkData(topk)(**data)  # Filter data to topk indices

    def mlm_topk(self, outputs, batch, **kwargs):
        logits = outputs.logits
        targets = batch["target"]

        # Mask out the -100 values (non-masked tokens)
        mask = targets != -100
        logits = logits[mask]  # (bs x seq_len, vocab_size)
        targets = targets[mask]  # (bs x seq_len)

        # Get topk predictions and check if any of them is correct
        _, pred = logits.topk(kwargs["topk"], 1)  # (bs x seq_len, topk)
        correct = pred.t().eq(targets)  # (topk, bs x seq_len)
        correct = correct.any(0)  # Check if any topk prediction is correct

        # Return the accuracy
        return correct.float().mean().item()

    @wrapper
    def false_positive_rate(self):
        return self.cm.false_positive_rate()

    @wrapper
    def false_negative_rate(self):
        return self.cm.false_negative_rate()

    @wrapper
    def true_positive_rate(self):
        return self.cm.true_positive_rate()

    @wrapper
    def true_negative_rate(self):
        return self.cm.true_negative_rate()

    @wrapper
    def precision(self):
        return self.cm.precision()

    @wrapper
    def recall(self):
        return self.cm.true_positive_rate()

    @wrapper
    def sensitivity(self):
        return self.cm.true_positive_rate()

    @wrapper
    def specificity(self):  # True Negative Rate
        return self.cm.true_negative_rate()

    @wrapper
    def accuracy(self):
        return self.cm.accuracy()

    @wrapper
    def f1_score(self):
        return self.cm.f1_score()

    @wrapper
    def dice(self):
        return self.cm.dice()


class ConfusionMatrix:
    """Ordering of the confusion matrix is as follows (from sci-kit):
    tn  fp
    fn  tp

    Matrix is fixed to Binary Classification
    """

    def __init__(self, true, pred):
        self.data = self._compute_matrix(true, pred)
        self.tn = self.data[0][0]
        self.tp = self.data[1][1]
        self.fp = self.data[0][1]
        self.fn = self.data[1][0]

    def __call__(self, data):
        return self._compute_matrix(data["targets"], data["predictions"])

    def _compute_matrix(self, true, pred):
        matrix = torch.zeros((2, 2), dtype=torch.int64)  # Confusion matrix
        # Populate the matrix -> matrix[true[i]][pred[i]] += 1
        matrix = matrix.index_put((true, pred), torch.tensor([1]), accumulate=True)

        return matrix

    def false_positive_rate(self):  # False Alarm Rate
        return self.fp / (self.fp + self.tn)

    def false_negative_rate(self):  # Miss Rate
        return self.fn / (self.fn + self.tp)

    def true_positive_rate(self):  # Recall, Sensitivity
        return self.tp / (self.tp + self.fn)

    def true_negative_rate(self):  # Specificity
        return self.tn / (self.tn + self.fp)

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):  # True Positive Rate
        return self.true_positive_rate()

    def sensitivity(self):  # True Positive Rate
        return self.true_positive_rate()

    def specificity(self):  # True Negative Rate
        return self.true_negative_rate()

    def accuracy(self):
        return (self.tp + self.tn) / (self.data.sum())

    def f1_score(self):
        return (
            2 * (self.precision() * self.recall()) / (self.precision() + self.recall())
        )

    def dice(self):
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn)
