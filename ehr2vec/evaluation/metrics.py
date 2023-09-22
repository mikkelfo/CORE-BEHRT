import logging

import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

logger = logging.getLogger(__name__)  # Get the logger for this module

"""Computes the precision@k for the specified value of k"""
class PrecisionAtK:
    def __init__(self, topk=10):
        """Computes the precision@k for the specified value of k"""
        self.topk = topk

    def __call__(self, outputs, batch):
        logits = outputs.logits
        target = batch['target']
        
        ind = torch.where((target != -100) & (target != 0))

        logits = logits[ind]
        target = target[ind]

        _, pred = logits.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target)
        if correct.numel() == 0:
            return 0
        else:
            return correct.any(0).float().mean().item()

def binary_hit(outputs, batch, threshold=0.5, average=True):
    logits = outputs.logits
    target = batch['target']

    probs = torch.sigmoid(logits)
    predictions = (probs > threshold).long().view(-1)         # TODO: Add uncertainty measure

    if not average:
        return (predictions == target).float()

    else:
        return (predictions == target).float().mean().item()


class BaseMetric:
    def _return_probas_and_targrets(self, outputs, batch):
        probas = torch.sigmoid(outputs.logits)
        return probas.cpu(), batch['target'].cpu()

    def _return_predictions_and_targrets(self, outputs, batch, threshold=0.5):
        probas, targets = self._return_probas_and_targrets(outputs, batch)
        predictions = (probas > threshold).long().view(-1)
        return predictions, targets

    def __call__(self, outputs, batch):
        raise NotImplementedError("Subclasses must implement this method")

class Accuracy(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targrets(outputs, batch)
        try:
            return accuracy_score(targets, predictions)
        except:
            logger.warn("Accuracy score could not be computed")
            return 0
        
class Precision(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targrets(outputs, batch)
        return precision_score(targets, predictions, zero_division=0)
    
class Recall(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targrets(outputs, batch)
        return recall_score(targets, predictions, zero_division=0)

class ROC_AUC(BaseMetric):
    def __call__(self, outputs, batch):
        probas, targets = self._return_probas_and_targrets(outputs, batch)
        try:
            return roc_auc_score(targets, probas)
        except:
            logger.warn("ROC AUC score could not be computed")
            return 0
        
class PR_AUC(BaseMetric):
    def __call__(self, outputs, batch):
        probas, targets = self._return_probas_and_targrets(outputs, batch)
        try:
            return average_precision_score(targets, probas)
        except:
            logger.warn("PR AUC score could not be computed")
            return 0

class F1(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targrets(outputs, batch)
        return f1_score(targets, predictions, zero_division=0)
        
def specificity(y, y_scores):
    tn, fp, fn, tp = confusion_matrix(y, y_scores).ravel()
    return tn / (tn + fp)