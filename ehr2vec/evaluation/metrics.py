import logging

import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

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
        
class LossAccessor:
    def __init__(self, loss_name):
        self.loss_name = loss_name
    
    def __call__(self, outputs, batch):
        return outputs.__getattribute__(self.loss_name)

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
    def __init__(self, threshold=0.5) -> None:
        self.threshold = threshold
        
    def _return_probas_and_targrets(self, outputs, batch):
        probas = torch.sigmoid(outputs.logits)
        return probas.cpu(), batch['target'].cpu()

    def _return_predictions_and_targrets(self, outputs, batch):
        probas, targets = self._return_probas_and_targrets(outputs, batch)
        predictions = (probas > self.threshold).long().view(-1)
        return predictions, targets
    
    def _return_confusion_matrix(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targrets(outputs, batch)
        return confusion_matrix(targets, predictions).ravel()

    def __call__(self, outputs, batch):
        raise NotImplementedError
    
class Accuracy(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targrets(outputs, batch)
        try:
            return accuracy_score(targets, predictions)
        except:
            logger.warn("Accuracy score could not be computed")
            return 0
        
class Dice(BaseMetric):
    def __call__(self, outputs, batch):
        tn, fp, fn, tp = self._return_confusion_matrix(outputs, batch)
        return (2. * tp) / (2 * tp + fp + fn)
class Balanced_Accuracy(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targrets(outputs, batch)
        try:
            return balanced_accuracy_score(targets, predictions)
        except:
            logger.warn("Balanced accuracy score could not be computed")
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
    
class Cohen_Kappa(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targrets(outputs, batch)
        return cohen_kappa_score(targets, predictions)
    
class Matthews_Correlation_Coefficient(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, targets = self._return_predictions_and_targrets(outputs, batch)
        return matthews_corrcoef(targets, predictions)

class Percentage_Positives(BaseMetric):
    def __call__(self, outputs, batch):
        predictions, _ = self._return_predictions_and_targrets(outputs, batch)
        return predictions.float().mean().item()

class Mean_Probability(BaseMetric):
    def __call__(self, outputs, batch):
        probas, _ = self._return_probas_and_targrets(outputs, batch)
        return probas.mean().item()
    
class True_Positives(BaseMetric):
    def __call__(self, outputs, batch):
        tn, fp, fn, tp = self._return_confusion_matrix(outputs, batch)
        return tp 

class False_Positives(BaseMetric):
    def __call__(self, outputs, batch):
        tn, fp, fn, tp = self._return_confusion_matrix(outputs, batch)
        return fp

class True_Negatives(BaseMetric):
    def __call__(self, outputs, batch):
        tn, fp, fn, tp = self._return_confusion_matrix(outputs, batch)
        return tn

class False_Negatives(BaseMetric):
    def __call__(self, outputs, batch):
        tn, fp, fn, tp = self._return_confusion_matrix(outputs, batch)
        return fn
        
def specificity(y, y_scores):
    tn, fp, fn, tp = confusion_matrix(y, y_scores).ravel()
    return tn / (tn + fp)