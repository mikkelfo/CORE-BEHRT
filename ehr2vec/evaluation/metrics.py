import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_score, recall_score, roc_auc_score,
                             confusion_matrix)

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


class Accuracy:
    def __call__(self, outputs, batch):
        probas = torch.sigmoid(outputs.logits)
        predictions = (probas > 0.5).long().view(-1) 
        try:
            score = accuracy_score(batch['target'], predictions)
            return score
        except Warning("Accuracy score could not be computed"):
            return None
class Precision:
    def __call__(self, outputs, batch):
        probas = torch.sigmoid(outputs.logits)
        predictions = (probas > 0.5).long().view(-1) 
        # print(predictions) 
        # breakpoint()
        return precision_score(batch['target'], predictions, zero_division=0)
    
class Recall:
    def __call__(self, outputs, batch):
        probas = torch.sigmoid(outputs.logits)
        predictions = (probas > 0.5).long().view(-1)
        return recall_score(batch['target'], predictions, zero_division=0)

class ROC_AUC:
    def __call__(self, outputs, batch):
        probas = torch.sigmoid(outputs.logits)
        try:
            score = roc_auc_score(batch['target'], probas)
            return score
        except:
            print("ROC AUC score could not be computed")
            return 0
class PR_AUC:
    def __call__(self, outputs, batch):
        probas = torch.sigmoid(outputs.logits)
        try:
            score = average_precision_score(batch['target'], probas)
            return score
        except:
            print("PR AUC score could not be computed")
            return 0
        
def specificity(y, y_scores):
    tn, fp, fn, tp = confusion_matrix(y, y_scores).ravel()
    return tn / (tn + fp)