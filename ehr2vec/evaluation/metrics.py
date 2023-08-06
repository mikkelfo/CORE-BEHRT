import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

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

class Accuracy():
    def __init__(self) -> None:
        pass
    def __call__(self, outputs, batch) -> Any:
        logits = outputs.get('prediction_logits', outputs.get('logits', None)) 
        probas = torch.nn.functional.softmax(logits, dim=-1)
        _, predictions = torch.max(probas, dim=-1)
        try:
            score = accuracy_score(batch['target'], predictions)
            return score
        except Warning("Accuracy score could not be computed"):
            return None

class Precision():
    def __init__(self) -> None:
        pass
    def __call__(self, outputs, batch) -> Any:
        logits = outputs.get('prediction_logits', outputs.get('logits', None)) 
        probas = torch.nn.functional.softmax(logits, dim=-1)
        _, predictions = torch.max(probas, dim=-1)
        return precision_score(batch['target'], predictions, zero_division=0)
    
class Recall():
    def __init__(self) -> None:
        pass
    def __call__(self, outputs, batch) -> Any:
        logits = outputs.get('prediction_logits', outputs.get('logits', None)) 
        probas = torch.nn.functional.softmax(logits, dim=-1)
        _, predictions = torch.max(probas, dim=-1)
        return recall_score(batch['target'], predictions, zero_division=0)

class ROC_AUC():
    def __init__(self) -> None:
        pass
    def __call__(self, outputs, batch) -> Any:
        logits = outputs.get('prediction_logits', outputs.get('logits', None)) 
        probas = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()
        try:
            score = roc_auc_score(batch['target'], probas[:,-1])
            return score
        except:
            print("ROC AUC score could not be computed")
            return 0