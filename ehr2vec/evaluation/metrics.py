import torch

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

