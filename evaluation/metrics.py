import torch

"""Computes the precision@k for the specified value of k"""
def top_k(outputs, batch, topk=10, average=False) -> dict:
    logits = outputs.logits
    target = batch['target']
    
    ind = torch.where((target != -100) & (target != 0))

    logits = logits[ind]
    target = target[ind]

    _, pred = logits.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target)
    if correct.numel() == 0:
        return 0
    if not average:
        return correct.any(0).float()
    else:
        return correct.any(0).float().mean().item()

def binary_hit(outputs, batch, threshold=0.5):
    logits = outputs.logits
    target = batch['target']

    probs = torch.sigmoid(logits)
    predictions = (probs > threshold).long().view(-1)         # TODO: Add uncertainty measure

    correct = (predictions == target).float().mean().item()

    return correct

