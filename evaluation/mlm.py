import torch

"""Computes the precision@k for the specified values of k"""
def top_k(outputs, batch, topk=10) -> dict:
    logits = outputs.logits
    target = batch['target']
    
    ind = torch.where(target != -100)
    maxk = max(topk)

    logits = logits[ind]
    target = target[ind]

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq()

    correct_k = correct[:topk].float().sum(0)
    return correct_k.sum() / len(correct_k)

