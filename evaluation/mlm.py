import torch

"""Computes the precision@k for the specified value of k"""
def top_k(outputs, batch, topk=10) -> dict:
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
    else:
        return correct.any(0).float().mean().item()

