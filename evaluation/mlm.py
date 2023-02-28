import torch


def top_k(outputs, batch, topk=10) -> dict[str, float]:
    logits = outputs.logits
    target = batch['target']
    """Computes the precision@k for the specified values of k"""
    ind = torch.where(target != -100)
    maxk = max(topk)

    logits = logits[ind]
    target = target[ind]

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq()

    correct_k = correct[:topk].float().sum(0)
    return correct_k.sum() / len(correct_k)

def multi_top_k(outputs, batch, topk=(1, 10, 30, 50, 100)):
    return {
        f'top_{k}': top_k(outputs, batch, k)
        for k in topk
    }

