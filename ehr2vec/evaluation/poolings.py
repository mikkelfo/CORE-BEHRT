import torch

class MeanPooling:
    def pool(self, hidden, mask, output):
        return torch.mean(mask.unsqueeze(-1) * hidden, dim=1)
    
class CLSPooling:
    def pool(self, hidden, mask, output):
        return hidden[:,0,:] # CLS token
    