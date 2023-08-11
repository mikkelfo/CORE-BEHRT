import torch

class MeanPooling:
    def pool(self, hidden, mask, output):
        return torch.mean(mask.unsqueeze(-1) * hidden, dim=1)
    
class CLSPooling:
    def pool(self, hidden, mask, output):
        return hidden[:,0,:] # CLS token
    
class AttentionWeightedSumPooling:
    def __init__(self, layers='last'):
        self.layers = layers
        
    def pool(self, hidden, mask, output):
        return self.attention_weighted_sum(output, hidden, mask)
    
    def attention_weighted_sum(self, outputs, hidden, mask):
        """Compute embedding using attention weights"""
        attention = outputs['attentions'] # tuple num layers (batch_size, num_heads, sequence_length, sequence_length)
        if self.layers=='all':
            attention = torch.stack(attention).mean(dim=0).mean(dim=1) # average over all layers and heads
        elif self.layers=='last':
            attention = attention[-1].mean(dim=1) # average over all layers and heads
        else:
            raise ValueError(f"Layers {self.layers} not implemented yet.")
        weights = torch.mean(attention, dim=1) # (batch_size, sequence_length, sequence_length)
        weights = weights / torch.sum(weights, dim=1, keepdim=True) # normalize, potentially uise softmax
        pooled_vec = torch.sum(mask.unsqueeze(-1) * hidden * weights.unsqueeze(-1), dim=1)
        return pooled_vec