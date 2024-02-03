import torch
import logging

logger = logging.getLogger(__name__)  # Get the logger for this module


class MLMHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # BertPredictionHeadTransform
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = torch.nn.GELU()
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # BertLMPredictionHead
        self.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.LayerNorm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

class HMLMHead(MLMHead):
    def __init__(self, config):
        super().__init__(config)

        # BertLMPredictionHead
        self.decoder = torch.nn.Linear(config.hidden_size, config.leaf_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(config.leaf_size))
        self.decoder.bias = self.bias

class AttentionPool(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(hidden_size))
        self.key = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # Compute attention scores using a dot product between a learnable query and the keys
        keys = self.key(x)  # Shape: [batch_size, seq_length, hidden_size]
        attn_scores = torch.matmul(keys, self.query)  # Shape: [batch_size, seq_length]

        # Apply softmax to get attention weights
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=1).unsqueeze(-1)  # Shape: [batch_size, seq_length, 1]

        # Compute weighted average
        weighted_avg = torch.sum(x * attn_weights, dim=1)  # Shape: [batch_size, hidden_size]
        return weighted_avg

class FineTuneHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = torch.nn.Linear(config.hidden_size, 1)
        if 'extend_head' in config.to_dict():
            self.initialize_extended_head(config)
        pool_type = config.pool_type
        if pool_type == 'cls':
            self.pool = self.pool_cls
        elif pool_type == 'mean':
            self.pool = self.pool_mean
        elif pool_type == 'sum':
            self.pool = self.pool_sum
        elif pool_type == 'attention':
            self.pool = AttentionPool(config.hidden_size)
        else:
            logger.warning(f'Unrecognized pool_type: {pool_type}. Defaulting to CLS pooling.')
            self.pool = self.pool_cls # Default to CLS pooling if pool_type is not recognized

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.pool(hidden_states)
        x = self.classifier(x)
        return x

    def pool_cls(self, x):
        return x[:, 0]

    def pool_mean(self, x):
        return x.mean(dim=1)

    def pool_sum(self, x):
        return x.sum(dim=1)
    
    def initialize_extended_head(self, config):
        if config.extend_head.get('hidden_size', None) is not None:
            intermediate_size = config.extend_head.hidden_size
        else:
            intermediate_size = config.hidden_size//3 *2
        self.activation = torch.nn.GELU()
        self.hidden_layer = torch.nn.Linear(config.hidden_size, intermediate_size)
        self.cls_layer = torch.nn.Linear(intermediate_size, 1)
        self.classifier = torch.nn.Sequential(self.hidden_layer, self.activation, self.cls_layer)

class ClassifierGRU(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rnn = torch.nn.GRU(config.hidden_size, config.hidden_size, batch_first=True)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pass the hidden states through the RNN
        output, _ = self.rnn(hidden_states)
        # Use the last output of the RNN as input to the classifier
        x = output[:, -1, :]
        x = self.classifier(x)
        return x
    
class ClassifierLSTM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rnn = torch.nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pass the hidden states through the RNN
        output, _ = self.rnn(hidden_states)
        # Use the last output of the RNN as input to the classifier
        x = output[:, -1, :]
        x = self.classifier(x)
        return x