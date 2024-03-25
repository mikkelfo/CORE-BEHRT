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

    def forward(self, hidden_states: torch.Tensor, attention_mask = None) -> torch.Tensor:
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

    def forward(self, x, attention_mask=None):
        # Compute attention scores using a dot product between a learnable query and the keys
        keys = self.key(x)  # Shape: [batch_size, seq_length, hidden_size]
        attn_scores = torch.matmul(keys, self.query)  # Shape: [batch_size, seq_length]

        expanded_attention_mask = attention_mask
        attn_scores[expanded_attention_mask == 0] = 0

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

        pool_type = config.pool_type.lower()
        if pool_type == 'cls':
            self.pool = self.pool_cls
        elif pool_type == 'mean':
            self.pool = self.pool_mean
        elif pool_type == 'sum':
            self.pool = self.pool_sum
        elif pool_type == 'max':
            self.pool = self.pool_max
        elif pool_type == 'attention':
            self.pool = AttentionPool(config.hidden_size)
        elif pool_type == 'gru':
            self.pool = BaseRNN(config, torch.nn.GRU)
        elif pool_type == 'lstm':
            self.pool = BaseRNN(config, torch.nn.LSTM)
        else:
            logger.warning(f'Unrecognized pool_type: {pool_type}. Defaulting to CLS pooling.')
            self.pool_type = 'cls' # Default to CLS pooling if pool_type is not recognized
            self.pool = self.pool_cls

    def forward(self, hidden_states: torch.Tensor, attention_mask=None) -> torch.Tensor:
        x = self.pool(hidden_states, attention_mask=attention_mask)
        x = self.classifier(x)
        return x

    def pool_cls(self, x, attention_mask=None):
        return x[:, 0]

    def pool_mean(self, x, attention_mask):
        sum_embeddings = torch.sum(x * attention_mask.unsqueeze(-1), dim=1)
        sum_mask = attention_mask.sum(dim=1).unsqueeze(-1)
        return sum_embeddings / sum_mask

    def pool_sum(self, x, attention_mask):
        return torch.sum(x * attention_mask.unsqueeze(-1), dim=1)
    
    def pool_max(self, x, attention_mask):
        expanded_attention_mask = attention_mask.unsqueeze(-1).expand(x.shape)
        x[expanded_attention_mask == 0] = -1e9
        return torch.max(x, dim=1).values
    
    def initialize_extended_head(self, config):
        if config.extend_head.get('hidden_size', None) is not None:
            intermediate_size = config.extend_head.hidden_size
        else:
            intermediate_size = config.hidden_size//3 *2
        self.activation = torch.nn.GELU()
        self.hidden_layer = torch.nn.Linear(config.hidden_size, intermediate_size)
        self.cls_layer = torch.nn.Linear(intermediate_size, 1)
        self.classifier = torch.nn.Sequential(self.hidden_layer, self.activation, self.cls_layer)



class BaseRNN(torch.nn.Module):
    def __init__(self, config, rnn_type) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.bidirectional = config.get('bidirectional', False)
        self.rnn = rnn_type(self.hidden_size, self.hidden_size, batch_first=True, 
                            bidirectional=self.bidirectional)
        # Adjust the input size of the classifier based on the bidirectionality
        classifier_input_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.classifier = torch.nn.Linear(classifier_input_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(hidden_states, lengths, batch_first=True, enforce_sorted=False)
        # Pass the hidden states through the RßNN
        output, _ = self.rnn(packed)
        # Unpack it back to a padded sequence
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        last_sequence_idx = lengths - 1
        
        # Use the last output of the RNN as input to the classifier
        # If bidirectional, we need to concatenate the last output from the forward
        # pass and the first output from the backward pass
        forward_output = output[torch.arange(output.shape[0]), last_sequence_idx, :self.hidden_size]  # Last non-padded output from the forward pass
        if self.bidirectional:
            backward_output = output[:, 0, self.hidden_size:]  # First output from the backward pass
            x = torch.cat((forward_output, backward_output), dim=-1)
        else:
            x = forward_output  # Last output for unidirectional
        x = self.classifier(x)
        return x
    