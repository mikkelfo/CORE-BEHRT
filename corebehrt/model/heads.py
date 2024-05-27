import torch


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

class FineTuneHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = torch.nn.Linear(config.hidden_size, 1)
        self.pool = BiGRU(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None) -> torch.Tensor:
        return self.pool(hidden_states, attention_mask=attention_mask)

class BiGRU(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = torch.nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                            bidirectional=True)
        # Adjust the input size of the classifier based on the bidirectionality
        classifier_input_size = self.hidden_size * 2
        self.classifier = torch.nn.Linear(classifier_input_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(hidden_states, lengths, batch_first=True, enforce_sorted=False)
        # Pass the hidden states through the RßNN
        output, _ = self.rnn(packed)
        # Unpack it back to a padded sequence
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        last_sequence_idx = lengths - 1
        
        # Use the last output of the RNN as input to the classifier
        # When bidirectional, we need to concatenate the last output from the forward
        # pass and the first output from the backward pass
        forward_output = output[torch.arange(output.shape[0]), last_sequence_idx, :self.hidden_size]  # Last non-padded output from the forward pass
        backward_output = output[:, 0, self.hidden_size:]  # First output from the backward pass
        x = torch.cat((forward_output, backward_output), dim=-1)
        x = self.classifier(x)
        return x

