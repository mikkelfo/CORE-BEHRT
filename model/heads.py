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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pool_type = self.config.pool_type
        if pool_type == 'cls':
            pooled_output = hidden_states[:, 0]
        elif pool_type == 'mean':
            pooled_output = hidden_states.mean(dim=1)
        elif pool_type == 'sum':
            pooled_output = hidden_states.sum(dim=1)
        else:
            pooled_output = hidden_states[:, 0]        # Default to cls

        x = self.classifier(pooled_output)

        return x