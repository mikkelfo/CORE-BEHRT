import torch


class MLMHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # BertPredictionHeadTransform
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = torch.nn.GELU()
        self.LayerNorm = torch.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        # BertLMPredictionHead
        self.decoder = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
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

        pool_type = config.pool_type
        if pool_type == "cls":
            self.pool = self.pool_cls
        elif pool_type == "mean":
            self.pool = self.pool_mean
        elif pool_type == "sum":
            self.pool = self.pool_sum
        else:
            self.pool = self.pool_cls

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


class HMLMHead(MLMHead):
    def __init__(self, config):
        super().__init__(config)

        # BertLMPredictionHead
        self.decoder = torch.nn.Linear(config.hidden_size, config.leaf_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(config.leaf_size))
        self.decoder.bias = self.bias
