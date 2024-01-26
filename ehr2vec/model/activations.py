import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.intermediate_size, config.intermediate_size, bias=False)
        self.linear2 = nn.Linear(config.intermediate_size, config.intermediate_size, bias=False)
        self.linear3 = nn.Linear(config.intermediate_size, config.intermediate_size, bias=False)
        self.swish = nn.SiLU()

    def forward(self, x):
        return self.linear3(self.swish(self.linear2(x)) * self.linear1(x))
