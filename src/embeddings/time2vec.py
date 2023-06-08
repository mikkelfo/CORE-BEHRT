import torch


class Time2Vec(torch.nn.Module):

    def __init__(self, input_dim=1, output_dim=768, function=torch.cos):
        super().__init__()
        self.f = function

        # for i = 0
        self.w0 = torch.nn.Parameter(torch.randn(input_dim, 1))
        self.phi0 = torch.nn.Parameter(torch.randn(1))
        # for 1 <= i <= k (input_size)
        self.w = torch.nn.Parameter(torch.randn(input_dim, output_dim - 1))
        self.phi = torch.nn.Parameter(torch.randn(output_dim - 1))

    def forward(self, tau: torch.Tensor):
        tau = tau.unsqueeze(2)
        v1 = torch.matmul(tau, self.w0) + self.phi0
        v2 = self.f(torch.matmul(tau, self.w) + self.phi)
        return torch.cat((v1, v2), dim=-1)
