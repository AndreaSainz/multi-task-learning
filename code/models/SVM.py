import torch
import torch.nn as nn

class TorchRbfSVM(nn.Module):
    def __init__(self, support_vectors, dual_coef, intercept, gamma):
        super().__init__()

        self.support_vectors = nn.Parameter(
            torch.tensor(support_vectors, dtype=torch.float32),
            requires_grad=False
        )

        self.dual_coef = nn.Parameter(
            torch.tensor(dual_coef, dtype=torch.float32),
            requires_grad=False
        )

        self.intercept = nn.Parameter(
            torch.tensor(intercept, dtype=torch.float32),
            requires_grad=False
        )

        self.gamma = gamma

    def forward(self, x):

        B = x.shape[0]

        x = x.view(B, -1)

        # [B,1,D] - [1,SV,D]
        diff = x.unsqueeze(1) - self.support_vectors.unsqueeze(0)

        dist2 = (diff ** 2).sum(dim=2)

        K = torch.exp(-self.gamma * dist2)

        scores = torch.matmul(K, self.dual_coef.t()) + self.intercept

        return scores