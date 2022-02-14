import torch

class NormalizedLoss(torch.nn.Module):
    def forward(self, x, y):
        numerator = ((x-y)**2).mean()**0.5
        denominator = (y**2).mean()**0.5
        return numerator / denominator