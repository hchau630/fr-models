import torch

class NormalizedMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        
    def forward(self, x, y):
        return self.mse_loss(x, y)/(y**2).mean()**0.5