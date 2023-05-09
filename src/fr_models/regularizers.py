import torch

class WeightNormReg(torch.nn.Module):
    def __init__(self, lamb=0.001):
        super().__init__()
        self.lamb = lamb
    
    def forward(r_model):
        return self.lamb*(r_model.a_model.W**2).mean()**0.5