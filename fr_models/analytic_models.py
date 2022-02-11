import abc

import numpy as np
import torch

from . import numerical_models as nmd
from . import kernels

class AnalyticModel(abc.ABC, torch.nn.Module):
    @abc.abstractmethod
    def numerical_model(self, *args):
        pass

class GaussianSSNModel(AnalyticModel):
    def __init__(self, W, sigma, power=2, w_dims=None, wn_order=9, period=2*torch.pi):
        super().__init__()
        
        if w_dims is None:
            w_dims = []
            
        assert W.ndim == 2 and W.shape[0] == W.shape[1]
        assert sigma.ndim == W.ndim + 1 and sigma.shape[:2] == W.shape
        
        self.W = W
        self.sigma = sigma
        self.power = power
        self.w_dims = w_dims
        self.wn_order = wn_order
        self.period = period
        self.n = W.shape[0] # number of cell types
        self.ndim = sigma.shape[-1] # number of spatial/feature dimensions
    
    @property
    def kernel(self):
        # Note: kernel will not be registered as a submodule, because the __setattr__ method is not called on kernel.
        # Hence, we don't have to worry about kernel.scale and kernel.cov showing up as parameters when 
        # calling GaussianSSNModel(*args, **kwargs).named_parameters()
        scale = self.W
        cov = torch.diag_embed(self.sigma**2)
        return kernels.K_wg(scale, cov, w_dims=self.w_dims, order=self.wn_order, period=self.period)
        
    def numerical_model(self, grid):
        W_discrete = self.kernel.discretize(grid) # (*shape, *shape, n, n)
        W_discrete = W_discrete.moveaxis(-2, 0).moveaxis(-1, 1+self.ndim) # (n, *shape, n, *shape)
        return nmd.MultiCellSSNModel(W_discrete, w_dims=self.w_dims, power=self.power)
    
class SpatialSSNModel(AnalyticModel):
    def __init__(self, W, sigma_s, ndim_s, sigma_f=None, power=2, w_dims=None, wn_order=9, period=2*torch.pi):
        super().__init__()
        
        if w_dims is None:
            w_dims = []
            
        assert W.ndim == 2 and W.shape[0] == W.shape[1]
        assert sigma_s.ndim == W.ndim and sigma_s.shape == W.shape
        if sigma_f is not None:
            assert sigma_f.ndim == W.ndim + 1 and sigma_f.shape[:2] == W.shape
        
        self.W = W
        self.sigma_s = sigma_s
        self.sigma_f = sigma_f
        self.ndim_s = ndim_s # number of spatial dimensions
        self.ndim_f = sigma_f.shape[-1] if sigma_f is not None else 0 # number of feature dimensions
        self.power = power
        self.w_dims = w_dims
        self.wn_order = wn_order
        self.period = period
        self.n = W.shape[0] # number of cell types
        self.ndim = self.ndim_s + self.ndim_f # number of spatial/feature dimensions
    
    @property
    def sigma(self):
        expanded_sigma_s = self.sigma_s.expand((self.ndim_s, *self.W.shape)).moveaxis(0,-1)
        if self.sigma_f is None:
            return expanded_sigma_s
        return torch.cat([expanded_sigma_s, expanded_sigma_f],dim=-1)
    
    def numerical_model(self, grid):
        a_model = GaussianSSNModel(self.W, self.sigma, power=self.power, w_dims=self.w_dims, wn_order=self.wn_order, period=self.period)
        return a_model.numerical_model(grid)