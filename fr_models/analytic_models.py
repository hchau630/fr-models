import abc

import numpy as np
import torch
from scipy.optimize import minimize

from . import numerical_models as nmd
from . import kernels

class AnalyticModel(abc.ABC, torch.nn.Module):
    @abc.abstractmethod
    def numerical_model(self, *args):
        pass

class GaussianSSNModel(AnalyticModel):
    # TODO: In the future, make a gaussian_ssn.py module with the static method being functions
    # in the module so that they can share the same name as the instance methods of GaussianSSNModel
    @staticmethod
    def get_W_f(W, sigma):
        def W_f(k):
            return W*torch.exp(-0.5*torch.einsum('ijk,...k->...ij',sigma**2,k**2))
        return W_f
    
    @staticmethod
    def get_f_prime(r_star, power=2):
        """
        Returns the gain f'(Wr+h) when network receives spatially uniform r_star
        """
        assert torch.all(r_star >= 0)
        return power*r_star**((power-1)/power)
    
    def sub_model(self, cell_types):
        """
        Returns a sub-system consisting of only the cell types in the cell_types
        """
        cell_types = tuple(cell_types)
        W = self.W[cell_types,:][:,cell_types]
        sigma = self.sigma[cell_types,:][:,cell_types]
        return GaussianSSNModel(W, sigma, power=self.power, w_dims=self.w_dims.copy(), wn_order=self.wn_order, period=self.period)
    
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
        
    def numerical_model(self, grid, **kwargs):
        W_discrete = self.kernel.discretize(grid) # (*shape, *shape, n, n)
        W_discrete = W_discrete.moveaxis(-2, 0).moveaxis(-1, 1+self.ndim) # (n, *shape, n, *shape)
        return nmd.MultiCellSSNModel(W_discrete, w_dims=self.w_dims, power=self.power, **kwargs)
    
    def f_prime(self, r_star):
        return self.get_f_prime(r_star, self.power)
    
    def max_eigval(self, r_star, criterion, trials=1, init_dist=torch.distributions.Normal(0.0, 1.0)):
        if not (r_star.ndim == 1 and len(r_star) == self.n):
            raise NotImplementedError("max_eigval currently only supports uniform r_star.")
        if not len(self.w_dims) == 0:
            raise NotImplementedError("max_eigval currently only supports spatial models.")
            
        device = r_star.device
        W = torch.diag(self.f_prime(r_star)) @ self.W # FW
        W_f = self.get_W_f(W, self.sigma)
        
        def loss(k):
            k = torch.tensor(k, dtype=torch.float, device=device)
            max_eigval_k = criterion(torch.linalg.eigvals(W_f(k))).max()
            return -max_eigval_k.item() # we want to maximize max_eigval_k
            
        max_eigval = -np.inf
        for _ in range(trials):
            result = minimize(loss, init_dist.sample((self.ndim,)))
            if result.success:
                cur_max_eigval = -loss(result.x)
                max_eigval = max(max_eigval, cur_max_eigval)
                
        if max_eigval == -np.inf:
            raise RuntimeError("Failed to find max eigval. Try increasing trials")

        return max_eigval
    
    def spectral_radius(self, *args, **kwargs):
        """
        Spectral radius of W of the model when it is linearized around a spatially uniform r_star.
        Returns a float
        """
        return self.max_eigval(*args, torch.abs, **kwargs)
    
    def instability(self, *args, **kwargs):
        """
        Instability of the model when it is linearized around a spatially uniform r_star
        The linearized model is unstable around r_star if self.instability(r_star) > 1
        Instability is defined as the maximum real eigenvalue of W of the linearzed network.
        Returns a float
        """
        # system is unstable around fixed point r_star if self.instability(r_star) > 1
        # instability is defined as the maximum real eigenvalue of the kernel of the linearzed network.
        return self.max_eigval(*args, torch.real, **kwargs)
        
    
class SpatialSSNModel(GaussianSSNModel):
    def __init__(self, W, sigma_s, ndim_s, sigma_f=None, **kwargs):
        assert sigma_s.ndim == W.ndim and sigma_s.shape == W.shape
        if sigma_f is not None:
            assert sigma_f.ndim == W.ndim + 1 and sigma_f.shape[:2] == W.shape
            
        super().__init__(W, self.get_sigma(W, sigma_s, ndim_s, sigma_f), **kwargs)
        
        self.sigma_s = sigma_s
        self.sigma_f = sigma_f
        self.ndim_s = ndim_s # number of spatial dimensions
        self.ndim_f = sigma_f.shape[-1] if sigma_f is not None else 0 # number of feature dimensions
        
    @staticmethod
    def get_sigma(W, sigma_s, ndim_s, sigma_f=None):
        expanded_sigma_s = sigma_s.expand((ndim_s, *W.shape)).moveaxis(0,-1)
        if sigma_f is None:
            return expanded_sigma_s
        return torch.cat([expanded_sigma_s, sigma_f],dim=-1)
    
    @property
    def sigma(self):
        return self.get_sigma(self.W, self.sigma_s, self.ndim_s, self.sigma_f)
    
    @sigma.setter
    def sigma(self, value):
        # Need to define a empty setter so that in super().__init__() the statement 'self.sigma = sigma'
        # won't throw an error.
        pass 
    
# class SpatialSSNModel(AnalyticModel):
#     def __init__(self, W, sigma_s, ndim_s, sigma_f=None, power=2, w_dims=None, wn_order=9, period=2*torch.pi):
#         super().__init__()
        
#         if w_dims is None:
#             w_dims = []
            
#         assert W.ndim == 2 and W.shape[0] == W.shape[1]
#         assert sigma_s.ndim == W.ndim and sigma_s.shape == W.shape
#         if sigma_f is not None:
#             assert sigma_f.ndim == W.ndim + 1 and sigma_f.shape[:2] == W.shape
        
#         self.W = W
#         self.sigma_s = sigma_s
#         self.sigma_f = sigma_f
#         self.ndim_s = ndim_s # number of spatial dimensions
#         self.ndim_f = sigma_f.shape[-1] if sigma_f is not None else 0 # number of feature dimensions
#         self.power = power
#         self.w_dims = w_dims
#         self.wn_order = wn_order
#         self.period = period
#         self.n = W.shape[0] # number of cell types
#         self.ndim = self.ndim_s + self.ndim_f # number of spatial/feature dimensions
    
#     @property
#     def sigma(self):
#         expanded_sigma_s = self.sigma_s.expand((self.ndim_s, *self.W.shape)).moveaxis(0,-1)
#         if self.sigma_f is None:
#             return expanded_sigma_s
#         return torch.cat([expanded_sigma_s, expanded_sigma_f],dim=-1)
    
#     def numerical_model(self, grid):
#         a_model = GaussianSSNModel(self.W, self.sigma, power=self.power, w_dims=self.w_dims, wn_order=self.wn_order, period=self.period)
#         return a_model.numerical_model(grid)