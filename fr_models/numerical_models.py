import abc
from functools import partial

from torchdiffeq import odeint
import numpy as np
import torch

from fr_models import utils

class NumericalModel(abc.ABC, torch.nn.Module):
    def __init__(self, W, tau=1.0):
        assert W.ndim == 2
        super().__init__()
        
        self.tau = tau
        self.N = W.shape[0]
        self.W = torch.nn.Parameter(W)
        
    @abc.abstractmethod
    def _drdt(self, t, r, h):
        return
        
    def forward(self, h, r0, t, **kwargs):
        """
        Solves an ODE with input h and initial state r0 at time stamps t.
        Args:
            h: a function that maps a 0-dimensional tensor t to a tensor with shape (N,),
               or a tensor with shape (N,) that is interpreted as constant input
            r0: a 1-dimensional tensor with shape (N,) or a 0-dimensional tensor, which is interpreted
               to be constant across neurons
            t: a 1-dimensional tensor that specifies the time stamps at which the ODE is solved,
               and whose first element represents the initial time,
               or a 0-dimensional tensor that specifies the end time, with the initial time interpreted as 0.
               
        Returns:
            r: If t is a 1-dimensional tensor with length L, r is a tensor with shape (L, N), with r[i]
               being the firing rates at t[i].
               If t is a 0-dimensional tensor, r is tensor with shape (N,) that represents the firing rates at t.
        """
        if not callable(h):
            _h = lambda t: h
        else:
            _h = h
            
        if t.ndim == 0: # if t is a scalar tensor
            _t = torch.Tensor([0,t]).to(t.device)
        else:
            _t = t
            
        if r0.ndim == 0: # if r is a scalar tensor
            _r0 = r0.expand((self.N,))
        else:
            _r0 = r0
            
        drdt = partial(self._drdt, h=_h)
        r = odeint(drdt, _r0, _t, **kwargs)

        if t.ndim == 0:
            return r[-1]
        return r
        
class MultiDimModel(NumericalModel):
    def __init__(self, W, *args, **kwargs):
        assert W.ndim % 2 == 0
        
        shape = W.shape[:W.ndim // 2]
        _W = W.reshape(np.prod(shape), np.prod(shape))
        
        super().__init__(_W, *args, **kwargs)
        
        self.shape = shape
        self.ndim = W.ndim // 2
        self.W_expanded = W
        
    def foward(self, h, r0, t, **kwargs):
        r0 = r0.reshape(-1)
        
        if not callable(h):
            _h = h.reshape(-1)
        else:
            _h = lambda t: h(t).reshape(-1)
        
        r = super().foward(_h, r0, t) # (L, N) or (N)

        if t.ndim == 0:
            r = r.reshape(*self.shape)
        else:
            r = r.reshape(len(t), *self.shape)
            
        return r
    
class TrivialVBModel(MultiDimModel):
    """
    A model with additional topological data. The model must be a trivial vector bundle E = B x F 
    over a manifold B that is expressable as an arbitrary, finite cartesian product of R^1 or S^1 
    with the usual metric. The fibre F is a vector space.
    This should be an abstract yet specific enough base class for any reasonable models of the cortex.
    """
    def __init__(self, W, F_dim, w_dims, *args, **kwargs):
        """
        Args:
          - W: The weight tensor, with the first F_dim dimensions corresponding to the fiber space F, 
               and the rest of the dimensions correspond to the base space B.
          - F_dim: Dimensionality of the fiber space. 
          - w_dims: a list of integers denoting the dimensions (within base space B) which are 'wrapped',
                    i.e. are S^1.
        """
        super().__init__(W, *args, **kwargs)
        self.F_dim = F_dim
        self.B_dim = self.ndim - F_dim
        assert self.F_dim >= 0 and self.B_dim >= 0
        self.F_shape = self.shape[:F_dim]
        self.B_shape = self.shape[F_dim:]
        self.F_N = np.prod(self.F_shape)
        self.B_N = np.prod(self.B_shape)
        self.w_dims = w_dims
        assert all([w_dim < self.B_dim for w_dim in self.w_dims])
        
    def get_h(amplitude, F_idx, B_idx=None, device='cpu'):
        """
        Returns an input vector h where the neurons with F index F_idx and B index B_idx
        is are given input with input strength amplitude.
        Args:
          - amplitude: scalar input strength
          - F_idx: Either a scalar integer, 1-dimensional array-like, or 
                   2-dimensional array-like object. Array-like means either tensor, 
                   tuple, or list. If 2-dimensional, then F_idx.shape[0] is the number 
                   of neurons with non-zero input. If a scalar, then F must be
                   1-dimensional.
          - B_idx: Similar to F_idx, but if None, then B_idx will be the index of the
                   neuron at the origin of the B space.
        """
        h = np.zeros(self.shape, device=device)
        
        F_idx = torch.tensor(F_idx, device=device)
        if B_idx is None:
            B_idx = utils.get_mids(self.B_shape, w_dims=self.w_dims)
        B_idx = torch.tensor(B_idx, device=device)
        F_idx = torch.atleast_2D(F_idx)
        B_idx = torch.atleast_2D(B_idx)
        
        h[(*F_idx.T,*B_idx.T)] = amplitude
        
        return h
    
class MultiCellModel(TrivialVBModel):
    def __init__(self, W, *args, **kwargs):
        assert W.ndim >= 4
        super().__init__(W, 1, *args, **kwargs)
                
class FRModel(NumericalModel):
    def __init__(self, W, f, *args, **kwargs):
        """
        User should make sure that f is a function that is compatible with torch.Tensor and not np.ndarray
        """
        super().__init__(W, *args, **kwargs)
        self.f = f
        
    def _drdt(self, t, r, h):
        return self.f(self.W @ r + h(t)) - r
    
class MultiCellFRModel(MultiCellModel, FRModel):
    pass
    
class SSNModel(FRModel):
    def __init__(self, W, power=2, *args, **kwargs):
        f = lambda x: torch.clip(x,0,None)**power
        super().__init__(W, f, *args, **kwargs)
        
class MultiCellSSNModel(MultiCellModel, SSNModel):
    pass
        
class LinearizedMultiCellSSNModel(MultiCellFRModel):
    def __init__(self, W, r_star, *args, **kwargs):
        assert torch.all(r_star >= 0)
        super().__init__(W, None, *args, **kwargs) # f is defined dynamically
        self._r_star = torch.nn.Parameter(r_star)
        
    @property
    def _f_prime(self):
        return 2*self._r_star*0.5
    
    @property
    def r_star(self):
        device = self._r_star.device
        return torch.einsum('i,...->i...', self._r_star, torch.ones(self.B_shape, device=device))
        
    @property
    def f_prime(self):
        return 2*self.r_star**0.5
    
    @property
    def f(self):
        return lambda x: self.f_prime * x
    
    def foward(self, delta_h, delta_r0, t, **kwargs):
        delta_r = super().forward(delta_h, delta_r0, t, **kwargs)
        return delta_r
        
class PerturbedMultiCellSSNModel(MultiCellSSNModel):
    def __init__(self, W, r_star, *args, **kwargs):
        assert torch.all(r_star >= 0)
        super().__init__(W, *args, **kwargs)
        self._r_star = torch.nn.Parameter(r_star)
        
    @property
    def r_star(self):
        device = self._r_star.device
        return torch.einsum('i,...->i...', self._r_star, torch.ones(self.B_shape, device=device))
        
    @property
    def h_star(self):
        r_star = self.r_star.reshape(-1)
        return (r_star**0.5 - self.W @ r_star).reshape(self.shape)
    
    def foward(self, delta_h, delta_r0, t, **kwargs):
        if not callable(delta_h):
            h = self.h_star + delta_h
        else:
            h = lambda t: self.h_star + delta_h(t)
            
        r0 = self.r_star + delta_r0
            
        r = super().forward(h, r0, t, **kwargs)
        
        delta_r = r - self.r_star
        
        return delta_r
    