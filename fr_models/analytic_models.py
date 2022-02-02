import abc
import dataclasses
import typing

import numpy as np
import torch

from fr_models import numerical_models as nmd
from fr_models import utils

@dataclasses.dataclass
class AnalyticModel(abc.ABC):
    @abc.abstractmethod
    def numerical_model(self, *args):
        pass
    
    def named_parameters(self):
        return dataclasses.asdict(self)
    
@dataclasses.dataclass
class GaussianSSNModel(AnalyticModel):
    W: torch.Tensor
    sigma: torch.Tensor
    power: int = 2
    w_dims: list[int] = dataclasses.field(default_factory=list)
    wn_order: int = 9
    period: typing.Union[float, list[float]] = 2*torch.pi
    
    def __post_init__(self):
        assert self.W.ndim == 2 and self.W.shape[0] == self.W.shape[1]
        assert self.sigma.ndim == self.W.ndim + 1 and self.sigma.shape[:2] == self.W.shape
        self.n = self.W.shape[0] # number of cell types
        self.ndim = self.sigma.shape[-1] # number of spatial/feature dimensions
        cov = torch.diag_embed(self.sigma**2)
        self.kernel = utils.kernels.K_wg(self.W, cov, w_dims=self.w_dims, order=self.wn_order, period=self.period)
    
    def numerical_model(self, Ls, shape, **kwargs):
        assert len(Ls) == len(shape) == self.ndim
        W_discrete = utils.kernels.discretize_K(self.kernel, Ls, shape, w_dims=self.w_dims, **kwargs) # (*shape, *shape, n, n)
        W_discrete = torch.moveaxis(W_discrete, -2, 0) # (n, *shape, *shape, n)
        W_discrete = torch.moveaxis(W_discrete, -1, 1+self.ndim) # (n, *shape, n, *shape)
        return nmd.MultiCellSSNModel(W_discrete, w_dims=self.w_dims, power=self.power)
    