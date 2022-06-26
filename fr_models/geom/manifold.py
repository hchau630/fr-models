import abc

import torch
import numpy as np

class Manifold(abc.ABC):
    """
    Abstract manifold base class. I wanted to use geomstats.geometry
    but their Hypersphere class intrinsic coordinate representation is
    too broken right now.
    """
    def __init__(self, ndim):
        self.ndim = ndim
    
    @property
    def origin(self):
        raise RuntimeError("Manifold does not have origin")
        
    @abc.abstractmethod
    def ni(self, x):
        """
        whether x is in Manifold ('ni' is the latex command for backwards inclusion, since it is the reverse of 'in')
        result has shape x.shape[:-1]
        """
        pass
        
    @abc.abstractmethod
    def dist(self, x, y=None):
        """
        Returns distance between x and y on manifold.
        If we want to be fancy, we should create a Metric
        class like geomstats.geometry and dist would be
        a Metric method, but let's not be too fancy right now.
        If y is None and manifold has attribute 'origin', then
        calculate distance of x from origin. If manifold does not
        have origin, raise error.
        """
        pass

class EuclideanManifold(Manifold):
    def __init__(self, ndim):
        super().__init__(ndim)
        self._origin = 0.0
        
    @property
    def origin(self):
        return torch.zeros(self.ndim)
        
    def ni(self, x):
        return torch.ones(x.shape[:-1], device=x.device).bool()
        
    def dist(self, x, y=None):
        if y is None:
            y = self._origin
        return torch.linalg.norm(x - y, dim=-1)
    
class CircleManifold(Manifold):
    """
    A circle manifold. Points are represented in intrinsic coordinates, meaning
    a point x is in circle if x is a scalar between [-np.pi, np.pi]
    """
    def __init__(self, period=2*np.pi):
        super().__init__(1)
        self.period = period
        self._origin = 0.0
        
    @property
    def origin(self):
        return torch.zeros(1)
        
    def ni(self, x):
        return (-self.period/2 <= x) & (x <= self.period/2)
        
    def dist(self, x, y=None):
        assert self.ni(x).all(), 'x must be on circle'
        
        if y is None:
            y = self._origin
        else:
            assert self.ni(y).all(), 'y must be on circle'
            
        z = (y - x).abs()
        
        return torch.minimum(z, self.period - z)
        
class ProductManifold(Manifold):
    def __init__(self, manifolds):
        assert isinstance(manifolds, list) and len(manifolds) > 1
        
        ndims = [manifold.ndim for manifold in manifolds]
        ndim = sum(ndims)
        
        super().__init__(ndim)
        
        self.manifolds = manifolds
        self.nmanifolds = len(manifolds)
        self.ndims = ndims
        self.cumdims = np.cumsum(self.ndims).tolist()
        
    def split(self, x):
        _cumdims = [0] + self.cumdims
        return [x[...,i:j] for i, j in zip(_cumdims[:-1], _cumdims[1:])]
        
    @property
    def origin(self):
        return torch.cat([manifold.origin for manifold in self.manifolds])
    
    def ni(self, x):
        xs = self.split(x)
        nis = torch.stack([manifold.ni(xi) for manifold, xi in zip(self.manifolds, xs)], dim=-1)
        return torch.all(nis, dim=-1)
    
    def dist(self, x, y=None):
        xs = self.split(x)
        if y is None:
            ys = [None]*self.nmanifolds
        else:
            ys = self.split(y)
            
        dists = torch.stack([manifold.dist(xi, yi) for manifold, xi, yi in zip(self.manifolds, xs, ys)], dim=-1)
        return torch.linalg.norm(dists, dim=-1)