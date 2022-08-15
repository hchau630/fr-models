import abc

import numpy as np
import torch

from . import _torch, periodic, gridtools

class Kernel(abc.ABC, torch.nn.Module):
    """
    An abstract base class of 'kernels', which here just means
    F-valued functions K(x,y), where x, y in M, where
    M is a D-dimensional manifold (though right now I only
    consider manifolds that gridtools.Grid represents, namely
    manifolds that can be written as a cartesian product of 
    R^1 and S^1), such that K(x,y) is both symmetric and 
    translationally invariant, i.e. K(x,y) = f(y-x) for some f.
    """
    @property
    @abc.abstractmethod
    def F_shape(self):
        pass
    
    @property
    def F_ndim(self):
        return len(self.F_shape)
    
    @property
    @abc.abstractmethod
    def D(self):
        pass
    
    def discretize(self, grid, use_symmetry=True, normalize=True):
        # check that grid is consistent with kernel
        assert grid.D == self.D # assert number of dimensions match
        if len(grid.w_dims) > 0 or hasattr(self, 'w_dims'):
            assert grid.w_dims == self.w_dims
        
        if use_symmetry:
            expanded_Ls = [L if i in grid.w_dims else 2*L for i, L in enumerate(grid.Ls)]
            expanded_shape = tuple([shape_i if i in grid.w_dims else 2*shape_i-1 for i, shape_i in enumerate(grid.grid_shape)])
            expanded_grid = gridtools.Grid(expanded_Ls, shape=expanded_shape, w_dims=grid.w_dims, device=grid.device, dtype=grid.dtype) # (*shape)
            W_base = self.forward(expanded_grid.tensor, 0.0) # (*expanded_shape,**)

            pad = [(shape_i//2-1,shape_i//2) if i in grid.w_dims else (0,0) for i, shape_i in enumerate(grid.grid_shape)] + \
                  [(0,0) for _ in range(self.F_ndim)]
            W_base = _torch.pad(W_base, pad, mode='wrap')

            indices = []
            for i, n in enumerate(grid.grid_shape):
                indices_n = gridtools.get_grid([n,n], method='arange', device=grid.device) # (n,n,2)
                indices_n[...,1] = indices_n[...,1] + indices_n[...,0]
                indices_n = indices_n.flip([0])[...,1:] # (n,n,1)
                # print(indices_n)
                indices.append(indices_n)
            indices = torch.cat(gridtools.meshgrid(indices), dim=-1) # (n_1,n_1,n_2,n_2,...,n_D,n_D,D)
            dim_indices = [-1] + list(np.arange(self.D)*2) + list(np.arange(self.D)*2+1)
            # print(dim_indices, indices.shape)
            indices = indices.permute(*dim_indices) # (D,n_1,n_2,...,n_D,n_1,n_2,...,n_D)
            indices = tuple([*indices,...]) # ((n_1,n_2,...,n_D,n_1,n_2,...,n_D), (n_1,n_2,...,n_D,n_1,n_2,...,n_D), ..., (n_1,n_2,...,n_D,n_1,n_2,...n_D), Ellipses)

            W = W_base[indices]
        
        else:
            outer_grid_x, outer_grid_y = gridtools.meshgrid([grid.tensor,grid.tensor])
            W = self.forward(outer_grid_x,outer_grid_y)
        
        if normalize:
            return W * grid.dA
            
        return W

class K_g(Kernel):
    def __init__(self, scale, cov, normalize=True):
        assert cov.shape[-2] == cov.shape[-1] and torch.all(cov == torch.swapaxes(cov, -2, -1))
        try:
            torch.linalg.cholesky(cov)
        except RuntimeError:
            print("The last two dimensions of cov must be a PSD matrix, but the provided cov")
            print(cov)
            print("is not.")
        cov_shape = cov.shape[:-2]
        try:
            torch.broadcast_to(scale, cov_shape)
        except RuntimeError:
            print(f"scale must be broadcastable to the shape {cov_shape}, but scale has shape {scale.shape}")
        
        super().__init__()        
        self.scale = scale
        self.cov = cov
        self.normalize = normalize
        
    @property
    def F_shape(self):
        return self.cov.shape[:-2]
    
    @property
    def D(self):
        return self.cov.shape[-1]
        
    def forward(self, x, y=0.0):
        z = y - x
        input_shape = z.shape[:-1]

        cov = self.cov.reshape(-1,self.D,self.D)
        result = torch.exp(-0.5*torch.einsum('...i,kij,...j->...k',z, torch.linalg.inv(cov), z))
        if self.normalize:
            result = 1/((2*torch.pi)**self.D*torch.linalg.det(cov))**0.5 * result
        result = self.scale*result.reshape(*input_shape,*self.F_shape)
        return result
    
class K_g_space(K_g): # better name would probably be K_g_iid
    def __init__(self, ndim_s, scale, sigma, normalize=True):
        super().__init__(scale, self.compute_cov(sigma, ndim_s), normalize=normalize)
        self.sigma = sigma
        self.ndim_s = ndim_s
        
    @staticmethod
    def compute_cov(sigma, ndim_s):
        return torch.diag_embed(sigma.reshape(*sigma.shape,1).expand(*sigma.shape,ndim_s)**2)
    
    @property
    def cov(self):
        return self.compute_cov(self.sigma, self.ndim_s)
    
    @cov.setter
    def cov(self, value):
        pass
    
class K_wg(K_g):
    """
    Returns a gaussian kernel which is a wrapped gaussian along the dimensions specified in the list w_dims
    
    Shapes:
    - cov has shape (**,D,D) where ** could be empty
    - K_wg_multi(x,y) takes in arguments x and y which are broadcast-compatible.
    - Let z=x-y (might have broadcasting), then z should have shape (*,D) where * could be empty
    - Returns results with the shape (*,**)
    """
    def __init__(self, *args, w_dims=None, order=3, period=2*torch.pi, **kwargs):
        super().__init__(*args, **kwargs)
        if w_dims is None:
            w_dims = []
        self.w_dims = w_dims
        self.order = order
        self.period = period
        
    def dist(self, x, y=0):
        return periodic.dist(x, y, self.w_dims, period=self.period)
        
    def forward(self, x, y=0):
        z = self.dist(x, y)
        func = periodic.wrap(super().forward, self.w_dims, order=self.order, period=self.period)
        return func(z)
    
class K_exp(Kernel):
    """
    An exponential kernel. Note that this is not meant for an exponential distribution, which is only
    1-dimensional and is defined on the positive real axis.
    """
    
    @property
    def F_shape(self):
        return tuple()
    
    @property
    def D(self):
        return len(self.sigma)
    
    def __init__(self, scale, sigma):
        assert sigma.ndim == 1
        super().__init__()
        self.scale = scale
        self.sigma = sigma
        
    def forward(self, x, y=0):
        z = torch.abs(y-x)
        return self.scale*torch.exp(-(z/self.sigma).norm(dim=-1))
    
class WrappedKernel(Kernel):
    def __init__(self, kernel, w_dims=None, order=3, period=2*torch.pi, mode='parallel'):
        super().__init__()
        if w_dims is None:
            w_dims = []
        self.kernel = kernel
        self.w_dims = w_dims
        self.order = order
        self.period = period
        self.mode = mode
        
    @property
    def F_shape(self):
        return self.kernel.F_shape
    
    @property
    def D(self):
        return self.kernel.D
    
        
    def dist(self, x, y=0):
        return periodic.dist(x, y, self.w_dims, period=self.period)
        
    def forward(self, x, y=0):
        z = self.dist(x, y)
        func = periodic.wrap(self.kernel.forward, self.w_dims, order=self.order, period=self.period, mode=self.mode)
        return func(z)

class OuterKernel(Kernel):
    def __init__(self, K1, K2):
        assert K1.F_shape == K2.F_shape
        
        super().__init__()
        
        self.K1 = K1
        self.K2 = K2
        
    @property
    def F_shape(self):
        return self.K1.F_shape
    
    @property
    def D(self):
        return self.K1.D + self.K2.D
        
    def forward(self, x, y=0):
        x1, x2 = x[...,:self.K1.D], x[...,self.K1.D:]
        if isinstance(y, torch.Tensor) and y.ndim > 0: # y is not a scalar, its last diension should be equal to self.D
            assert y.shape[-1] == self.D
            y1, y2 = y[...,:self.K1.D], y[...,self.K1.D:]
        else:
            y1, y2 = y, y
        return self.K1(x1, y1) * self.K2(x2, y2)
    
class ProductKernel(Kernel):
    def __init__(self, K1, K2):
        assert K1.F_shape == K2.F_shape
        assert K1.D == K2.D
        
        super().__init__()
        
        self.K1 = K1
        self.K2 = K2
        
    @property
    def F_shape(self):
        return self.K1.F_shape
    
    @property
    def D(self):
        return self.K1.D
    
    def forward(self, x, y=0):
        return self.K1(x, y) * self.K2(x, y)
       
class SumKernel(Kernel):
    def __init__(self, K1, K2):
        assert K1.F_shape == K2.F_shape
        assert K1.D == K2.D
        
        super().__init__()
        
        self.K1 = K1
        self.K2 = K2
        
    @property
    def F_shape(self):
        return self.K1.F_shape
    
    @property
    def D(self):
        return self.K1.D
    
    def forward(self, x, y=0):
        return self.K1(x, y) + self.K2(x, y)
       