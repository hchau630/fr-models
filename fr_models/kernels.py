import abc

import numpy as np
import torch
import torch.nn.functional as F

from . import _torch, periodic, gridtools

class Kernel(abc.ABC, torch.nn.Module):
    def discretize(self, grid):
        outer_grid_x, outer_grid_y = gridtools.meshgrid([grid,grid])
        W = self.forward(outer_grid_x,outer_grid_y)*grid.dA
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
        scale = _torch.atleast_0d(scale)
        try:
            torch.broadcast_to(scale, cov_shape)
        except RuntimeError:
            print(f"scale must be broadcastable to the shape {cov_shape}, but scale has shape {scale.shape}")
        
        super().__init__()
        self.D = cov.shape[-1]
        self.cov_shape = cov_shape
        
        self.scale = scale
        self.cov = cov
        self.normalize = normalize
        
    def forward(self, x, y=0.0):
        z = y - x
        input_shape = z.shape[:-1]
        
        cov = self.cov.reshape(-1,self.D,self.D)
        result = torch.exp(-0.5*torch.einsum('...i,kij,...j->...k',z, torch.linalg.inv(cov), z))
        if self.normalize:
            result = 1/((2*torch.pi)**self.D*torch.linalg.det(cov))**0.5 * result
        result = self.scale*result.reshape(*input_shape,*self.cov_shape)
        return result
    
class K_wg(K_g):
    """
    Returns a gaussian kernel which is a wrapped gaussian along the dimensions specified in the list w_dims
    
    Shapes:
    - cov has shape (**,D,D) where ** could be empty
    - K_wg_multi(x,y) takes in arguments x and y which are broadcast-compatible.
    - Let z=x-y (might have broadcasting), then z should have shape (*,D) where * could be empty
    - Returns results with the shape (*,**)
    """
    def __init__(self, *args, w_dims=[], order=3, period=2*torch.pi, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_dims = w_dims
        self.order = order
        self.period = period
        
    def forward(self, x, y=0):
        z = periodic.dist(x, y, self.w_dims, period=self.period)
        func = periodic.wrap(super().forward, self.w_dims, order=self.order, period=self.period)
        return func(z)

def discretize_K(K, Ls, shape, w_dims=[], sigma=0, device='cpu'):
    """
    Discretizes the kernel K(x,y) for a grid of neurons with shape SHAPE and lengths Ls
    Kernel is a torch.nn.Module, whose forward function is a function of 
    two torch.Tensors that returns a torch.Tensor with any shape (**),
    and accepts batched x and y where the shape of y-x is denoted (*), 
    such that the batched result has shape (*,**).
    Returns a torch.Tensor with shape (*shape, *shape, **)
    
    Speed benchmark:
    When discretizing a kernel with shape (50,50), we have the following times:
      - device='cpu': ~0.21s
      - device='cuda': ~0.04s
      
    IMPORTANT: BECAUSE PYTORCH DOES NOT HAVE A FUNCTIONAL PAD, I AM CURRENTLY USING np.pad
    IF devce='cpu', WHICH MEANS GRADIENT DOES NOT PASS THROUGH THIS FUNCTION.
    """ 
    assert len(Ls) == len(shape)
    N = np.prod(shape)
    D = len(shape)
    dA = gridtools.get_grid_size(Ls, shape, w_dims=w_dims)
    
    K.to(device)
    
    if device == 'cpu' or device == torch.device('cpu'): # significantly faster on cpu
        expanded_Ls = [L if i in w_dims else 2*L for i, L in enumerate(Ls)]
        expanded_shapes = tuple([shape[i] if i in w_dims else 2*shape[i]-1 for i in range(D)])
        grid = gridtools.get_grid(expanded_Ls, expanded_shapes, w_dims) # (*shape)
        W_base = K(grid, 0.0)*dA # (*shape,**)
        K_shape = W_base.shape[D:] # (**)
        pad = [(shape[i]//2,shape[i]//2) if i in w_dims else (0,0) for i in range(D)] + [(0,0) for _ in range(len(K_shape))]
        expanded_W_base = torch.from_numpy(np.pad(W_base.detach().numpy(), pad, mode='wrap')) # !! gradient cannot pass here !!
        # pad = utils.itertools.flatten_seq([(shape[i]//2,shape[i]//2) if i in w_dims else (0,0) for i in range(D)])
        # pad += utils.itertools.flatten_seq([(0,0) for _ in range(len(K_shape))])
        # expanded_W_base = F.pad(W_base, tuple(pad[::-1]), mode='circular')
        W = torch.zeros((*shape, *shape, *K_shape))
        mids = gridtools.get_mids(expanded_W_base.shape[:D], w_dims)
        
        for ndidx in np.ndindex(shape):
            indices = tuple([slice(mids[i]-ndidx[i], mids[i]-ndidx[i]+shape[i]) for i in range(D)])
            W[ndidx] = expanded_W_base[indices]
    else:
        grid = gridtools.get_grid(Ls, shape, w_dims, device=device)
        outer_grid_x, outer_grid_y = gridtools.meshgrid([grid,grid])
        W = K(outer_grid_x,outer_grid_y)*dA
            
    if sigma != 0:
        assert sigma > 0
        W += torch.normal(0,sigma/np.sqrt(N),size=(*shape,*shape),device=device) # can add cell-type specific sigma later

    return W

def discretize_nK(nK, Ls, shape, w_dims=[], device='cpu'):
    """
    Discretize a size (n,n) np.ndarray of kernels K(x,y), i.e. a matrix-valued function.
    """
    assert nK.ndim == 2 and nK.shape[0] == nK.shape[1]
    n = nK.shape[0]
    nK_discrete = _torch.tensor([
        [discretize_K(nK[i,j], Ls, shape, w_dims=w_dims, device=device) for j in range(n)] for i in range(n)
    ])
    nK_discrete = torch.moveaxis(nK_discrete, 1, 1+len(shape))
    return nK_discrete