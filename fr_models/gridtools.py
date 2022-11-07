import numpy as np
import torch

from . import _torch
import utils

class Grid(torch.Tensor):
    def __new__(cls, *args, data=None, require_grads=False, **kwargs):
        # Reference: https://discuss.pytorch.org/t/subclassing-torch-longtensor/100377/3
        if data is None:
            data = get_grid(*args, **kwargs)
        return torch.Tensor._make_subclass(cls, data, require_grads) 
    
    def __init__(self, extents, shape=None, dxs=None, w_dims=None, **kwargs):
        if w_dims is None:
            w_dims = []

        self._extents = [extent if isinstance(extent, tuple) else (-extent/2, extent/2) for extent in extents]
        self._dxs = dxs if dxs is not None else get_dxs(extents, shape, w_dims=w_dims)
        self._dA = np.prod(self._dxs)
        self._w_dims = w_dims
        
    @classmethod
    def from_x(cls, x, w_dims=None, check_valid=True, rtol=None, atol=None, device=None):
        if w_dims is None:
            w_dims = []
        
        if x.ndim == 1:
            x = x[:,None]
            
        assert x.ndim - 1 == x.shape[-1]
        D = x.shape[-1]
            
        extents = [[x[...,i].min().item(), x[...,i].max().item()] for i in range(D)]
        for i in w_dims:
            extents[i][1] = extents[i][1] + (extents[i][1] - extents[i][0])/(x.shape[i] - 1)
        extents = [tuple(extent) for extent in extents]
        grid = cls(extents, shape=x.shape[:-1], w_dims=w_dims, device=device)
        
        if check_valid:
            try:
                torch.testing.assert_close(grid.tensor, x, rtol=rtol, atol=atol)
            except AssertionError as err:
                for i in range(grid.D):
                    print(f"Dimension: {i}")
                    print("original: ", slice_coord(x, i))
                    print("new: ", slice_coord(grid.tensor, i))
                raise err
        
        return grid
        
    @property
    def tensor(self):
        return self.as_subclass(torch.Tensor) # return a pure torch.Tensor without all the extra attribute
    
    @property
    def grid_shape(self):
        return self.tensor.shape[:-1]
    
    @property
    def D(self):
        return len(self.grid_shape)
    
    @property
    def extents(self):
        return self._extents
    
    @property
    def Ls(self):
        return [extent[1] - extent[0] for extent in self.extents]
    
    @property
    def dxs(self):
        return self._dxs
    
    @property
    def dA(self):
        return self._dA
    
    @property
    def w_dims(self):
        return self._w_dims
    
    @property
    def mids(self):
        return get_mids(self.grid_shape, w_dims=self.w_dims) # throws AssertionError if grid does not have a point that lies exactly in the middle
    
    def slice(self, i):
        return slice_coord(self.tensor, i)
    
    def to(self, *args, **kwargs):
        new_tensor = self.tensor.to(*args, **kwargs)
        return self.__class__(data=new_tensor, extents=self.extents, shape=self.grid_shape, w_dims=self.w_dims)
    
    def cpu(self, *args, **kwargs):
        new_tensor = self.tensor.cpu(*args, **kwargs)
        return self.__class__(data=new_tensor, extents=self.extents, shape=self.grid_shape, w_dims=self.w_dims)
        
    def clone(self, *args, **kwargs):
        new_tensor = self.tensor.clone(*args, **kwargs)
        return self.__class__(data=new_tensor, extents=self.extents, shape=self.grid_shape, w_dims=self.w_dims)
    
    def double(self, *args, **kwargs):
        new_tensor = self.tensor.double(*args, **kwargs)
        return self.__class__(data=new_tensor, extents=self.extents, shape=self.grid_shape, w_dims=self.w_dims)
    
    def float(self, *args, **kwargs):
        new_tensor = self.tensor.float(*args, **kwargs)
        return self.__class__(data=new_tensor, extents=self.extents, shape=self.grid_shape, w_dims=self.w_dims)
#     @staticmethod
#     def meshgrid(grids):
#         tensors = [grid.points for grid in grids]
#         new_tensor = torch.stack(meshgrid(tensors),dim=-1)
#         new_Ls = [L for grid in grids for L in grid.Ls]
#         new_shape = [n for grid in grids for n in grid.shape]
        
#         new_w_dims = []
#         cum_ndim = 0
#         for i, grid in enumerate(grids):
#             new_w_dims += [dim + cum_ndim for dim in grid.w_dims]
#             cum_ndim += grid.ndim
            
#         return Grid(new_Ls, new_shape, w_dims=new_w_dims, points=new_tensor)

def get_dA(*args, **kwargs):
    """
    returns scalar
    """
    return np.prod(get_dxs(*args, **kwargs))

def get_dxs(extents, shape, w_dims=None):
    """
    returns list
    """
    Ls = [extent[1]-extent[0] if isinstance(extent, tuple) else extent for extent in extents]
    if w_dims is None:
        w_dims = []
    return (np.array(Ls)/np.array([shape[i] if i in w_dims else shape[i]-1 for i in range(len(shape))])).tolist()

def get_mids(shape, w_dims=None):
    """
    Returns the index of the center of shape.
    Raises AssertionError if the center of shape does not coincide with a particular index
    """
    if w_dims is None:
        w_dims = []
    D = len(shape)
    if not all([shape[d] % 2 == 0 if d in w_dims else shape[d] % 2 == 1 for d in range(D)]):
        raise ValueError(f"the given shape {shape} with w_dims {w_dims} does not have a center point")
    return tuple([shape[d]//2 if d in w_dims else (shape[d]-1)//2 for d in range(D)])

def get_grid(extents, shape=None, dxs=None, w_dims=None, method='linspace', device='cpu', dtype=None):
    """
    Get a grid (i.e. a tensor with shape (n_1, n_2, ..., n_N, n_1+...+n_N) where grid[*idx,:] are coordinates)
    by specifying the extents in each dimension and the shape (n_1, ..., n_N).
    
    Parameters:
        extents: list of tuples of scalars (2,) or list of scalars - If list of tuples, each i-th tuple
                 indicates the lower and upper bound of the i-th dimension. If list of scalars, the lower
                 and upper bound are interpreted as (-scalar/2, scalar/2) for method='linspace' and
                 (0, scalar) for method='arange'.
        shape: tuple - shape of the grid, ignored if method is not 'linspace'
        dxs: tuple - step sizes in each dimension, ignored if method is not 'arange'. If None, defaults to
             step sizes of 1 along each dimension.
        w_dims: list of ints - a list of the dimensions along which the endpoint is not included, which is
                useful for dimensions which are periodic/wrapped (w in w_dims stands for wrapped). Ignored
                if method='arange'.
        method: 'linspace' or 'arange'. Specifies whether torch.linspace or torch.arange is used.
        device: The device on which the grid is created.
        
    Returns:
        grid: torch.Tensor
        
    """
    assert all([len(extent) == 2 for extent in extents if isinstance(extent, tuple)])
    if method == 'linspace':
        if shape is None:
            raise ValueError("shape must be provided when method='linspace'")
        else:
            assert len(extents) == len(shape)
    if method == 'arange':
        if dxs is None:
            dxs = [1 for _ in range(len(extents))]
        else:
            assert len(extents) == len(dxs)
    if w_dims is None:
        w_dims = []
    else:
        assert isinstance(w_dims, list)
    if len(w_dims) > 0:
        assert max(w_dims) < len(extents)
    
    if method == 'linspace':
        extents = [extent if isinstance(extent, tuple) else (-extent/2, extent/2) for extent in extents]
        endpoints = [False if i in w_dims else True for i in range(len(extents))]
        grids_per_dim = [_torch.linspace(extent[0],extent[1],N,endpoint,device=device,dtype=dtype) for extent, N, endpoint in zip(extents, shape, endpoints)]
    elif method == 'arange':
        extents = [extent if isinstance(extent, tuple) else (0, extent) for extent in extents]
        grids_per_dim = [torch.arange(extent[0],extent[1],dx,device=device,dtype=dtype) for extent, dx in zip(extents, dxs)]
    else:
        raise NotImplementedError()
        
    grid = torch.stack(torch.meshgrid(*grids_per_dim, indexing='ij'), dim=-1)
    return grid

def meshgrid(tensors):
    """
    A generalization of torch.meshgrid
    Mesh together list of tensors of shapes (n_1_1,...,n_1_{M_1},N_1), (n_2_1,...,n_2_{M_2},N_2), ...
    Returns tensors of shapes (n_1_1,...,n_1_{M_1},n_2_1,...,n_2_{M_2},...,N_1), (n_2_1,...,n_2_{M_2},...,N_2)
    """
    sizes = [list(tensor.shape[:-1]) for tensor in tensors] # [[n_1,...,n_{M_1}],[n_1,...,.n_{M_2}],...]
    Ms = np.array([tensor.ndim - 1 for tensor in tensors]) # [M_1, M_2, ...]
    M_befores = np.cumsum(np.insert(Ms[:-1],0,0))
    M_afters = np.sum(Ms) - np.cumsum(Ms)
    Ns = [tensor.shape[-1] for tensor in tensors]
    shapes = [[1]*M_befores[i]+sizes[i]+[1]*M_afters[i]+[Ns[i]] for i, tensor in enumerate(tensors)]
    expanded_tensors = [tensor.reshape(shapes[i]).expand(utils.itertools.flatten_seq(sizes)+[Ns[i]]) for i, tensor in enumerate(tensors)]
    return expanded_tensors

def slice_coord(tensor, i):
    """
    Assume tensor has shape (n_1, n_2, ..., n_N, n_1+...+n_N)
    which is a stacked meshgrid of 1D tensors.
    Then slice_coord(tensor,i) returns the slice (0,...,:,0,...,0,i)
    where : is at the i-th index.
    """
    N = tensor.ndim - 1
    return tensor[tuple([0]*i + [slice(None)] + [0]*(N-i-1))][:,i]