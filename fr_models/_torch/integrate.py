import torch
import numpy as np

from fr_models import gridtools

def definite(f, extents, shape, method='riemann_sum', device=None):
    if method == 'riemann_sum':
        dxs = [(extent[1] - extent[0])/N for extent, N in zip(extents, shape)]
        extents = [(extent[0] + dx/2, extent[1] - dx/2) for extent, dx in zip(extents, dxs)]
        grid = gridtools.Grid(extents, shape=shape, device=device)
        y = f(grid.tensor)
        return riemann_sum(y, dxs)
    else:
        raise NotImplementedError()

def riemann_sum(y, dxs, reduce_dims='first'):
    """
    Compute n-dimensional integral using riemann sum.
    This is the fastest, though most inaccurate, way of computing an integral
    y - (N_1-1,...,N_n-1,*)
    dxs - (n)
    N is the number of sample points along each dimension
    """
    n = len(dxs)
    dA = np.prod(dxs)
    if reduce_dims == 'first':
        reduce_dims = tuple(torch.arange(n))
    if reduce_dims == 'last':
        reduce_dims = tuple(-torch.arange(n)-1)
    return y.sum(dim=reduce_dims)*dA
    