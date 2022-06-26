import torch

def riemann_sum(y, dxs):
    """
    Compute n-dimensional integral using left riemann sum.
    This is the fastest, though most inaccurate, way of computing an integral
    y - (*,N_1-1,...,N_n-1)
    dxs - (n)
    N is the number of sample points along each dimension
    """
    n = len(dxs)
    dA = torch.prod(dxs)
    return y.sum(dim=tuple(-torch.arange(n)))*dA