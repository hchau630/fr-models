import functools

import numpy as np
import torch

from . import _torch

def wrap(f, w_dims, order=3, period=2*torch.pi):
    """
    f is a function from R^D to R. 
    If D = 1 and w_dims = [0],
    returns a wrapped version of f, f_wrapped, where f_wrapped(x) = sum_{n=-(order-1)/2}^{(order-1)/2} f(x+period*n)
    For higher dimensional cases, each dimension specified in w_dims is wrapped in the manner described in the 1D case.
    period can be either a scalar or a list. If a list, period[i] is interpreted as the period of the dimension w_dims[i].
    """
    if len(w_dims) == 0:
        return f

    w_dims = torch.as_tensor(w_dims)
    period = torch.atleast_1d(torch.as_tensor(period))
    n = torch.arange(order)-(order-1)//2 # e.g. if order = 3, we have [-1,0,1]
    
    assert order % 2 == 1
    assert w_dims.ndim == period.ndim == 1
    if len(period) == 1:
        period = period.expand_as(w_dims)
    assert w_dims.shape == period.shape
        
    @functools.wraps(f)
    def f_wrapped(x, w_dims=w_dims, n=n, period=period):
        device = x.device
        
        w_dims = w_dims.to(device)
        n = n.to(device)
        period = period.to(device)
        
        assert torch.all(torch.abs(x[...,w_dims]) <= period/2)
        
        x[...,w_dims] = torch.abs(x[...,w_dims]) # take absolute value of values along dimensions specified by w_dims
        
        for i in range(len(w_dims)):
            basis = torch.zeros(x.shape, device=device)
            basis[...,w_dims[i]] = 1.0
            x = x + torch.einsum('i,...j->i...j',period[i]*n, basis)
            
        result = f(x).sum(dim=tuple(np.arange(len(w_dims))))
        return result

    return f_wrapped

def dist(x, y, w_dims, period=2*torch.pi):
    """
    Returns |y-x| for dimensions not in w_dims and min{|y-x|, period-|y-x|} otherwise.
    For dimensions in w_dims, x and y must satisfy -period/2 <= x, y <= period/2.
    """
    if len(w_dims) == 0:
        return torch.abs(x-y)
    
    _x, _y = torch.broadcast_tensors(torch.as_tensor(x), torch.as_tensor(y))
    w_dims = torch.as_tensor(w_dims)
    period = torch.atleast_1d(torch.as_tensor(period))
    assert w_dims.ndim == period.ndim == 1
    if len(period) == 1:
        period = period.expand_as(w_dims)
    assert w_dims.shape == period.shape
    
    z = torch.abs(x - y)
    D = z.shape[-1]
    device = z.device
    
    _x = _x.to(device)
    _y = _y.to(device)
    w_dims = w_dims.to(device)
    period = period.to(device)
    
    assert torch.all(-period/2 <= _x[...,w_dims]) and torch.all(_x[...,w_dims] <= period/2)
    assert torch.all(-period/2 <= _y[...,w_dims]) and torch.all(_y[...,w_dims] <= period/2)
    
    one = torch.zeros(D, device=device)
    one[w_dims] = 1.0
    
    return torch.minimum(z, one*period - z)