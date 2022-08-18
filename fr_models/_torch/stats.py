import numbers
import functools

import torch
import numpy as np

from fr_models import gridtools

def bucketize(y, N_bins=50, mode='edge', ymin=None, ymax=None, device=None):
    if ymin is None:
        ymin = y.min().item()
    if ymax is None:
        ymax = y.max().item()
        
    if mode == 'edge':
        grid_edges = gridtools.Grid([(ymin, ymax)], shape=(N_bins+1,), device=device)
        indices = torch.bucketize(y, grid_edges.tensor.squeeze(), right=True) - 1
        indices[indices == N_bins] = N_bins - 1
        
    elif mode == 'mid':
        dx = (ymax - ymin)/(N_bins - 1)
        grid_edges = gridtools.Grid([(ymin-dx/2, ymax+dx/2)], shape=(N_bins+1,), device=device)
        indices = torch.bucketize(y, grid_edges.tensor.squeeze(), right=True) - 1
        
    elif mode == 'wrap':
        grid_centers, indices = bucketize(y, N_bins=N_bins+1, mode='mid', ymin=ymin, ymax=ymax, device=device)
        indices[indices == N_bins] = 0 # Treat last bin and first bin as the same bin
        grid_centers = gridtools.Grid(grid_centers.extents, shape=(N_bins,), w_dims=[0], device=device)
        return grid_centers, indices
    
    else:
        raise NotImplementedError()
        
    grid_centers = gridtools.Grid([(grid_edges.extents[0][0] + grid_edges.dxs[0]/2, grid_edges.extents[0][1] - grid_edges.dxs[0]/2)], shape=(N_bins,), device=device)
    
    return grid_centers, indices

def bucketize_n(y, mode=None, grid=None, shape=None, ymin=None, ymax=None, device=None):
    if grid is not None:
        assert (mode is None) and (shape is None) and (ymin is None) and (ymax is None)
        mode = ['mid'] * grid.D
        for w_dim in grid.w_dims:
            mode[w_dim] = 'wrap'
        shape = grid.grid_shape
        ymin = [extent[0] for extent in grid.extents]
        ymax = [extent[1] for extent in grid.extents]
    
    if shape is None:
        shape = (50,)
    
    n = len(shape)
    assert y.shape[-1] == n
    
    if mode is None:
        mode = ['edge'] * n
    elif isinstance(mode, str):
        mode = [mode] * n
    else:
        assert len(mode) == n
        
    if ymin is None or isinstance(ymin, numbers.Number):
        ymin = [ymin] * n
    if ymax is None or isinstance(ymax, numbers.Number):
        ymax = [ymax] * n
    
    all_grids = []
    all_indices = []
    
    for i in range(n):
        grid, indices = bucketize(y[...,i], N_bins=shape[i], mode=mode[i], ymin=ymin[i], ymax=ymax[i], device=device)
        all_grids.append(grid)
        all_indices.append(indices)
        
    new_extents = [(grid.extents[0][0], grid.extents[0][1]) for grid in all_grids]
    new_w_dims = [i for i, grid in enumerate(all_grids) if len(grid.w_dims) == 1]
    grid = gridtools.Grid(new_extents, shape=shape, w_dims=new_w_dims, device=device)
    indices = torch.stack(all_indices, dim=-1)
    
    return grid, indices

def bin_values_n(y, values, shape=(50,), device=None, **bucketize_kwargs):
    n = len(shape)
    assert y.shape[:-1] == values.shape
    
    grid, indices = bucketize_n(y, shape=shape, device=device, **bucketize_kwargs)
    binned_values = np.empty(shape, dtype=object)
    masks = np.empty(n, dtype=object)
    
    for i in range(n):
        masks[i] = indices[...,i].unsqueeze(-1) == torch.arange(shape[i], device=device)
        
    for indices in np.ndindex(shape):
        prod_mask = functools.reduce(lambda mask_1, mask_2: mask_1 & mask_2, [masks[i][...,idx] for i, idx in enumerate(indices)])
        binned_values[indices] = values[prod_mask]
        
    return grid, binned_values

def bin_values(y, values, N_bins=50, device=None, **bucketize_kwargs):
    assert y.shape == values.shape
    y = y.unsqueeze(-1)
    shape = (N_bins,)
    
    return bin_values_n(y, values, shape=shape, device=device, **bucketize_kwargs)

def binned_statistic_n(y, values=None, shape=(50,), statistics=None, device=None, **bucketize_kwargs):
    """
    Efficient pytorch implementation of scipy.stats.binned_statistic_dd using torch.scatter_add_
    Interestingly, given fixed number of values, it seems that the larger shape is, the faster this runs,
    suggesting it runs faster with less elements in each bin.
    """
    
    n = len(shape)
    N = np.prod(y.shape[:-1])
    if statistics is None:
        statistics = ['count','mean','std','stderr']
    stats = {}
        
    assert len(statistics) > 0
    if values is None or statistics == ['count']:
        assert (values is None) and (statistics == ['count'])
    else:
        assert y.shape[:-1] == values.shape
        values = values.reshape(-1)
    ones = torch.ones(N, dtype=torch.long, device=device)
    
    grid, indices = bucketize_n(y, shape=shape, device=device, **bucketize_kwargs)
    flat_indices = torch.arange(np.prod(shape), device=device).reshape(shape)[tuple(indices.movedim(-1,0))].reshape(-1) # (N,)
    
    count = torch.zeros(np.prod(shape), dtype=torch.long, device=device)
    count.scatter_add_(0, flat_indices, ones)
    
    stats['count'] = count.reshape(shape)
    statistics = list(filter(lambda s: s != 'count', statistics))
    
    if len(statistics) == 0:
        return grid, stats
    
    mean = torch.zeros(np.prod(shape), device=device)
    mean.scatter_add_(0, flat_indices, values)
    mean = mean / count
    
    stats['mean'] = mean.reshape(shape)
    statistics = list(filter(lambda s: s != 'mean', statistics))
    
    if len(statistics) == 0:
        return grid, stats
        
    std = torch.zeros(np.prod(shape), device=device)
    std.scatter_add_(0, flat_indices, values**2)
    std = (1/(count-1) * std - count/(count-1) * mean**2)**0.5 # bessel correction
    
    stats['std'] = std.reshape(shape)
    statistics = list(filter(lambda s: s != 'std', statistics))
    
    if len(statistics) == 0:
        return grid, stats
    
    stderr = std / count**0.5
    
    stats['stderr'] = stderr.reshape(shape)
    statistics = list(filter(lambda s: s != 'stderr', statistics))
    
    if len(statistics) == 0:
        return grid, stats
    
    raise NotImplementedError()
    
def binned_statistic(y, values=None, N_bins=50, statistics=None, device=None, **bucketize_kwargs):
    y = y.unsqueeze(-1)
    shape = (N_bins,)
    
    grid, stats = binned_statistic_n(y, values=values, shape=shape, statistics=statistics, device=device, **bucketize_kwargs)
    
    for k, v in stats.items():
        stats[k] = v.squeeze()
        
    return grid, stats