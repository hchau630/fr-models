import numbers
import functools

import torch
import numpy as np

from fr_models import gridtools
from . import core

def bucketize(y, N_bins=50, mode='mid', ymin=None, ymax=None, device=None):
    if ymin is None:
        ymin = y.min().item()
    if ymax is None:
        dynamic_ymax = True
        ymax = y.max().item()
    else:
        dynamic_ymax = False
        
    if mode == 'edge':
        grid_edges = gridtools.Grid([(ymin, ymax)], shape=(N_bins+1,), device=device)
        indices = torch.bucketize(y, grid_edges.tensor.squeeze(), right=True) - 1
        if dynamic_ymax:
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
        mode = ['mid'] * grid.D # each grid point represents the midpoint of a bin
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
        mode = ['mid'] * n
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

def _bin_values_n_0(y, values, device=None, **bucketize_kwargs):
    assert all([values.shape[-i-1] == k for i, k in enumerate(y.shape[:-1][::-1])])
    
    n = y.shape[-1]
    batch_ndim = len(y.shape[:-1])
    y = y.reshape(-1,n)
    values_batch_shape = values.shape[:-batch_ndim]
    values = values.reshape(*values_batch_shape,-1).movedim(-1,0)
    
    grid, indices = bucketize_n(y, device=device, **bucketize_kwargs)
    shape = grid.grid_shape
    n = len(shape)
    binned_values = np.empty(shape, dtype=object)
    
    for ndindex in np.ndindex(shape):
        binned_values[ndindex] = list()
    
    for ndindex, vi in zip(indices, values):
        if all([idx >= 0 and idx < shape[i] for i, idx in enumerate(ndindex)]):
            binned_values[tuple(ndindex)].append(vi)
        
    for ndindex in np.ndindex(shape):
        if len(binned_values[ndindex]) >= 1:
            binned_values[ndindex] = torch.stack(binned_values[ndindex], dim=0)
        else:
            binned_values[ndindex] = torch.empty((0,*values_batch_shape), device=device)
        
    return grid, binned_values

def _bin_values_0(y, values, N_bins=None, device=None, **bucketize_kwargs):
    assert all([values.shape[-i-1] == k for i, k in enumerate(y.shape[::-1])])
    
    y = y.unsqueeze(-1)
    if N_bins is not None:
        assert 'shape' not in bucketize_kwargs, 'binned_statistic does not accept "shape" and "N_bins" arguments at the same time'
        bucketize_kwargs['shape'] = (N_bins,)
    
    return _bin_values_n_0(y, values, device=device, **bucketize_kwargs)

def _bin_values_n_1(y, values, device=None, **bucketize_kwargs):
    assert y.shape[:-1] == values.shape
    
    grid, indices = bucketize_n(y, device=device, **bucketize_kwargs)
    shape = grid.grid_shape
    n = len(shape)
    binned_values = np.empty(shape, dtype=object)
    masks = np.empty(n, dtype=object)
    
    for i in range(n):
        masks[i] = indices[...,i].unsqueeze(-1) == torch.arange(shape[i], device=device)
        
    for indices in np.ndindex(shape):
        prod_mask = functools.reduce(lambda mask_1, mask_2: mask_1 & mask_2, [masks[i][...,idx] for i, idx in enumerate(indices)])
        binned_values[indices] = values[prod_mask]
        
    return grid, binned_values

def _bin_values_1(y, values, N_bins=None, device=None, **bucketize_kwargs):
    assert y.shape == values.shape
    y = y.unsqueeze(-1)
    if N_bins is not None:
        assert 'shape' not in bucketize_kwargs, 'binned_statistic does not accept "shape" and "N_bins" arguments at the same time'
        bucketize_kwargs['shape'] = (N_bins,)
    
    return _bin_values_n_1(y, values, device=device, **bucketize_kwargs)

def bin_values_n(*args, bin_mode=0, **kwargs):
    if bin_mode == 0:
        return _bin_values_n_0(*args, **kwargs)
    if bin_mode == 1:
        return _bin_values_n_1(*args, **kwargs)
    raise NotImplementedError()
    
def bin_values(*args, bin_mode=0, **kwargs):
    if bin_mode == 0:
        return _bin_values_0(*args, **kwargs)
    if bin_mode == 1:
        return _bin_values_1(*args, **kwargs)
    raise NotImplementedError()
    
def binned_statistic_n(y, values=None, values_err=None, mask=None, nanstats=True, statistics=None, device=None, **bucketize_kwargs):
    """
    Efficient pytorch implementation of scipy.stats.binned_statistic_dd using torch.scatter_add_
    Interestingly, given fixed number of values, it seems that the larger shape is, the faster this runs,
    suggesting it runs faster with less elements in each bin.
    
    values - trailing dimensions of values must match leading dimensions of y. leading dimensions of values is treated as
    batch dimensions (called bshape in code). statistics will have shape (*bshape, *bin_shape)
    
    nanstats - If values is None, this has no effect. Otherwise, if true, all statistics (i.e. count, mean, std, stderr)
    will all be performed on non-nan values, and if false, count will include nan values, and mean, std, stderr will be 
    nan if the corresponding bin has a nan value.
    
    statistics - By default, if values is None, statistics will be ['count'], and if values is not None, statistics 
    will include count, mean, std, and stderr. Error will be raised if values is None but statistics contains any 
    one of mean, std, or stderr.
    """
    assert not torch.isnan(y).any() # does not handle the case where y has np.nan entries

    if mask is None:
        mask = torch.zeros(y.shape[:-1], dtype=torch.bool, device=device)
    
    if statistics is None:
        if values is None:
            statistics = ['count']
        elif values_err is None:
            statistics = ['count','mean','std','stderr']
        else:
            statistics = ['count', 'mean', 'std', 'stderr', 'meanerr']
    stats = {}
        
    assert len(statistics) > 0
    if values is None or statistics == ['count']:
        assert (values is None) and (statistics == ['count'])
        bshape = tuple()
    else:
        assert all([values.shape[-i-1] == k for i, k in enumerate(y.shape[:-1][::-1])])
        batch_ndim = len(y.shape[:-1])
        bshape = values.shape[:-batch_ndim]
        if values_err is not None:
            assert values_err.shape == values.shape, f'{values_err.shape=}, {values.shape=}'
        values = values[...,~mask] # (*,N,), where N is the number non-masked values
        if values_err is not None:
            values_err = values_err[...,~mask]  # (*,N,), where N is the number non-masked values
    
    grid, indices = bucketize_n(y, device=device, **bucketize_kwargs)
    shape = grid.grid_shape
    s_ind = torch.arange(np.prod(shape), device=device).reshape(shape)
    s_ind = core.pad(s_ind, [(0,1) for _ in range(len(shape))], mode='const', const=np.nan) # np.nan for out-of-bounds indices
    indices = s_ind[tuple(indices.movedim(-1,0))][~mask] # (N,)
    
    indices_nan_mask = torch.isnan(indices)
    indices = indices[~indices_nan_mask].long() # (M,)
    if values is not None:
        values = values[...,~indices_nan_mask] # (*,M)
        if values_err is not None:
            values_err = values_err[...,~indices_nan_mask] # (*,M)
        indices = indices.broadcast_to(values.shape) # (*,M)
        if nanstats:
            values_nan_mask = torch.isnan(values)
            if values_err is not None:
                values_nan_mask = values_nan_mask | torch.isnan(values_err)
            values[values_nan_mask] = 0
            if values_err is not None:
                values_err[values_nan_mask] = 0
    
    ones = torch.ones(indices.shape, dtype=torch.long, device=device)
    if nanstats and values is not None:
        ones[values_nan_mask] = 0
    count = torch.zeros((*bshape, np.prod(shape)), dtype=torch.long, device=device)
    count.scatter_add_(-1, indices, ones)
    
    stats['count'] = count.reshape((*bshape, *shape))
    statistics = list(filter(lambda s: s != 'count', statistics))
    
    if len(statistics) == 0:
        return grid, stats
    
    mean = torch.zeros((*bshape, np.prod(shape)), device=device)
    mean.scatter_add_(-1, indices, values)
    mean = mean / count
    
    stats['mean'] = mean.reshape((*bshape, *shape))
    statistics = list(filter(lambda s: s != 'mean', statistics))
    
    if len(statistics) == 0:
        return grid, stats
        
    std = torch.zeros((*bshape, np.prod(shape)), device=device)
    std.scatter_add_(-1, indices, values**2)
    std = (1/(count-1) * std - count/(count-1) * mean**2)**0.5 # bessel correction
    
    stats['std'] = std.reshape((*bshape, *shape))
    statistics = list(filter(lambda s: s != 'std', statistics))
    
    if len(statistics) == 0:
        return grid, stats
    
    stderr = std / count**0.5
    
    stats['stderr'] = stderr.reshape((*bshape, *shape))
    statistics = list(filter(lambda s: s != 'stderr', statistics))
    
    if len(statistics) == 0:
        return grid, stats
    
    meanerr = torch.zeros((*bshape, np.prod(shape)), device=device)
    meanerr.scatter_add_(-1, indices, values_err**2)
    meanerr = meanerr**0.5 / count
    
    stats['meanerr'] = meanerr.reshape((*bshape, *shape))
    statistics = list(filter(lambda s: s != 'meanerr', statistics))
    
    if len(statistics) == 0:
        return grid, stats
    
    raise NotImplementedError()

def binned_statistic(y, values=None, values_err=None, N_bins=None, statistics=None, device=None, **bucketize_kwargs):
    y = y.unsqueeze(-1)
    if N_bins is not None:
        assert 'shape' not in bucketize_kwargs, 'binned_statistic does not accept "shape" and "N_bins" arguments at the same time'
        bucketize_kwargs['shape'] = (N_bins,)
    
    grid, stats = binned_statistic_n(y, values=values, values_err=values_err, statistics=statistics, device=device, **bucketize_kwargs)
    
    for k, v in stats.items():
        stats[k] = v.squeeze()
        
    return grid, stats