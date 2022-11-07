import pytest
import torch
import numpy as np

from fr_models import _torch
import utils

def old_binned_statistic_n(y, values=None, mask=None, nanstats=True, statistics=None, device=None, **bucketize_kwargs):
    """
    Efficient pytorch implementation of scipy.stats.binned_statistic_dd using torch.scatter_add_
    Interestingly, given fixed number of values, it seems that the larger shape is, the faster this runs,
    suggesting it runs faster with less elements in each bin.
    
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
        else:
            statistics = ['count','mean','std','stderr']
    stats = {}
        
    assert len(statistics) > 0
    if values is None or statistics == ['count']:
        assert (values is None) and (statistics == ['count'])
    else:
        assert y.shape[:-1] == values.shape
        values = values[~mask] # (N,), where N is the number non-masked values
    
    grid, indices = _torch.stats.bucketize_n(y, device=device, **bucketize_kwargs)
    shape = grid.grid_shape
    s_ind = torch.arange(np.prod(shape), device=device).reshape(shape)
    s_ind = _torch.pad(s_ind, [(0,1) for _ in range(len(shape))], mode='const', const=np.nan) # np.nan for out-of-bounds indices
    flat_indices = s_ind[tuple(indices.movedim(-1,0))][~mask] # (N,)
    
    indices_nan_mask = torch.isnan(flat_indices)
    flat_indices = flat_indices[~indices_nan_mask].long()
    if values is not None:
        values = values[~indices_nan_mask]
        if nanstats:
            values_nan_mask = torch.isnan(values)
            flat_indices, values = flat_indices[~values_nan_mask], values[~values_nan_mask]
    
    ones = torch.ones(len(flat_indices), dtype=torch.long, device=device)
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

def test_bin_values():
    values = torch.normal(mean=0.0, std=1.0,size=(3,10,11))
    y = torch.normal(mean=0.0, std=1.0, size=(10,11,2))
    shape = (4,4)
    ymin = [-1,-1]
    ymax = [1,1]

    grid, binned_values_0 = _torch.stats.bin_values_n(y, values, bin_mode=0, ymin=ymin, ymax=ymax, shape=shape)
    for i in range(len(values)):
        grid, binned_values_1 = _torch.stats.bin_values_n(y, values[i], bin_mode=1, ymin=ymin, ymax=ymax, shape=shape)

        for ndindex in np.ndindex(shape):
            try:
                torch.testing.assert_close(binned_values_0[ndindex][:,i], binned_values_1[ndindex])
            except AssertionError as err:
                print(f'{ndindex}, {i}')
                print(binned_values_0[ndindex][:,i])
                print(binned_values_1[ndindex])
                raise err

    values = torch.normal(mean=0.0, std=1.0,size=(10,11))
    y = torch.normal(mean=0.0, std=1.0, size=(10,11,2))
    shape = (4,4)

    grid, binned_values_0 = _torch.stats.bin_values_n(y, values, bin_mode=0, shape=shape)
    grid, binned_values_1 = _torch.stats.bin_values_n(y, values, bin_mode=1, shape=shape)

    for ndindex in np.ndindex(shape):
        torch.testing.assert_close(binned_values_0[ndindex], binned_values_1[ndindex])

    values = torch.normal(mean=0.0, std=1.0,size=(2,15,))
    y = torch.normal(mean=0.0, std=1.0, size=(15,))
    N_bins = 5

    grid, binned_values_0 = _torch.stats.bin_values(y, values, bin_mode=0, N_bins=N_bins)

    for i in range(len(values)):
        grid, binned_values_1 = _torch.stats.bin_values(y, values[i], bin_mode=1, N_bins=N_bins)
        
        for j in range(N_bins):
            torch.testing.assert_close(binned_values_0[j][:,i], binned_values_1[j])
            
def test_binned_statistic():
    values = torch.normal(mean=0.0, std=1.0,size=(3,5,6))
    values[0,1,2] = np.nan
    values[2,3,1] = np.nan
    y = torch.normal(mean=0.0, std=1.0, size=(5,6,2))
    shape = (4,4)
    ymin = [-1,-1]
    ymax = [1,1]
    mask = torch.zeros(y.shape[:-1]).bool()
    mask[4,1] = False
    mask[2,3] = False

    grid, stats_0 = _torch.stats.binned_statistic_n(y, values, shape=shape, ymin=ymin, ymax=ymax, mask=mask)
    for i in range(len(values)):
        grid, stats_1 = old_binned_statistic_n(y, values[i], shape=shape, ymin=ymin, ymax=ymax, mask=mask)

        for k, v0, v1 in utils.itertools.dict_zip(stats_0, stats_1):
            try:
                torch.testing.assert_close(v0[i], v1, equal_nan=True)
            except AssertionError as err:
                print(f'{k}, {i}, nanstats=True')
                print(v0[i])
                print(v1)
                raise err
            
    values = torch.normal(mean=0.0, std=1.0,size=(3,5,6))
    values[0,1,2] = np.nan
    values[2,3,1] = np.nan
    y = torch.normal(mean=0.0, std=1.0, size=(5,6,2))
    shape = (4,4)
    ymin = [-1,-1]
    ymax = [1,1]
    mask = torch.zeros(y.shape[:-1]).bool()
    mask[4,1] = False
    mask[2,3] = False

    grid, stats_0 = _torch.stats.binned_statistic_n(y, values, shape=shape, ymin=ymin, ymax=ymax, mask=mask, nanstats=False)
    for i in range(len(values)):
        grid, stats_1 = old_binned_statistic_n(y, values[i], shape=shape, ymin=ymin, ymax=ymax, mask=mask, nanstats=False)

        for k, v0, v1 in utils.itertools.dict_zip(stats_0, stats_1):
            try:
                torch.testing.assert_close(v0[i], v1, equal_nan=True)
            except AssertionError as err:
                print(f'{k}, {i}, nanstats=False')
                print(v0[i])
                print(v1)
                raise err
                
    y = torch.normal(mean=0.0, std=1.0, size=(5,6,2))
    shape = (4,4)
    ymin = [-1,-1]
    ymax = [1,1]
    mask = torch.zeros(y.shape[:-1]).bool()
    mask[4,1] = False
    mask[2,3] = False

    grid, stats_0 = _torch.stats.binned_statistic_n(y, shape=shape, ymin=ymin, ymax=ymax, mask=mask)
    grid, stats_1 = old_binned_statistic_n(y, shape=shape, ymin=ymin, ymax=ymax, mask=mask)

    for k, v0, v1 in utils.itertools.dict_zip(stats_0, stats_1):
        try:
            torch.testing.assert_close(v0, v1, equal_nan=True)
        except AssertionError as err:
            print(f'{k}, {i}, nanstats=False')
            print(v0[i])
            print(v1)
            raise err
                
    y = torch.normal(mean=0.0, std=1.0, size=(5,6,2))
    shape = (4,4)
    ymin = [-1,-1]
    ymax = [1,1]
    mask = torch.zeros(y.shape[:-1]).bool()
    mask[4,1] = False
    mask[2,3] = False

    grid, stats_0 = _torch.stats.binned_statistic_n(y, shape=shape, ymin=ymin, ymax=ymax, mask=mask)
    grid, stats_1 = old_binned_statistic_n(y, shape=shape, ymin=ymin, ymax=ymax, mask=mask)

    for k, v0, v1 in utils.itertools.dict_zip(stats_0, stats_1):
        try:
            torch.testing.assert_close(v0, v1, equal_nan=True)
        except AssertionError as err:
            print(f'{k}, {i}, nanstats=False')
            print(v0[i])
            print(v1)
            raise err