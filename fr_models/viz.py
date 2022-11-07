import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import torch

import utils
from fr_models import analytic_models as amd
from fr_models import geom

def scaterr(x, y, yerr, ax=None, **kwargs):
    default_kwargs = {
        'marker': '.',
        'ls': 'none',
    }
    default_kwargs.update(kwargs)
    
    if ax is None:
        ax = plt.gca()
    
    return ax.errorbar(x.tolist(), y.tolist(), yerr=yerr.tolist(), **default_kwargs)

def field_1D(field, grid, dim, length_scales=None, half=True, shift=0, ax=None, label=None):
    assert field.shape == grid.grid_shape
    
    if ax is None:
        ax = plt.gca()
        
    indices = list(grid.mids)
    if half:
        indices[dim] = slice(indices[dim] + shift, None)
    else:
        indices[dim] = slice(shift, None)
    indices = tuple(indices)
    
    if length_scales is not None:
        x = grid.tensor[indices][...,dim] * length_scales[dim]
    else:
        x = grid.tensor[indices][...,dim]
    
    line, = ax.plot(x.tolist(), field[indices].tolist(), label=label)
    
    return line

def field_2D(field, grid=None, mask=None, mode='imshow', cscheme='diverging', indexing='xy', ax=None, **kwargs):
    if grid is not None:
        assert grid.shape[-1] == 2 and grid.ndim == 3
        
    assert indexing in ['xy', 'ij']
    
    if mode == 'contourf' and indexing != 'xy':
        raise ValueError("when mode == 'contourf', indexing must be 'xy'.")
    
    if ax is None:
        ax = plt.gca()
        
    if mask is not None:
        field[mask] = np.nan
        
    if cscheme == 'diverging':
        cmap = 'bwr'
        norm = colors.CenteredNorm()
    elif cscheme == 'cyclic':
        cmap = 'hsv'
        norm = None
    else:
        raise NotImplementedError()
        
    if mode == 'imshow':
        assert grid.shape[:-1] == field.shape
        extent = None
        if grid is not None:
            dx = grid[1,0,0].item() - grid[0,0,0].item()
            dy = grid[0,1,1].item() - grid[0,0,1].item()
            extent = [
                grid[0,0,0].item() - dx/2,
                grid[-1,0,0].item() + dx/2,
                grid[0,0,1].item() - dy/2,
                grid[0,-1,1].item() + dy/2,
            ]
        if indexing == 'xy':
            origin = 'lower'
            field = field.t()
        else:
            origin = 'upper'
            
        im = ax.imshow(field.cpu().numpy(), cmap=cmap, interpolation='none', norm=norm, extent=extent, origin=origin, **kwargs)
        
    elif mode == 'contourf':
        assert grid is not None
        im = ax.contourf(grid[...,0].cpu().numpy(), grid[...,1].cpu().numpy(), field.cpu().numpy(), cmap=cmap, **kwargs)
    
    else:
        raise NotImplementedError()
        
    ax.set_aspect('equal')
    
    return im

def kernel_1D(kern, dpcurve, ax=None, labels=None):
    x = dpcurve.t
    y = kern(dpcurve.points).reshape(len(x),-1).t()
    
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    
    if ax is None:
        ax = plt.gca()
        
    if labels is not None:
        labels = np.array(labels)
        
    lines = []
    for i, y_i in enumerate(y):
        if labels is not None:
            line, = ax.plot(x, y[i], label=labels[np.unravel_index(i, kern.F_shape)])
        else:
            line, = ax.plot(x, y[i], label=np.unravel_index(i, kern.F_shape))
        lines.append(line)
            
    ax.legend()
    
    return lines

def a_model_kernel_1D(a_model, dim, lims=None, steps=50, unit=None, length_scale=1.0, ax=None, labels=None, device='cpu'):
    assert isinstance(a_model, amd.SSNModel)
    
    if lims is None:
        if dim in a_model.w_dims:
            if isinstance(a_model.period, list) or isinstance(a_model.period, tuple):
                period = a_model.period[a_model.w_dims.index(dim)]
            else:
                period = a_model.period
            lims = [-period/2, period/2]
        elif hasattr(a_model, 'sigma'):
            three_sigma = a_model.sigma[...,dim].max().item()*3
            lims = [-three_sigma, three_sigma]
        else:
            raise RuntimeError("lims must be provided if dim does not correspond to a circular dimension and a_model does not have sigma")
    else:
        assert len(lims) == 2
    
    t = torch.linspace(lims[0], lims[1], steps=steps, device=device)
    w = torch.zeros(a_model.ndim)
    w[dim] = 1.0
    pline = geom.curve.PLine(w)
    pline.to(device)
    dpcurve = geom.curve.DPCurve(t, pline(t))
    
    lines = kernel_1D(a_model.kernel, dpcurve, ax=ax, labels=labels)
    
    if ax is None:
        ax = plt.gca()
        
    if unit is None:
        pass
    elif unit == 'a.u.':
        length_scale = 1.0
        ax.set_xlabel('$\Delta$ distance (a.u.)')
    elif unit == 'degrees':
        length_scale = 180 / np.pi
        ax.set_xlabel('$\Delta$ angle (degrees)')
    else:
        raise NotImplementedError(f"{unit=} has not been implemented")
        
    for line in lines:
        xdata = line.get_xdata()
        line.set_xdata(xdata*length_scale)
    ax.relim()
        
    return lines

def r_model_kernel_1D(r_model, dim, lims=None, steps=50, unit='microns', ax=None, labels=None, device='cpu'):
    if lims == 'boundary':
        lims = r_model.grid.extents[dim]
    
    lines = a_model_kernel_1D(r_model.a_model, dim, lims=lims, steps=steps, unit='a.u.', fig=fig, ax=ax, labels=labels, device=device)
    
    if ax is None:
        ax = plt.gca()
    
    length_scales = r_model.length_scales.detach().cpu().numpy()
    if len(length_scales) == 1:
        length_scale = length_scales
    else:
        length_scale = length_scales[dim]
        
    if unit == 'microns':
        ax.set_xlabel('$\Delta$ distance ($\mu$m)')
    elif unit == 'degrees':
        length_scale = length_scale * 180 / np.pi
        ax.set_xlabel('$\Delta$ angle (degrees)')
    else:
        raise NotImplementedError(f"{unit=} has not been implemented")
        
    for line in lines:
        xdata = line.get_xdata()
        line.set_xdata(xdata*length_scale)
    ax.relim()
        
    return lines

def kernel_2D(kern, dplane):
    pass