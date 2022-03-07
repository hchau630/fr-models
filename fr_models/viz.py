import matplotlib.pyplot as plt
import numpy as np
import torch

import utils
from fr_models import analytic_models as amd
from fr_models import geom

def kernel_1D(kern, dpcurve, fig=None, ax=None, labels=None):
    if (fig is None and ax is not None) or (fig is not None and ax is None):
        raise ValueError("User must either provide both fig and ax together or none of them.")
    
    x = dpcurve.t
    y = kern(dpcurve.points).reshape(len(x),-1).t()
    
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    
    if ax is None:
        fig, ax = utils.plot.subplots(1,1)
        
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
    
    return fig, ax, lines

def a_model_kernel_1D(a_model, dim, lims=None, unit='a.u.', fig=None, ax=None, labels=None, device='cpu'):
    assert isinstance(a_model, amd.GaussianSSNModel)
    
    if dim in a_model.w_dims:
        if isinstance(a_model.period, list) or isinstance(a_model.period, tuple):
            period = a_model.period[a_model.w_dims.index(dim)]
        else:
            period = a_model.period
        lims = [-period/2, period/2]
    else:
        three_sigma = a_model.sigma[...,dim].max().item()*3
        lims = [-three_sigma, three_sigma]
    
    t = torch.linspace(lims[0], lims[1], steps=50, device=device)
    w = torch.zeros(a_model.ndim)
    w[dim] = 1.0
    pline = geom.curve.PLine(w)
    pline.to(device)
    dpcurve = geom.curve.DPCurve(t, pline(t))
    
    fig, ax, lines = kernel_1D(a_model.kernel, dpcurve, fig=fig, ax=ax, labels=labels)
        
    if unit == 'a.u.':
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
        
    return fig, ax, lines

def r_model_kernel_1D(r_model, dim, lims=None, unit='microns', fig=None, ax=None, labels=None, device='cpu'):
    fig, ax, lines = a_model_kernel_1D(r_model.a_model, dim, lims=lims, unit='a.u.', fig=fig, ax=ax, labels=labels, device=device)
    
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
        
    return fig, ax, lines

def kernel_2D(kern, dplane):
    pass