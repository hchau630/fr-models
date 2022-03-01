import pytest

import numpy as np
import torch

from fr_models import gridtools

def get_xy(acts, plot_dim, Ls, w_dims=[], marg_dims=[], half=True, offset=1, x_scale=1.0):
    shape = acts.shape
    D = len(shape)
    
    mids = gridtools.get_mids(shape, w_dims=w_dims)
    if isinstance(offset, int):
        offset = [offset if d == plot_dim else 0 for d in range(D)]
    locs = [mids[i] + offset[i] for i in range(len(mids))]
    endpoint = False if plot_dim in w_dims else True
    
    start = offset[plot_dim]
    if half:
        start += mids[plot_dim]
    
    x = np.linspace(-Ls[plot_dim]/2,Ls[plot_dim]/2,shape[plot_dim],endpoint=endpoint)
    indices = tuple(locs[:plot_dim]+[slice(None)]+locs[plot_dim+1:])
    y = acts[indices]
    
    if plot_dim in w_dims:
        x = np.append(x, -x[:1])
        y = np.append(y, y[:1])
        
    return x[start:]*x_scale, y[start:]

@pytest.mark.parametrize('method', ['dynamic', 'static'])
@pytest.mark.parametrize('r_model_trained, trained_responses', [
    ({'idx': 0}, {'idx': 0}) if i != 7 else pytest.param({'idx': 7}, {'idx': 7}, marks=pytest.mark.xfail(reason='inaccurate ground truth')) for i in range(10)], indirect=True)
def test_response(method, r_model_trained, trained_responses, device, benchmark):
    plot_dim = 0
    i, j = 0, 0
    
    r_model_trained.to(device)
    
    a_model = r_model_trained.a_model
    grid = r_model_trained.grid
    r_star = r_model_trained.r_star
    amplitude = r_model_trained.amplitude
    
    n_model = a_model.numerical_model(grid).nonlinear_perturbed_model(r_star)

    delta_h = n_model.get_h(amplitude, j)
    delta_r0 = torch.tensor(0.0, device=delta_h.device)
    max_t = torch.tensor(500.0, device=delta_h.device)
    
    delta_r, t = n_model.steady_state(delta_h, delta_r0, max_t, method=method, dr_rtol=1.0e-3, dr_atol=1.0e-5, solver_kwargs=None)
    # delta_r, t = benchmark(n_model.steady_state, delta_h, delta_r0, max_t, method=method)
    delta_r = delta_r.cpu().numpy()

    x_scale = r_model_trained.length_scales.cpu().numpy()
    E_x, E_y = get_xy(delta_r[0], plot_dim, grid.Ls, w_dims=grid.w_dims, x_scale=x_scale, offset=0)
    I_x, I_y = get_xy(delta_r[1], plot_dim, grid.Ls, w_dims=grid.w_dims, x_scale=x_scale, offset=0)

    E_rel_err = np.abs(E_y-trained_responses['E_y'])/trained_responses['E_y']
    I_rel_err = np.abs(I_y-trained_responses['I_y'])/trained_responses['I_y']
    # print(np.max(E_rel_err), E_y[np.argmax(E_rel_err)], trained_responses['E_y'][np.argmax(E_rel_err)])
    # print(np.max(I_rel_err), I_y[np.argmax(I_rel_err)], trained_responses['I_y'][np.argmax(I_rel_err)])
    assert np.allclose(E_x, trained_responses['E_x'], rtol=1.0e-3, atol=1.0e-5)
    assert np.allclose(E_y, trained_responses['E_y'], rtol=1.0e-3, atol=1.0e-5)
    assert np.allclose(I_x, trained_responses['I_x'], rtol=1.0e-3, atol=1.0e-5)
    assert np.allclose(I_y, trained_responses['I_y'], rtol=1.0e-3, atol=1.0e-5)
