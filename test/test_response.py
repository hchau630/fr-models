import pathlib

import numpy as np
import torch

from fr_models import analytic_models as amd
from fr_models import numerical_models as nmd
from fr_models import response_models as rmd
from fr_models import gridtools
import utils

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

def get_data(path):
    data = utils.io.load_data(path)
    
    popt = data['popt']
    fit_options = data['meta']['fit_options']
    w_dims = data['meta']['w_dims']
    Ls = fit_options['Ls']
    shape = fit_options['shape']
    if 'r_star' in fit_options:
        r_star = fit_options['r_star']
    else:
        r_star = np.copy(popt['r_star'])
    
    x_data = data['x_data']
    y_data = data['y_data']
    new_y_data = data['new_y_data']
    x_scale = data['x_scale']
    
    W_pop, sigma_pop, amplitude = np.copy(popt['W_pop']), np.copy(popt['sigma_pop']), np.copy(popt['amplitude'])
    
    return (W_pop, sigma_pop, amplitude, r_star, Ls, shape, w_dims), (x_data, y_data, new_y_data, x_scale)

def test_response():
    plot_dim = 0
    i, j = 0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_path = pathlib.Path('/home/hc3190/ken/V1-perturb/test/data/responses')
    for filename in data_path.glob('*.pkl'):
        print(filename)
        data = utils.io.load_data(filename)
        
        model_data, exp_data = get_data(data['path'])
        W_pop, sigma_pop, amplitude, r_star, Ls, shape, w_dims = model_data
        x_data, y_data, new_y_data, x_scale = exp_data
        
        W = torch.tensor(W_pop, dtype=torch.float, device=device)
        sigma = torch.tensor(sigma_pop, dtype=torch.float, device=device)
        r_star = torch.tensor(r_star, dtype=torch.float, device=device)
        amplitude = torch.tensor(amplitude, dtype=torch.float, device=device)
        
        grid = gridtools.Grid(Ls, shape, w_dims=w_dims, device=device)
        a_model = amd.GaussianSSNModel(W, sigma, w_dims=w_dims)
        n_model = a_model.numerical_model(grid).nonlinear_perturbed_model(r_star)
        n_model.to(device)

        delta_h = n_model.get_h(amplitude, torch.tensor(j, device=device))
        delta_r0 = torch.tensor(0.0, device=device)
        # t = torch.tensor(500.0, device=device)
        # delta_r, _ = n_model(delta_h, delta_r0, t)
        t0 = torch.tensor(0.0, device=device)
        delta_r, t = n_model.steady_state(delta_h, delta_r0, t0, method='dopri5')
        print(t)
        delta_r = delta_r.detach().cpu().numpy()

        E_x, E_y = get_xy(delta_r[0], plot_dim, Ls, w_dims=w_dims, x_scale=x_scale, offset=0)
        I_x, I_y = get_xy(delta_r[1], plot_dim, Ls, w_dims=w_dims, x_scale=x_scale, offset=0)
        
        E_rel_err = np.abs(E_y-data['data']['E_y'])/data['data']['E_y']
        I_rel_err = np.abs(I_y-data['data']['I_y'])/data['data']['I_y']
        print(np.max(E_rel_err), E_y[np.argmax(E_rel_err)], data['data']['E_y'][np.argmax(E_rel_err)])
        print(np.max(I_rel_err), I_y[np.argmax(I_rel_err)], data['data']['I_y'][np.argmax(I_rel_err)])
        assert np.allclose(E_x, data['data']['E_x'], rtol=1.0e-3, atol=1.0e-4)
        assert np.allclose(E_y, data['data']['E_y'], rtol=1.0e-3, atol=1.0e-4)
        assert np.allclose(I_x, data['data']['I_x'], rtol=1.0e-3, atol=1.0e-4)
        assert np.allclose(I_y, data['data']['I_y'], rtol=1.0e-3, atol=1.0e-4)

# def test_interpolated_response():
#     plot_dim = 0
#     i, j = 0, 0
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     data_path = pathlib.Path('/home/hc3190/ken/V1-perturb/test/data/responses')
#     for filename in data_path.glob('*.pkl'):
#         print(filename)
#         data = utils.io.load_data(filename)
        
#         model_data, exp_data = get_data(data['path'])
#         W_pop, sigma_pop, amplitude, r_star, Ls, shape, w_dims = model_data
#         x_data, y_data, new_y_data, x_scale = exp_data
        
#         W = torch.tensor(W_pop, dtype=torch.float, device=device)
#         sigma = torch.tensor(sigma_pop, dtype=torch.float, device=device)
#         a_model = amd.GaussianSSNModel(W, sigma, w_dims=w_dims)
#         grid = gridtools.Grid(Ls, shape, w_dims=w_dims, device=device)
#         r_star = torch.tensor(r_star, dtype=torch.float, device=device)
#         amplitude = torch.tensor(amplitude, dtype=torch.float, device=device)

#         torch_x_data = torch.tensor(x_data, dtype=torch.float, device=device)
        
#         print(y_data)
        
#         # solver_kwargs = {'method': 'dopri5', 'rtol': 1.0e-3, 'atol': 1.0e-5, 'max_t': 10000.0}
#         solver_kwargs = None
#         model = rmd.SteadyStateResponse(a_model, grid, r_star, amplitude, torch.tensor(0), torch.tensor(0), solver_kwargs=solver_kwargs)
#         model.to(device)
        
#         E_y = model(torch_x_data).cpu().numpy()
        
#         model = rmd.SteadyStateResponse(a_model, grid, r_star, amplitude, torch.tensor(1), torch.tensor(0), solver_kwargs=solver_kwargs)
#         model.to(device)
        
#         I_y = model(torch_x_data).cpu().numpy()

# #         W = torch.tensor(W_pop, dtype=torch.float)
# #         sigma = torch.tensor(sigma_pop, dtype=torch.float)
# #         r_star = torch.tensor(r_star, dtype=torch.float)

# #         a_model = amd.GaussianSSNModel(W, sigma, w_dims=w_dims)
# #         n_model = a_model.numerical_model(Ls, shape).nonlinear_perturbed_model(r_star)
# #         n_model.to(device)

# #         delta_h = n_model.get_h(amplitude, j, device=device)
# #         delta_r0 = torch.tensor(0.0, device=device)
# #         # t = torch.tensor(500.0, device=device)
# #         # delta_r, _ = n_model(delta_h, delta_r0, t)
# #         t0 = torch.tensor(0.0, device=device)
# #         delta_r, t = n_model.steady_state(delta_h, delta_r0, t0)
# #         print(t)
# #         delta_r = delta_r.detach().cpu().numpy()

#         # E_x, E_y = get_xy(delta_r[0], plot_dim, Ls, w_dims=w_dims, x_scale=x_scale, offset=0)
#         # I_x, I_y = get_xy(delta_r[1], plot_dim, Ls, w_dims=w_dims, x_scale=x_scale, offset=0)
        
#         E_rel_err = np.abs(E_y-data['data']['E_y'])/data['data']['E_y']
#         I_rel_err = np.abs(I_y-data['data']['I_y'])/data['data']['I_y']
#         print(np.max(E_rel_err), E_y[np.argmax(E_rel_err)], data['data']['E_y'][np.argmax(E_rel_err)])
#         print(np.max(I_rel_err), I_y[np.argmax(I_rel_err)], data['data']['I_y'][np.argmax(I_rel_err)])
#         # assert np.allclose(E_x, data['data']['E_x'], rtol=1.0e-3, atol=1.0e-4)
#         assert np.allclose(E_y, data['data']['E_y'], rtol=1.0e-3, atol=1.0e-4)
#         # assert np.allclose(I_x, data['data']['I_x'], rtol=1.0e-3, atol=1.0e-4)
#         assert np.allclose(I_y, data['data']['I_y'], rtol=1.0e-3, atol=1.0e-4)