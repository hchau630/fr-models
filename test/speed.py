import timeit
import time
import pathlib

import numpy as np
import torch

from fr_models import analytic_models as amd
from fr_models import numerical_models as nmd
import hyclib as lib

def get_data(path):
    data = lib.io.load_data(path)
    
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

def setup():
    data_path = pathlib.Path('/home/hc3190/ken/V1-perturb/test/data/responses')
    filename = list(data_path.glob('*.pkl'))[0]
    data = lib.io.load_data(filename)
        
    model_data, exp_data = get_data(data['path'])
    W_pop, sigma_pop, amplitude, r_star, Ls, shape, w_dims = model_data
    x_data, y_data, new_y_data, x_scale = exp_data

    W = torch.tensor(W_pop, dtype=torch.float)
    sigma = torch.tensor(sigma_pop, dtype=torch.float)
    r_star = torch.tensor(r_star, dtype=torch.float)

    return W, sigma, amplitude, r_star, Ls, w_dims

def get_response(W, sigma, amplitude, r_star, Ls, shape, w_dims, T=500.0, device='cpu'):
    a_model = amd.GaussianSSNModel(W, sigma, w_dims=w_dims)
    n_model = a_model.numerical_model(Ls, shape, device=device).nonlinear_perturbed_model(r_star)
    n_model.to(device)

    delta_h = n_model.get_h(amplitude, 0, device=device)
    delta_r0 = torch.tensor(0.0, device=device)
    t = torch.tensor(T, device=device)

    with torch.no_grad():
        delta_r = n_model(delta_h, delta_r0, t, rtol=1e-5, atol=1e-7)
    delta_r = delta_r.detach().cpu().numpy()

def main():
    shape = (1501,)
    W, sigma, amplitude, r_star, Ls, w_dims = setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')

    result = timeit.repeat(
        'get_response(W, sigma, amplitude, r_star, Ls, shape, w_dims, T=500.0, device=device)',
        repeat=10,
        number=1,
        globals={'get_response': get_response, **locals()},
    )
    print(result)

if __name__ == '__main__':
    main()