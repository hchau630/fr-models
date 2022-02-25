import torch
import numpy as np

import utils
from fr_models import analytic_models as amd
from fr_models import gridtools

ENV = {
    "models_path": "/home/hc3190/ken/V1-perturb/data/models",
    "exp_path": "/home/hc3190/ken/V1-perturb/data/experiment",
}

def load_exp_data(filepath, x_cutoff=300.0, symmetric=False, normalize=None):
    data = np.loadtxt(filepath, delimiter=',')
    x_data, y_data, y_data_sem = data[0], data[1], data[2]
    length = np.sum(x_data < x_cutoff)
    if symmetric:
        length = length - 1 if length % 2 == 0 else length
    x_data, y_data, y_data_sem = x_data[:length], y_data[:length], y_data_sem[:length]
    if normalize is not None:
        scale = np.max(x_data)/normalize
        x_data = x_data/scale
        return x_data.reshape(-1,1), y_data, y_data_sem, scale
    return x_data.reshape(-1,1), y_data, y_data_sem

class TestMultiCellSSNModel:
    def test_W(self):
        data_path = '/home/hc3190/ken/fr-models/test/data'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_name in ['model_1', 'model_2']:
            model_config = utils.io.load_data(f'{data_path}/{model_name}/a_model_config.pkl')
            model_W = torch.tensor(utils.io.load_data(f'{data_path}/{model_name}/n_model_W.pkl'), dtype=torch.float)
            
            W = torch.tensor(model_config['W'], dtype=torch.float, device=device)
            sigma = torch.tensor(model_config['sigma'], dtype=torch.float, device=device)
            
            period = [L for i, L in enumerate(model_config['Ls']) if i in model_config['w_dims']]
            a_model = amd.GaussianSSNModel(W, sigma, w_dims=model_config['w_dims'], wn_order=11, period=period)
            
            grid = gridtools.Grid(model_config['Ls'], model_config['shape'], w_dims=model_config['w_dims'], device=device)
            n_model = a_model.numerical_model(grid)
            n_model_W = n_model.W.cpu()
            
            assert torch.allclose(model_W, n_model_W)
            
class TestLinearizedMultiCellSSNModel:
    def test_spectral_radius_circulant(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # W
        W = torch.tensor([
            [0.9283, -1.0534],
            [0.3060, -0.2324],
        ], device=device) # for debugging, see github branch debug_2

        # sigma
        sigma_s = torch.tensor(
            [[0.2336, 0.0679],
             [ 0.2584, 0.3015]],
            device=device,
        ) # for debugging, see github branch debug_2

        # r_star
        _, y_data_base, _ = load_exp_data(f'{ENV["exp_path"]}/baselines/base_by_dist.txt')
        r_star = torch.tensor([1.0, 1.2], dtype=torch.float) * torch.tensor(np.mean(y_data_base), dtype=torch.float)
        r_star = r_star.to(device)

        ndim_s = 1
        Ls = [2*1.0]*ndim_s # we want our model to be twice the data length scale
        shape = tuple([100]*ndim_s)
        w_dims = [0]
        period = Ls[0]

        a_model = amd.SpatialSSNModel(W, sigma_s, ndim_s, w_dims=w_dims, period=period)
        grid = gridtools.Grid(Ls, shape, w_dims=w_dims, device=device)
        
        a_model.to(device)
        
        n_model = a_model.numerical_model(grid)
        lp_model = n_model.linear_perturbed_model(r_star)
        
        spectral_radius_1 = lp_model.spectral_radius()
        spectral_radius_2 = lp_model.spectral_radius(use_circulant=True)
        
        print(spectral_radius_1, spectral_radius_2)
        torch.testing.assert_close(spectral_radius_1, spectral_radius_2)