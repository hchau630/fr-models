import pytest

import torch

import utils
from fr_models import analytic_models as amd
from fr_models import gridtools

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
    @pytest.mark.parametrize('r_model_untrained', [{'ndim_s': 1, 'periodic': True}], indirect=True)
    def test_spectral_radius_circulant(self, r_model_untrained, device):
        r_model_untrained.to(device)
        
        a_model = r_model_untrained.a_model
        grid = r_model_untrained.grid
        r_star = r_model_untrained.r_star
        
        n_model = a_model.numerical_model(grid)
        lp_model = n_model.linear_perturbed_model(r_star)
        
        spectral_radius_1 = lp_model.spectral_radius()
        spectral_radius_2 = lp_model.spectral_radius(use_circulant=True)
        
        print(spectral_radius_1, spectral_radius_2)
        torch.testing.assert_close(spectral_radius_1, spectral_radius_2)
        