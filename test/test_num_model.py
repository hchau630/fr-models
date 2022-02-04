import torch

import utils
from fr_models import analytic_models as amd

class TestMultiCellSSNModel:
    def test_W(self):
        data_path = '/home/hc3190/ken/fr-models/test/data'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_name in ['model_1', 'model_2']:
            model_config = utils.io.load_data(f'{data_path}/{model_name}/a_model_config.pkl')
            model_W = torch.tensor(utils.io.load_data(f'{data_path}/{model_name}/n_model_W.pkl'), dtype=torch.float)
            W = torch.tensor(model_config['W'], dtype=torch.float)
            sigma = torch.tensor(model_config['sigma'], dtype=torch.float)
            period = [L for i, L in enumerate(model_config['Ls']) if i in model_config['w_dims']]
            a_model = amd.GaussianSSNModel(W, sigma, w_dims=model_config['w_dims'], wn_order=11, period=period)
            n_model = a_model.numerical_model(model_config['Ls'], model_config['shape'], device=device)
            n_model_W = n_model.W.detach().cpu()
            assert torch.allclose(model_W, n_model_W)