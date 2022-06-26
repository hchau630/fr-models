import copy

import pytest
import torch

from fr_models import response_models as rmd

class TestRadialSteadyStateResponse:
    @pytest.mark.skip(reason='use_circulant functionality currently disabled.')
    @pytest.mark.parametrize('r_model_untrained',
                             [
                                 {'ndim_s': 1, 'periodic': True},
                                 # {'ndim_s': 2, 'periodic': True}
                             ],
                             indirect=True)
    def test_circulant(self, r_model_untrained, device, response_data, make_benchmark):
        r_model_untrained.check_interpolation_range = False
        r_model_untrained_circulant = copy.deepcopy(r_model_untrained)
        r_model_untrained_circulant.grid = r_model_untrained.grid.clone()
        r_model_untrained_circulant.n_model_kwargs = {'use_circulant': True}
        
        r_model_untrained.to(device)
        r_model_untrained_circulant.to(device)
        
        x_data, _, _ = response_data
        
        x_data = x_data.to(device)
        
        # resp_1 = make_benchmark()(r_model_untrained, x_data)
        resp_1 = r_model_untrained(x_data)
        # resp_2 = make_benchmark()(r_model_untrained_circulant, x_data)
        resp_2 = r_model_untrained_circulant(x_data)
        
        torch.testing.assert_close(resp_1, resp_2)
        
class TestLinearizedSteadyStateResponse:
    @pytest.mark.parametrize('r_model_untrained',
                             [
                                 {'ndim_s': 1, 'periodic': True},
                                 # {'ndim_s': 2, 'periodic': True}
                             ],
                             indirect=True)
    def test_untrained(self, r_model_untrained, device, response_data):
        r_model_untrained.linearized = True
        
        r_model_untrained_theory = copy.deepcopy(r_model_untrained)
        r_model_untrained_theory.grid = r_model_untrained.grid.clone()
        r_model_untrained_theory.method = 'theory'
        
        r_model_untrained.to(device)
        r_model_untrained_theory.to(device)
        
        x_data, _, _ = response_data
        
        x_data = x_data.to(device)
        
        print("Running non-theory...")
        resp_1 = r_model_untrained(x_data)
        print("Running theory...")
        resp_2 = r_model_untrained_theory(x_data)

        torch.testing.assert_close(resp_1, resp_2)
        
    @pytest.mark.parametrize('r_model_trained',
                             [
                                 {'idx': i} for i in range(10) if i != 7
                             ],
                             indirect=True)
    def test_trained(self, r_model_trained, device, response_data):
        r_model_trained.linearized = True
        
        r_model_trained_theory = copy.deepcopy(r_model_trained)
        r_model_trained_theory.grid = r_model_trained.grid.clone()
        r_model_trained_theory.method = 'theory'
        
        r_model_trained.to(device)
        r_model_trained_theory.to(device)
        
        x_data, _, _ = response_data
        
        x_data = x_data.to(device)
        
        print("Running non-theory...")
        resp_1 = r_model_trained(x_data)
        print("Running theory...")
        resp_2 = r_model_trained_theory(x_data)

        torch.testing.assert_close(resp_1, resp_2)