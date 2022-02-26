import copy

import pytest
import torch

from fr_models import response_models as rmd

class TestRadialSteadyStateResponse:
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