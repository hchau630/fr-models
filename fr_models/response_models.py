import torch
import logging

from . import interp
from . import exceptions
from . import analytic_models as amd

logger = logging.getLogger(__name__)

class SteadyStateResponse(torch.nn.Module):
    def __init__(self, a_model, grid, r_star, amplitude, i, j, length_scales, method='dynamic', dr_rtol=1.0e-3, dr_atol=1.0e-5, max_t=500.0, solver_kwargs=None, n_model_kwargs=None, check_interpolation_range=True):
        super().__init__()
        self.a_model = a_model
        self.register_buffer('grid', grid, persistent=False) # persistent=False means don't store in state_dict
        self.r_star = r_star
        self.amplitude = amplitude
        self.i = i # output cell type number
        self.j = j # input cell type number
        # Note: length_scale = what 1.0 in model means in the units of the data
        length_scales = torch.as_tensor(length_scales, dtype=torch.float)
        self.register_buffer('length_scales', length_scales, persistent=False) # persistent=False means don't store in state_dict
        self.method = method
        self.dr_rtol = dr_rtol
        self.dr_atol = dr_atol
        self.max_t = max_t
        self.solver_kwargs = {} if solver_kwargs is None else solver_kwargs
        self.n_model_kwargs = {} if n_model_kwargs is None else n_model_kwargs
        self.check_interpolation_range = check_interpolation_range
        # Let's hope that pytorch will have a nn.Buffer() feature one day so we can get rid of register_buffer...
        # There is an open issue on this: https://github.com/pytorch/pytorch/issues/35735, but no progress so far...
        
    def forward(self, x):
        """
        x - shape (*batch_shape, a_model.ndim)
        """
        # preprocess x to scale it according to length scale
        x = x / self.length_scales

        dxs = torch.tensor(self.grid.dxs, dtype=torch.float, device=x.device)
        if self.check_interpolation_range and torch.any((torch.abs(x) < dxs) & (x != 0)):
            raise RuntimeError("x must not be within interpolation range near 0. Try increasing number of neurons.")
                
        n_model = self.a_model.numerical_model(self.grid, **self.n_model_kwargs)
        nlp_model = n_model.nonlinear_perturbed_model(self.r_star, **self.n_model_kwargs)
        
        delta_h = n_model.get_h(self.amplitude, self.j)
        delta_r0 = torch.tensor(0.0, device=delta_h.device)
        
        if self.method == 'dynamic':
            t0 = torch.tensor(0.0, device=delta_h.device)
            nlp_delta_r, nlp_t = nlp_model.steady_state(delta_h, delta_r0, t0, dr_rtol=self.dr_rtol, dr_atol=self.dr_atol, max_t=self.max_t, **self.solver_kwargs)
        elif self.method == 'fixed':
            t = torch.tensor(self.max_t, device=delta_h.device)
            nlp_delta_r, nlp_t = nlp_model(delta_h, delta_r0, t, **self.solver_kwargs)
            raise NotImplementedError() # Need to change PerturbedNonlinearModel _drdt so that we can check whether it is at steady state
        else:
            raise NotImplementedError()
        
        delta_ri_curve = interp.RegularGridInterpolator.from_grid(self.grid, nlp_delta_r[self.i])

        return delta_ri_curve(x)
    
class RadialSteadyStateResponse(SteadyStateResponse):
    def __init__(self, a_model, *args, **kwargs):
        assert isinstance(a_model, amd.SpatialSSNModel) # TODO: define a SpatialModel abstract base class and we can check that inheritance of that class instead
        super().__init__(a_model, *args, **kwargs)
    
    def forward(self, x):
        """
        x - shape (*batch_shape, ndim_f+1), where the leading dim along the last axis is the radial spatial dimension
        """
        if self.a_model.ndim_s > 1:
            x_space, x_feature = x[...,:1], x[...,1:]
            x_pad = torch.zeros((*x.shape[:-1],self.a_model.ndim_s-1), device=x.device)
            x = torch.cat([x_space, x_pad, x_feature],dim=-1)
        return super().forward(x)