import torch

from . import interp
from . import exceptions

class SteadyStateResponse(torch.nn.Module):
    def __init__(self, a_model, grid, r_star, amplitude, i, j, method='dynamic', dr_rtol=1.0e-3, dr_atol=1.0e-5, max_t=500.0, solver_kwargs=None):
        super().__init__()
        self.a_model = a_model
        self.register_buffer('grid', grid)
        self.r_star = r_star
        self.amplitude = amplitude
        self.register_buffer('i', i) # output cell type number
        self.register_buffer('j', j) # input cell type number
        self.method = method
        self.dr_rtol = dr_rtol
        self.dr_atol = dr_atol
        self.max_t = max_t
        self.solver_kwargs = {} if solver_kwargs is None else solver_kwargs
        # Let's hope that pytorch will have a nn.Buffer() feature one day so we can get rid of register_buffer...
        # There is an open issue on this: https://github.com/pytorch/pytorch/issues/35735, but no progress so far...
        
    def forward(self, x):
        dxs = torch.tensor(self.grid.dxs, dtype=torch.float, device=x.device)
        if torch.any((torch.abs(x) < dxs) & (x != 0)):
            raise RuntimeError("x must not be within interpolation range near 0. Try increasing number of neurons.")
                
        n_model = self.a_model.numerical_model(self.grid)
        nlp_model = n_model.nonlinear_perturbed_model(self.r_star)
        
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
    