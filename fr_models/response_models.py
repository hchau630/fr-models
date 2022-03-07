import torch
import logging

from . import interp
from . import exceptions
from . import analytic_models as amd
from . import gridtools

logger = logging.getLogger(__name__)

class SteadyStateResponse(torch.nn.Module):
    def __init__(self, a_model, grid, r_star, amplitude, i, j, length_scales, max_t=500.0, avg_dims=None, method='dynamic', steady_state_kwargs=None, n_model_kwargs=None, check_interpolation_range=True):
        super().__init__()
        self.a_model = a_model # should be torch.nn.Module
        self.r_star = r_star # should be torch.nn.Parameter or optim.Parameter
        self.amplitude = amplitude # should be torch.nn.Parameter or optim.Parameter
        
        # register following objects as non-persisent buffer so that model.to(device) will move all of them to device
        # but state_dict() will not contain them, which is desirable since they are not parameters that should change
        self.register_buffer('grid', grid, persistent=False)
        self.register_buffer('i', torch.as_tensor(i, dtype=torch.long), persistent=False) # output cell type number
        self.register_buffer('j', torch.as_tensor(i, dtype=torch.long), persistent=False) # input cell type number
        # Note: length_scale = what 1.0 in model means in the units of the data
        self.register_buffer('length_scales', torch.as_tensor(length_scales, dtype=torch.float), persistent=False)
        self.register_buffer('max_t', torch.as_tensor(max_t, dtype=torch.float), persistent=False)
        
        self.avg_dims = avg_dims
        if self.avg_dims is not None:
            self.data_dims = [dim for dim in range(self.grid.D) if dim not in self.avg_dims]
        else:
            self.data_dims = list(range(self.grid.D))
        self.method = method
         
        self.steady_state_kwargs = {} if steady_state_kwargs is None else steady_state_kwargs
        self.n_model_kwargs = {} if n_model_kwargs is None else n_model_kwargs
        self.check_interpolation_range = check_interpolation_range # should always be True unless you're just testing stuff
        
    def forward(self, x):
        """
        x - shape (*batch_shape, len(self.data_dims))
        """      
        n_model = self.a_model.numerical_model(self.grid, **self.n_model_kwargs)
        nlp_model = n_model.nonlinear_perturbed_model(self.r_star, **self.n_model_kwargs)
        
        delta_h = n_model.get_h(self.amplitude, self.j)
        delta_r0 = torch.tensor(0.0, device=delta_h.device)
        
        nlp_delta_r, nlp_t = nlp_model.steady_state(delta_h, delta_r0, self.max_t, method=self.method, **self.steady_state_kwargs)
        
        if self.avg_dims is not None:
            nlp_delta_r = nlp_delta_r.mean(dim=[1+dim for dim in self.avg_dims]) # add 1 because we don't average over the cell types dimension
            
            grid = gridtools.Grid([self.grid.extents[dim] for dim in self.data_dims], shape=tuple([self.grid.grid_shape[dim] for dim in self.data_dims]), w_dims=[dim for dim in self.grid.w_dims if dim in self.data_dims], device=self.grid.tensor.device)
        else:
            grid = self.grid
        
        # preprocess x to scale it according to length scale
        x = x / self.length_scales[self.data_dims]

        dxs = torch.as_tensor(grid.dxs, dtype=torch.float, device=x.device)
        if self.check_interpolation_range and torch.any((torch.abs(x) < dxs) & (x != 0)):
            raise RuntimeError("x must not be within interpolation range near 0. Try increasing number of neurons.")
          
        delta_ri_curve = interp.RegularGridInterpolator.from_grid(grid, nlp_delta_r[self.i])

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