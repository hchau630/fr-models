import torch
import logging

from . import interp
from . import exceptions
from . import analytic_models as amd
from . import gridtools

logger = logging.getLogger(__name__)

class NModelSteadyStateResponse(torch.nn.Module):
    def __init__(self, n_model, grid, r_star, amplitude, i, j, length_scales, max_t=500.0, avg_dims=None, method='dynamic', steady_state_kwargs=None, n_model_kwargs=None, check_interpolation_range=True):
        super().__init__()
        self.n_model = n_model # should be torch.nn.Module
        self.r_star = r_star # should be torch.nn.Parameter or optim.Parameter
        self.amplitude = amplitude # should be torch.nn.Parameter or optim.Parameter
        
        # register following objects as non-persisent buffer so that model.to(device) will move all of them to device
        # but state_dict() will not contain them, which is desirable since they are not parameters that should change
        self.register_buffer('grid', grid, persistent=False)
        self.register_buffer('i', torch.as_tensor(i, dtype=torch.long), persistent=False) # output cell type number
        self.register_buffer('j', torch.as_tensor(j, dtype=torch.long), persistent=False) # input cell type number
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
        
    def get_delta_h(self, stim_loc=None):
        """
        Returns an input vector h where the neurons with index stim_loc in population j
        are given input with input strength amplitude.
        - stim_loc: Either a scalar integer, 1-dimensional array-like, or 
                   2-dimensional array-like object. Array-like means either tensor, 
                   tuple, or list. If 2-dimensional, then stim_loc.shape[0] is the number 
                   of neurons with non-zero input. If a scalar, then model must be
                   1-dimensional. If None, then stim_loc will be the index of the
                   neuron at the origin.
        """
        device = self.amplitude.device
        
        if stim_loc is None:
            stim_loc = self.grid.mids
        
        stim_loc = torch.as_tensor(stim_loc, device=device)
        stim_loc = torch.atleast_2d(stim_loc)
            
        h_j = torch.zeros(self.grid.grid_shape, device=device)        
        h_j[(*stim_loc.T,)] = self.amplitude
        h = torch.zeros(self.n_model.shape, device=device)
        h[self.j] = h_j
        
        return h
        
    def is_valid(self, x, stim_loc=None, assert_valid=False):
        # preprocess x to scale it according to length scale, so that x is now in model units
        x = x / self.length_scales[self.data_dims] # (*batch_shape, grid.D)
        
        delta_h = self.get_delta_h(stim_loc=stim_loc)
        
        if self.avg_dims is not None:
            grid = gridtools.Grid([self.grid.extents[dim] for dim in self.data_dims], shape=tuple([self.grid.grid_shape[dim] for dim in self.data_dims]), w_dims=[dim for dim in self.grid.w_dims if dim in self.data_dims], device=self.grid.tensor.device)
        else:
            grid = self.grid
            
        stim_coords = grid.tensor[delta_h[self.j] != 0.0] # coordinates of stimuli (N_stim, grid.D)
        delta_x = x.unsqueeze(-2) - stim_coords # (*batch_shape, N_stim, grid.D)

        dxs = torch.as_tensor(grid.dxs, dtype=torch.float, device=x.device)
        close_to_stim = torch.all(torch.abs(delta_x) < dxs, dim=-1) # (*batch_shape, N_stim)
        if assert_valid and torch.any(close_to_stim):
            raise RuntimeError(f"x must not be within interpolation range near any stimulus.\n" \
                               f"dxs: {dxs}.\n" \
                               f"Problem x indices:\n{torch.nonzero(close_to_stim)[:,:-1]}.\n" \
                               f"Problem stim indices:\n{torch.nonzero(delta_h[self.j])[torch.nonzero(close_to_stim)[:,-1]]}.\n" \
                               f"Problem x coords:\n{x[torch.nonzero(close_to_stim, as_tuple=True)[:-1]]}.\n" \
                               f"Problem stim coords:\n{stim_coords[torch.nonzero(close_to_stim, as_tuple=True)[-1]]}.")
        else:
            return ~torch.any(close_to_stim, dim=-1)
        
        
    def forward(self, x=None, stim_loc=None):
        """
        x - shape (*batch_shape, len(self.data_dims))
        """      
        nlp_model = self.n_model.nonlinear_perturbed_model(self.r_star, **self.n_model_kwargs)
        
        delta_h = self.n_model.get_h(self.amplitude, self.j, B_idx=stim_loc)
        torch.testing.assert_close(delta_h, self.get_delta_h(stim_loc=stim_loc))
        delta_r0 = torch.tensor(0.0, device=delta_h.device)
        
        nlp_delta_r, nlp_t = nlp_model.steady_state(delta_h, delta_r0, self.max_t, method=self.method, **self.steady_state_kwargs)
        
        if x is None:
            return nlp_delta_r
        
        if self.avg_dims is not None:
            nlp_delta_r = nlp_delta_r.mean(dim=[1+dim for dim in self.avg_dims]) # add 1 because we don't average over the cell types dimension
            
            grid = gridtools.Grid([self.grid.extents[dim] for dim in self.data_dims], shape=tuple([self.grid.grid_shape[dim] for dim in self.data_dims]), w_dims=[dim for dim in self.grid.w_dims if dim in self.data_dims], device=self.grid.tensor.device)
        else:
            grid = self.grid
        
        if self.check_interpolation_range:
            self.is_valid(x, stim_loc=stim_loc, assert_valid=True)
          
        delta_ri_curve = interp.RegularGridInterpolator.from_grid(grid, nlp_delta_r[self.i])
        
        # preprocess x to scale it according to length scale, so that x is now in model units
        x = x / self.length_scales[self.data_dims] # (*batch_shape, grid.D)

        return delta_ri_curve(x)
    
class SteadyStateResponse(torch.nn.Module):
    def __init__(self, a_model, grid, r_star, amplitude, i, j, length_scales, max_t=500.0, linearized=False, avg_dims=None, method='dynamic', steady_state_kwargs=None, n_model_kwargs=None, check_interpolation_range=True):
        super().__init__()
        self.a_model = a_model # should be torch.nn.Module
        self.r_star = r_star # should be torch.nn.Parameter or optim.Parameter
        self.amplitude = amplitude # should be torch.nn.Parameter or optim.Parameter
        
        # register following objects as non-persisent buffer so that model.to(device) will move all of them to device
        # but state_dict() will not contain them, which is desirable since they are not parameters that should change
        self.register_buffer('grid', grid, persistent=False)
        self.register_buffer('i', torch.as_tensor(i, dtype=torch.long), persistent=False) # output cell type number
        self.register_buffer('j', torch.as_tensor(j, dtype=torch.long), persistent=False) # input cell type number
        # Note: length_scale = what 1.0 in model means in the units of the data
        self.register_buffer('length_scales', torch.as_tensor(length_scales, dtype=torch.float), persistent=False)
        self.register_buffer('max_t', torch.as_tensor(max_t, dtype=torch.float), persistent=False)
        
        self.linearized = linearized
        self.avg_dims = avg_dims
        if self.avg_dims is not None:
            self.data_dims = [dim for dim in range(self.grid.D) if dim not in self.avg_dims]
        else:
            self.data_dims = list(range(self.grid.D))
        self.method = method
         
        self.steady_state_kwargs = {} if steady_state_kwargs is None else steady_state_kwargs
        self.n_model_kwargs = {} if n_model_kwargs is None else n_model_kwargs
        self.check_interpolation_range = check_interpolation_range # should always be True unless you're just testing stuff
        
    def get_delta_h(self, stim_loc=None):
        """
        Returns an input vector h where the neurons with index stim_loc in population j
        are given input with input strength amplitude.
        - stim_loc: Either a scalar integer, 1-dimensional array-like, or 
                   2-dimensional array-like object. Array-like means either tensor, 
                   tuple, or list. If 2-dimensional, then stim_loc.shape[0] is the number 
                   of neurons with non-zero input. If a scalar, then model must be
                   1-dimensional. If None, then stim_loc will be the index of the
                   neuron at the origin.
        """
        device = self.amplitude.device
        
        if stim_loc is None:
            stim_loc = self.grid.mids
        
        stim_loc = torch.as_tensor(stim_loc, device=device)
        stim_loc = torch.atleast_2d(stim_loc)
            
        h_j = torch.zeros(self.grid.grid_shape, device=device)        
        h_j[(*stim_loc.T,)] = self.amplitude
        h = torch.zeros((self.a_model.n, *self.grid.grid_shape), device=device)
        h[self.j] = h_j
        
        return h
        
    def is_valid(self, x, stim_loc=None, assert_valid=False):
        # preprocess x to scale it according to length scale, so that x is now in model units
        x = x / self.length_scales[self.data_dims] # (*batch_shape, grid.D)
        
        delta_h = self.get_delta_h(stim_loc=stim_loc)
        
        if self.avg_dims is not None:
            grid = gridtools.Grid([self.grid.extents[dim] for dim in self.data_dims], shape=tuple([self.grid.grid_shape[dim] for dim in self.data_dims]), w_dims=[dim for dim in self.grid.w_dims if dim in self.data_dims], device=self.grid.tensor.device)
        else:
            grid = self.grid
            
        stim_coords = grid.tensor[delta_h[self.j] != 0.0] # coordinates of stimuli (N_stim, grid.D)
        delta_x = x.unsqueeze(-2) - stim_coords # (*batch_shape, N_stim, grid.D)

        dxs = torch.as_tensor(grid.dxs, dtype=torch.float, device=x.device)
        close_to_stim = torch.all(torch.abs(delta_x) < dxs, dim=-1) # (*batch_shape, N_stim)
        if assert_valid and torch.any(close_to_stim):
            raise RuntimeError(f"x must not be within interpolation range near any stimulus.\n" \
                               f"dxs: {dxs}.\n" \
                               f"Problem x indices:\n{torch.nonzero(close_to_stim)[:,:-1]}.\n" \
                               f"Problem stim indices:\n{torch.nonzero(delta_h[self.j])[torch.nonzero(close_to_stim)[:,-1]]}.\n" \
                               f"Problem x coords:\n{x[torch.nonzero(close_to_stim, as_tuple=True)[:-1]]}.\n" \
                               f"Problem stim coords:\n{stim_coords[torch.nonzero(close_to_stim, as_tuple=True)[-1]]}.")
        else:
            return ~torch.any(close_to_stim, dim=-1)
        
    def forward(self, x=None, stim_loc=None):
        """
        x - shape (*batch_shape, len(self.data_dims))
        """      
        n_model = self.a_model.numerical_model(self.grid, **self.n_model_kwargs)
        if self.linearized:
            p_model = n_model.linear_perturbed_model(self.r_star, **self.n_model_kwargs)
        else:
            p_model = n_model.nonlinear_perturbed_model(self.r_star, **self.n_model_kwargs)
        
        delta_h = n_model.get_h(self.amplitude, self.j, B_idx=stim_loc)
        torch.testing.assert_close(delta_h, self.get_delta_h(stim_loc=stim_loc))
        delta_r0 = torch.tensor(0.0, device=delta_h.device)
        
        p_delta_r, _ = p_model.steady_state(delta_h, delta_r0, self.max_t, method=self.method, **self.steady_state_kwargs)
        
        if x is None:
            return p_delta_r
        
        if self.avg_dims is not None:
            p_delta_r = p_delta_r.mean(dim=[1+dim for dim in self.avg_dims]) # add 1 because we don't average over the cell types dimension
            
            grid = gridtools.Grid([self.grid.extents[dim] for dim in self.data_dims], shape=tuple([self.grid.grid_shape[dim] for dim in self.data_dims]), w_dims=[dim for dim in self.grid.w_dims if dim in self.data_dims], device=self.grid.tensor.device)
        else:
            grid = self.grid
        
        if self.check_interpolation_range:
            self.is_valid(x, stim_loc=stim_loc, assert_valid=True)
          
        delta_ri_curve = interp.RegularGridInterpolator.from_grid(grid, p_delta_r[self.i])
        
        # preprocess x to scale it according to length scale, so that x is now in model units
        x = x / self.length_scales[self.data_dims] # (*batch_shape, grid.D)

        return delta_ri_curve(x)
    
class RadialSteadyStateResponse(SteadyStateResponse):
    def __init__(self, a_model, *args, **kwargs):
        assert isinstance(a_model, amd.SpatialSSNModel) # TODO: define a SpatialModel abstract base class and we can check that inheritance of that class instead
        super().__init__(a_model, *args, **kwargs)
    
    def forward(self, x, **kwargs):
        """
        x - shape (*batch_shape, ndim_f+1), where the leading dim along the last axis is the radial spatial dimension
        """
        if self.a_model.ndim_s > 1:
            x_space, x_feature = x[...,:1], x[...,1:]
            x_pad = torch.zeros((*x.shape[:-1],self.a_model.ndim_s-1), device=x.device)
            x = torch.cat([x_space, x_pad, x_feature],dim=-1)
        return super().forward(x, **kwargs)