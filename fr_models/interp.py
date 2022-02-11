import torch
import torch_interpolations

from . import _torch

class RegularGridInterpolator(torch_interpolations.RegularGridInterpolator):
    @classmethod
    def from_grid(cls, grid, values):
        points = []
        for i in range(grid.ndim-1):
            p = grid.slice(i)
            if i in grid.w_dims:
                p = torch.cat([p, p[:1]])
            points.append(p)
        values = _torch.pad(values, [(0,1) if i in grid.w_dims else (0,0) for i in range(values.ndim)], mode='wrap')
        return cls(points, values)
    
    def __call__(self, points_to_interp, bounds_error=True):
        # For some reason the guy who made this package decided
        # he would ignore the call signature of scipy's RegularGridInterpolator
        # and ignore usual conventions of putting the batch dimensions
        # in the preceding dimensions and decided to make the RegularGridInterpolator's
        # __call__ function accept points_to_interp as a list of 1D tensors
        # where len(points_to_interp) is the number of dimensions.
        # This slight wrapper accepts a normal input, namely that
        # points_to_interp is a tensor of shape (...,n), where n is the number of dimensions
        # and returns a results of shape (...)
        # Also adds bounds_error option
        batch_shape = points_to_interp.shape[:-1]
        n = points_to_interp.shape[-1]
        
        if bounds_error:
            bounds = [(p.min(), p.max()) for p in self.points]
            within_bounds = [(bounds[i][0] <= points_to_interp[...,i]) & (points_to_interp[...,i] <= bounds[i][1]) for i in range(n)]
            
            if not all([torch.all(within_bounds[i]).item() for i in range(n)]):
                description = '\n'.join([f"Dimension {i} - bad indices: {torch.nonzero(~within_bounds[i])}, bad values: {points_to_interp[~within_bounds[i]]}" for i in range(n)])
                raise ValueError(f"points_to_interp contain out of bounds values: \n {description}")
            
        else:
            raise NotImplementedError()
        
        points_to_interp = points_to_interp.reshape(-1, n).T.contiguous() # .contiguous() suppresses a warning
        points_to_interp = [points_to_interp[i] for i in range(n)]
        result = super().__call__(points_to_interp)
        result = result.reshape(batch_shape)
        return result