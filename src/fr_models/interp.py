from itertools import product

import torch

from . import _torch

class BaseRegularGridInterpolator:

    def __init__(self, points, values):
        self.points = points
        self.values = values

        assert isinstance(self.points, tuple) or isinstance(self.points, list)
        assert isinstance(self.values, torch.Tensor)

        self.ms = list(self.values.shape)
        self.n = len(self.points)

        assert len(self.ms) == self.n

        for i, p in enumerate(self.points):
            assert isinstance(p, torch.Tensor)
            assert p.shape[0] == self.values.shape[i]

    def __call__(self, points_to_interp):
        assert self.points is not None
        assert self.values is not None

        assert len(points_to_interp) == len(self.points)
        K = points_to_interp[0].shape[0]
        for x in points_to_interp:
            assert x.shape[0] == K

        idxs = []
        dists = []
        overalls = []
        for p, x in zip(self.points, points_to_interp):
            idx_right = torch.bucketize(x, p)
            idx_right[idx_right >= p.shape[0]] = p.shape[0] - 1
            idx_left = (idx_right - 1).clamp(0, p.shape[0] - 1)
            dist_left = x - p[idx_left]
            dist_right = p[idx_right] - x
            dist_left[dist_left < 0] = 0.
            dist_right[dist_right < 0] = 0.
            both_zero = (dist_left == 0) & (dist_right == 0)
            dist_left[both_zero] = dist_right[both_zero] = 1.

            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)

        numerator = 0.
        for indexer in product([0, 1], repeat=self.n):
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
            numerator += self.values[as_s] * torch.prod(torch.stack(bs_s), dim=0)
        denominator = torch.prod(torch.stack(overalls), dim=0)
        return numerator / denominator

class RegularGridInterpolator(BaseRegularGridInterpolator):
    @classmethod
    def from_grid(cls, grid, values):
        assert grid.grid_shape == values.shape
        
        points = []
        for i in range(grid.ndim-1):
            p = grid.slice(i)
            if i in grid.w_dims:
                p = torch.cat([p, -p[:1]])
            points.append(p)
        values = _torch.pad(values, [(0,1) if i in grid.w_dims else (0,0) for i in range(values.ndim)], mode='wrap')
        return cls(points, values)
    
    def __call__(self, points_to_interp, exclude=None, bounds_error=True):
        # For some reason the guy who made this package decided
        # he would ignore the call signature of scipy's RegularGridInterpolator
        # and ignore usual conventions of putting the batch dimensions
        # in the preceding dimensions and decided to make the RegularGridInterpolator's
        # __call__ function accept points_to_interp as a list of 1D tensors
        # where len(points_to_interp) is the number of dimensions.
        # This slight wrapper accepts a normal input, namely that
        # points_to_interp is a tensor of shape (...,n), where n is the number of dimensions
        # and returns a results of shape (...)
        # Also adds exclude and bounds_error option
        assert points_to_interp.ndim > 1
        batch_shape = points_to_interp.shape[:-1]
        assert points_to_interp.shape[-1] == self.n, f'{points_to_interp.shape=}, but {self.n=}'
        
        if exclude is not None:
            for p_idx in exclude:
                assert len(p_idx) == self.n
                lb = torch.as_tensor([self.points[i][max(p_idx[i]-1,0)] for i in range(self.n)], device=points_to_interp.device)
                ub = torch.as_tensor([self.points[i][min(p_idx[i]+1,len(self.points[i])-1)] for i in range(self.n)], device=points_to_interp.device)
                if torch.any(mask := ((points_to_interp > lb) & (points_to_interp < ub))):
                    raise ValueError(f"Point(s) {points_to_interp[mask]} is (are) within the excluded box at {p_idx}: {lb.tolist(), ub.tolist()}")
        
        if bounds_error:
            bounds = [(p.min(), p.max()) for p in self.points]
            within_bounds = [(bounds[i][0] <= points_to_interp[...,i]) & (points_to_interp[...,i] <= bounds[i][1]) for i in range(self.n)]
            
            if not all([torch.all(within_bounds[i]).item() for i in range(self.n)]):
                description = '\n'.join([f"Dimension {i} - bad indices: {torch.nonzero(~within_bounds[i])}, bad values: {points_to_interp[~within_bounds[i]]}" for i in range(self.n)])
                raise ValueError(f"points_to_interp contain out of bounds values:\n" \
                                 f"bounds: {bounds}\n" \
                                 f"{description}")
            
        else:
            raise NotImplementedError()
        
        points_to_interp = points_to_interp.reshape(-1, self.n).T.contiguous() # .contiguous() suppresses a warning
        points_to_interp = [points_to_interp[i] for i in range(self.n)]
        result = super().__call__(points_to_interp)
        result = result.reshape(batch_shape)
        return result
    
def interp(grid, values):
    return RegularGridInterpolator.from_grid(grid, values)
