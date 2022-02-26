import pytest

import torch
from scipy.interpolate import RegularGridInterpolator
import numpy as np

from fr_models import interp

def test_regular_grid_interpolator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    points = [torch.linspace(-.5, 2.5, steps=30, device=device, dtype=torch.double), torch.linspace(-.5, 2.5, steps=15, device=device, dtype=torch.double)]
    values = torch.sin(points[0])[:, None] + 2 * torch.cos(points[1])[None, :] + torch.sin(5 * points[0][:, None] @ points[1][None, :])
    X, Y = torch.meshgrid(torch.linspace(-.5, 2.5, steps=60, device=device, dtype=torch.double), torch.linspace(-.5, 2.5, steps=30, device=device, dtype=torch.double), indexing='xy')
    points_to_interp = torch.stack([X,Y],dim=-1)

    gi = interp.RegularGridInterpolator(points, values)
    rgi = RegularGridInterpolator([p.cpu().numpy() for p in points], values.cpu().numpy())

    np.testing.assert_allclose(
        gi(points_to_interp).cpu().numpy(), rgi(points_to_interp.cpu().numpy()))
    
def test_regular_grid_interpolator_oob():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    points = [torch.linspace(-.5, 2.5, steps=30, device=device, dtype=torch.double), torch.linspace(-.5, 2.5, steps=15, device=device, dtype=torch.double)]
    values = torch.sin(points[0])[:, None] + 2 * torch.cos(points[1])[None, :] + torch.sin(5 * points[0][:, None] @ points[1][None, :])
    X, Y = torch.meshgrid(torch.linspace(-.55, 2.55, steps=60, device=device, dtype=torch.double), torch.linspace(-.55, 2.55, steps=30, device=device, dtype=torch.double), indexing='xy')
    points_to_interp = torch.stack([X,Y],dim=-1)

    gi = interp.RegularGridInterpolator(points, values)
    rgi = RegularGridInterpolator([p.cpu().numpy() for p in points], values.cpu().numpy())

    with pytest.raises(ValueError) as e_info:
        gi(points_to_interp).cpu().numpy()
    print(e_info.value.args[0])
        
    with pytest.raises(ValueError) as e_info:
        rgi(points_to_interp.cpu().numpy())
    print(e_info.value.args[0])

def test_regular_grid_interpolator_derivative():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    points = [torch.linspace(-.5, 2.5, steps=30, device=device, dtype=torch.double), torch.linspace(-.5, 2.5, steps=15, device=device, dtype=torch.double)]
    values = torch.sin(points[0])[:, None] + 2 * torch.cos(points[1])[None, :] + torch.sin(5 * points[0][:, None] @ points[1][None, :])
    values.requires_grad_(True)
    X, Y = torch.meshgrid(torch.linspace(-.5, 2.5, steps=60, device=device, dtype=torch.double), torch.linspace(-.5, 2.5, steps=30, device=device, dtype=torch.double), indexing='xy')
    points_to_interp = torch.stack([X,Y],dim=-1)

    def f(values):
        return interp.RegularGridInterpolator(points, values)(points_to_interp)

    torch.autograd.gradcheck(f, (values,))