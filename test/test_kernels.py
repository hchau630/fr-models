import pytest

import torch
import numpy as np

from fr_models import kernels, gridtools
from fr_models import analytic_models as amd

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_scale():
    return torch.tensor([
        [0.5,-1.2],
        [1.5,-0.6],
    ])

def get_sigma():
    return torch.tensor([
        [[1.0, 1.0], [0.5, 0.75]],
        [[1.5, 2.0], [1.5, 0.5]]
    ])

def get_cov():
    return torch.tensor([
        [[[1.0, 0.0],
          [0.0, 1.0]],
         [[0.5, 0.0],
          [0.0, 0.75]]],
        [[[1.5, 0.0],
          [0.0, 2.0]],
         [[1.5, 0.0],
          [0.0, 0.5]]],
    ])**2

def get_points(period):
    if isinstance(period, list):
        return torch.tensor([
            [0.0,0.0],
            [period[0]/2,0.0],
            [0.0,period[1]/2],
            [period[0]/2,period[1]/2],
            [-period[0]/2,period[1]/2],
            [period[0]/2,-period[1]/2],
            [-period[0]/2,-period[1]/2],
            [period[0]/4, period[1]/8],
            [-period[0]/3, period[1]/6],
        ])
    return torch.tensor([
        [0.0,0.0],
        [period/2,0.0],
        [0.0,period/2],
        [period/2,period/2],
        [-period/2,period/2],
        [period/2,-period/2],
        [-period/2,-period/2],
        [period/4, period/8],
        [-period/3, period/6],
    ])

def test_discretize(device):
    period = 1.0
    L = [1.0,1.0]
    shape = (9,12)
    w_dims = [1]

    W = torch.tensor([
        [0.5,-1.2],
        [1.5,-0.6],
    ], device=device)
    sigma = torch.tensor([
        [[period/3,period/5], [period/2,period/4.5]],
        [[period/4,period/3.1], [period/2.5,period/3.5]]
    ], device=device)

    grid = gridtools.Grid(L, shape=shape, w_dims=w_dims, device=device)
    a_model = amd.GaussianSSNModel(W, sigma, period=period, w_dims=w_dims)
    torch.testing.assert_close(a_model.kernel.discretize(grid, use_symmetry=True), a_model.kernel.discretize(grid, use_symmetry=False))

@pytest.mark.parametrize("w_dims", [([]), ([0]), ([1]), ([0,1])])
@pytest.mark.parametrize("order", [(1),(3)])
@pytest.mark.parametrize("period", [(2*np.pi), (np.pi), (1.0)])
@pytest.mark.parametrize("mode", [('parallel'),('sequential')])
def test_K_wg_1_point(w_dims, order, period, mode, device):
    scale = get_scale().to(device)
    sigma = get_sigma().to(device)
    cov = get_cov().to(device)
    points = get_points(period).to(device)
    
    assert torch.allclose(torch.diag_embed(sigma**2), cov)
    
    if isinstance(period, list):
        period = [period[dim] for dim in w_dims]
        
    # W = kernels.K_wg(scale, cov, w_dims=w_dims, order=order, period=period)
    W = kernels.WrappedKernel(
        kernels.K_g(scale, cov),
        w_dims=w_dims,
        order=order, 
        period=period,
        mode=mode,
    )
    
    W_points = W(points)
    
    assert W_points.shape == (len(points), *scale.shape)
    
    for W_point, point in zip(W_points, points):
        expected = 1.0
        for i in range(2):
            if i in w_dims:
                if order == 1:
                    expected *= 1/((2*np.pi)**0.5*sigma[:,:,i]) * torch.exp(-point[i]**2/(2*sigma[:,:,i]**2))
                elif order == 3:
                    expected *= 1/((2*np.pi)**0.5*sigma[:,:,i]) * (
                        torch.exp(-point[i]**2/(2*sigma[:,:,i]**2)) + 
                        torch.exp(-(point[i]+period)**2/(2*sigma[:,:,i]**2)) +
                        torch.exp(-(point[i]-period)**2/(2*sigma[:,:,i]**2))
                    )
                else:
                    raise NotImplementedError()
            else:
                expected *= 1/((2*np.pi)**0.5*sigma[:,:,i]) * torch.exp(-point[i]**2/(2*sigma[:,:,i]**2))
        expected *= scale
        torch.testing.assert_close(W_point, expected)
        
@pytest.mark.parametrize("w_dims", [([]), ([0]), ([1]), ([0,1])])
@pytest.mark.parametrize("order", [(1),(3)])
@pytest.mark.parametrize("period", [(2*np.pi), (np.pi), (1.0), ([1.0,2*np.pi])])
@pytest.mark.parametrize("mode", [('parallel'),('sequential')])
def test_K_wg_2_points(w_dims, order, period, mode, device):
    torch.set_printoptions(precision=20)
    scale = get_scale().to(device)
    sigma = get_sigma().to(device)
    cov = get_cov().to(device)
    points_x = get_points(period).to(device)
    points_y = get_points(period).to(device)
    outer_x, outer_y = gridtools.meshgrid([points_x,points_y])
    
    assert torch.allclose(torch.diag_embed(sigma**2), cov)
    
    if isinstance(period, list):
        period = [period[dim] for dim in w_dims]
        
    # W = kernels.K_wg(scale, cov, w_dims=w_dims, order=order, period=period)
    W = kernels.WrappedKernel(
        kernels.K_g(scale, cov),
        w_dims=w_dims,
        order=order, 
        period=period,
        mode=mode,
    )
    
    W_outer = W(outer_x, outer_y)
    
    assert W_outer.shape == (len(points_x), len(points_y), *scale.shape)
    
    for i, point_x in enumerate(points_x):
        for j, point_y in enumerate(points_y):
            expected = 1.0
            point = point_y - point_x
            print('before: ', point)
            for k in range(2):
                if k in w_dims:
                    p = period[w_dims.index(k)] if isinstance(period, list) else period
                    point_k = point[k] + p/2
                    point_k = point_k % p
                    point_k = point_k - p/2
                    point[k] = point_k
                    # point_k, sign = point[k].abs(), torch.sign(point[k])
                    # point[k] = sign*torch.min(point_k, p-point_k)
            # print('after: ', point)
                    
            for k in range(2):
                if k in w_dims:
                    p = period[w_dims.index(k)] if isinstance(period, list) else period
                    
                    if order == 1:
                        expected *= 1/((2*np.pi)**0.5*sigma[:,:,k]) * torch.exp(-point[k]**2/(2*sigma[:,:,k]**2))
                    elif order == 3:
                        expected *= 1/((2*np.pi)**0.5*sigma[:,:,k]) * (
                            torch.exp(-point[k]**2/(2*sigma[:,:,k]**2)) + 
                            torch.exp(-(point[k]+p)**2/(2*sigma[:,:,k]**2)) +
                            torch.exp(-(point[k]-p)**2/(2*sigma[:,:,k]**2))
                        )
                    else:
                        raise NotImplementedError()
                else:
                    expected *= 1/((2*np.pi)**0.5*sigma[:,:,k]) * torch.exp(-point[k]**2/(2*sigma[:,:,k]**2))
            expected *= scale
            torch.testing.assert_close(W_outer[i,j], expected)