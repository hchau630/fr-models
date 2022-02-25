import pytest

import torch
import numpy as np

from fr_models import kernels, gridtools

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_scale():
    return torch.tensor([
        [0.5,-1.2],
        [1.5,-0.6],
    ])

def get_sigma(period):
    return torch.tensor([
        [[1.0, period], [0.5, period/2]],
        [[1.5, period*2], [1.5, period/2]]
    ])

def get_cov(period):
    return torch.tensor([
        [[[1.0, 0.0],
          [0.0, period]],
         [[0.5, 0.0],
          [0.0, period/2]]],
        [[[1.5, 0.0],
          [0.0, period*2]],
         [[1.5, 0.0],
          [0.0, period/2]]],
    ])**2

def get_points(period):
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

@pytest.mark.parametrize("w_dims", [([]), ([0]), ([1]), ([0,1])])
@pytest.mark.parametrize("order", [(1),(3)])
@pytest.mark.parametrize("period", [(2*np.pi), (np.pi), (1.0)])
def test_K_wg_1_point(w_dims, order, period, device):
    scale = get_scale().to(device)
    sigma = get_sigma(period).to(device)
    cov = get_cov(period).to(device)
    points = get_points(period).to(device)
    
    assert torch.allclose(torch.diag_embed(sigma**2), cov)
        
    W = kernels.K_wg(scale, cov, w_dims=w_dims, order=order, period=period)
    
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
        assert torch.allclose(W_point, expected)
        
@pytest.mark.parametrize("w_dims", [([]), ([0]), ([1]), ([0,1])])
@pytest.mark.parametrize("order", [(1),(3)])
@pytest.mark.parametrize("period", [(2*np.pi), (np.pi), (1.0)])
def test_K_wg_2_points(w_dims, order, period, device):
    scale = get_scale().to(device)
    sigma = get_sigma(period).to(device)
    cov = get_cov(period).to(device)
    points_x = get_points(period).to(device)
    points_y = get_points(period).to(device)
    outer_x, outer_y = gridtools.meshgrid([points_x,points_y])
    
    assert torch.allclose(torch.diag_embed(sigma**2), cov)
        
    W = kernels.K_wg(scale, cov, w_dims=w_dims, order=order, period=period)
    
    W_outer = W(outer_x, outer_y)
    
    assert W_outer.shape == (len(points_x), len(points_y), *scale.shape)
    
    for i, point_x in enumerate(points_x):
        for j, point_y in enumerate(points_y):
            expected = 1.0
            point = point_y - point_x
            for k in range(2):
                if k in w_dims:
                    point_k, sign = point[k].abs(), torch.sign(point[k])
                    point[k] = sign*torch.min(point_k, period-point_k)
                    
            for k in range(2):
                if k in w_dims:
                    if order == 1:
                        expected *= 1/((2*np.pi)**0.5*sigma[:,:,k]) * torch.exp(-point[k]**2/(2*sigma[:,:,k]**2))
                    elif order == 3:
                        expected *= 1/((2*np.pi)**0.5*sigma[:,:,k]) * (
                            torch.exp(-point[k]**2/(2*sigma[:,:,k]**2)) + 
                            torch.exp(-(point[k]+period)**2/(2*sigma[:,:,k]**2)) +
                            torch.exp(-(point[k]-period)**2/(2*sigma[:,:,k]**2))
                        )
                    else:
                        raise NotImplementedError()
                else:
                    expected *= 1/((2*np.pi)**0.5*sigma[:,:,k]) * torch.exp(-point[k]**2/(2*sigma[:,:,k]**2))
            expected *= scale
            assert torch.allclose(W_outer[i,j], expected)