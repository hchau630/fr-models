import pytest
import torch
import numpy as np

from fr_models import analytic_models as amd
from fr_models import gridtools, _torch

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_eigvalsbc(device):
    period = 1.0
    L = [1.0]
    shape = (10,)
    w_dims = [0]

    W = torch.tensor([
        [0.5,-1.2],
        [1.5,-0.6],
    ], device=device)
    sigma = torch.tensor([
        [period/3, period/2],
        [period/4, period/2]
    ], device=device)
    sigma = sigma[...,None]

    grid = gridtools.Grid(L, shape=shape, w_dims=w_dims, device=device)
    a_model = amd.GaussianSSNModel(W, sigma, period=period, w_dims=w_dims)
    n_model = a_model.numerical_model(grid)

    true_tensor = torch.tensor([
        [True,True],
        [True,True]
    ], device=device)
    
    assert (_torch.linalg.is_circulant(n_model.W_expanded.moveaxis(2,1)) == true_tensor).all()
    assert (_torch.linalg.is_hermitian(n_model.W_expanded.moveaxis(2,1)) == true_tensor).all()
    assert (_torch.linalg.is_symmetric(n_model.W_expanded.moveaxis(2,1)) == true_tensor).all()
    
    eigvals_1 = torch.linalg.eigvals(n_model.W)
    eigvals_2 = _torch.linalg.eigvalsbc(n_model.W_expanded)
    eigvals_2h = _torch.linalg.eigvalsbc(n_model.W_expanded, hermitian=True)

    eigvals_1 = np.sort_complex(eigvals_1.cpu().numpy())
    eigvals_2 = np.sort_complex(eigvals_2.cpu().numpy())
    eigvals_2h = np.sort_complex(eigvals_2h.cpu().numpy())

    # print(*zip(eigvals_1, eigvals_2, eigvals_2h), sep='\n')
    
    torch.testing.assert_close(eigvals_1, eigvals_2h)
    torch.testing.assert_close(eigvals_2, eigvals_2h)
    
def test_eigvalsbnc(device):
    period = 1.0
    L = [1.0,1.0]
    shape = (10,12)
    w_dims = [0,1]

    W = torch.tensor([
        [0.52,-1.25],
        [1.51,-0.64],
    ], device=device)
    sigma = torch.tensor([
        [[period/3.4,period/5.2], [period/2.9,period/4.5]],
        [[period/4.2,period/3.1], [period/2.5,period/3.5]]
    ], device=device)

    grid = gridtools.Grid(L, shape=shape, w_dims=w_dims, device=device)
    a_model = amd.GaussianSSNModel(W, sigma, period=period, w_dims=w_dims)
    n_model = a_model.numerical_model(grid)

    eigvals_1 = torch.linalg.eigvals(n_model.W)
    eigvals_2 = _torch.linalg.eigvalsbnc(n_model.W_expanded, 2)
    eigvals_2h = _torch.linalg.eigvalsbnc(n_model.W_expanded, 2, hermitian=True)

    eigvals_1 = np.sort_complex(eigvals_1.cpu().numpy())
    eigvals_2 = np.sort_complex(eigvals_2.cpu().numpy())
    eigvals_2h = np.sort_complex(eigvals_2h.cpu().numpy())
    
    # print(*zip(eigvals_1, eigvals_2, eigvals_2h), sep='\n')

    torch.testing.assert_close(eigvals_1, eigvals_2h)
    torch.testing.assert_close(eigvals_2, eigvals_2h)