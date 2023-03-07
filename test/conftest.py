import pathlib

import torch
import numpy as np
import pytest
import pytest_benchmark.plugin

from fr_models import analytic_models as amd
from fr_models import response_models as rmd
from fr_models import gridtools
import utils

@pytest.fixture
def data_path(request):
    return f'{request.config.rootdir}/test/data'

def pytest_sessionstart(session):
    """
    Disable using TF32 cores on Ampere devices (e.g. A40), since that reduces precision so much that
    a lot of the tests will fail. pytest_sessionstart is a method that runs before tests are run
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def load_exp_data(filepath, x_cutoff=300.0, symmetric=False, normalize=None):
    data = np.loadtxt(filepath, delimiter=',')
    x_data, y_data, y_data_sem = data[0], data[1], data[2]
    length = np.sum(x_data < x_cutoff)
    if symmetric:
        length = length - 1 if length % 2 == 0 else length
    x_data, y_data, y_data_sem = x_data[:length], y_data[:length], y_data_sem[:length]
    if normalize is not None:
        scale = np.max(x_data)/normalize
        x_data = x_data/scale
        return x_data.reshape(-1,1), y_data, y_data_sem, scale
    return x_data.reshape(-1,1), y_data, y_data_sem

def get_data(path):
    data = utils.io.load_data(path)
    
    popt = data['popt']
    fit_options = data['meta']['fit_options']
    w_dims = data['meta']['w_dims']
    Ls = fit_options['Ls']
    shape = fit_options['shape']
    if 'r_star' in fit_options:
        r_star = fit_options['r_star']
    else:
        r_star = np.copy(popt['r_star'])
    
    x_data = data['x_data']
    y_data = data['y_data']
    new_y_data = data['new_y_data']
    x_scale = data['x_scale']
    
    W_pop, sigma_pop, amplitude = np.copy(popt['W_pop']), np.copy(popt['sigma_pop']), np.copy(popt['amplitude'])
    
    return (W_pop, sigma_pop, amplitude, r_star, Ls, shape, w_dims), (x_data, y_data, new_y_data, x_scale)

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def r_model_untrained(request, data_path):
    ndim_s = request.param['ndim_s']
    periodic = request.param['periodic']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    length_scale = 575.0 # 1.0 unit in model = 575.0 microns
    length_scales = [length_scale]*ndim_s
    
    # W
    W = torch.nn.Parameter(
        torch.tensor([
            [0.9283, -1.0534],
            [0.3060, -0.2324],
        ]),
        requires_grad=False,
    )

    # sigma
    sigma_s = torch.nn.Parameter(
        torch.tensor([
            [0.2336, 0.0679],
            [0.2584, 0.3015],
        ]),
        requires_grad=False,
    )
    
    # r_star
    _, y_data_base, _ = load_exp_data(f'{data_path}/experiment/baselines/base_by_dist.txt')
    r_star = torch.tensor([1.0, 1.2], dtype=torch.float) * torch.tensor(np.mean(y_data_base), dtype=torch.float)
    r_star = torch.nn.Parameter(r_star, requires_grad=False)
    
    # amplitude
    amplitude = torch.nn.Parameter(
        torch.tensor(0.5666), # for debugging, see github branch debug_2
        requires_grad=False,
    )

    if ndim_s == 1:
        Ls = [2*1.0]*ndim_s # we want our model to be twice the data length scale
    else:
        Ls = [2*1.0]*ndim_s
    
    if periodic:
        if ndim_s == 1:
            shape = tuple([100]*ndim_s)
        else:
            shape = tuple([30]*ndim_s)
        w_dims = list(range(ndim_s))
        period = Ls[0]
    else:
        if ndim_s == 1:
            shape = tuple([101]*ndim_s)
        else:
            shape = tuple([31]*ndim_s)
        w_dims = []
        period = 2*np.pi
    
    r_model = rmd.RadialSteadyStateResponse(
        amd.SpatialSSNModel(W, sigma_s, ndim_s, w_dims=w_dims, period=period), 
        gridtools.Grid(Ls, shape, w_dims=w_dims, device=device).cpu(), 
        r_star, 
        amplitude, 
        0, 
        0,
        length_scales,
    )
    
    return r_model

@pytest.fixture
def r_model_trained(request, data_path):
    idx = request.param['idx']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    responses_path = pathlib.Path(f'{data_path}/responses')
    filename = list(sorted(responses_path.glob('*.pkl')))[idx]

    data = utils.io.load_data(filename)

    model_data, exp_data = get_data(f"{data_path}/{data['path']}")
    W_pop, sigma_pop, amplitude, r_star, Ls, shape, w_dims = model_data
    x_data, y_data, new_y_data, x_scale = exp_data
    
    length_scales = [x_scale]*sigma_pop.shape[-1]

    W = torch.nn.Parameter(torch.tensor(W_pop, dtype=torch.float), requires_grad=False)
    sigma = torch.nn.Parameter(torch.tensor(sigma_pop, dtype=torch.float), requires_grad=False)
    r_star = torch.nn.Parameter(torch.tensor(r_star, dtype=torch.float), requires_grad=False)
    amplitude = torch.nn.Parameter(torch.tensor(amplitude, dtype=torch.float), requires_grad=False)
    
    r_model = rmd.SteadyStateResponse(
        amd.GaussianSSNModel(W, sigma, w_dims=w_dims), 
        gridtools.Grid(Ls, shape, w_dims=w_dims, device=device).cpu(), 
        r_star, 
        amplitude, 
        0, 
        0,
        length_scales,
    )
    
    return r_model

@pytest.fixture
def trained_responses(request, data_path):
    idx = request.param['idx']
    
    responses_path = pathlib.Path(f'{data_path}/responses')
    filename = list(sorted(responses_path.glob('*.pkl')))[idx]

    data = utils.io.load_data(filename)
    
    return data['data']

@pytest.fixture
def response_data(data_path):
    data_path = f'{data_path}/experiment/space_resp/resp_geq500_min.txt'
    
    x_data, y_data_mean, y_data_sem = load_exp_data(data_path)
    
    x_data = torch.tensor(x_data, dtype=torch.float)
    y_data_mean = torch.tensor(y_data_mean, dtype=torch.float)
    y_data_sem = torch.tensor(y_data_sem, dtype=torch.float)
    
    return x_data, y_data_mean, y_data_sem

@pytest.fixture()
def make_benchmark(request):
    def _make_benchmark():
        return pytest_benchmark.plugin.benchmark.__pytest_wrapped__.obj(request)
    return _make_benchmark