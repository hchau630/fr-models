import pprint
import logging
import uuid
import pathlib

import torch
import numpy as np

from fr_models import analytic_models as amd
from fr_models import response_models as rmd
from fr_models import optimize as optim
from fr_models import gridtools, criteria, regularizers
from fr_models import constraints as con
import utils

logger = logging.getLogger(__name__)

def load_exp_data(filepath, x_cutoff=300.0, symmetric=False, normalize=0.5):
    data = np.loadtxt(filepath, delimiter=',')
    x_data, y_data, y_data_sem = data[0], data[1], data[2]
    length = np.sum(x_data < x_cutoff)
    if symmetric:
        length = length - 1 if length % 2 == 0 else length
    x_data, y_data, y_data_sem = x_data[:length], y_data[:length], y_data_sem[:length]
    if normalize is not None:
        scale = np.max(x_data)/normalize
        x_data = x_data/scale
    else:
        scale = 1.0
    return x_data.reshape(-1,1), y_data, y_data_sem, scale
    
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define length scale for model data
    L = 1.0
    
    # Define parameters and model
    b = optim.Bounds(epsilon=1.0e-8)
    
    # W
    w_dist = torch.distributions.Normal(0.0,1.0)
    W = optim.Parameter(
        # torch.tensor([
        #     [w_dist.sample().abs(), -w_dist.sample().abs()],
        #     [w_dist.sample().abs(), -w_dist.sample().abs()],
        # ]),
        # torch.tensor([
        #     [0.79903045, -0.22798239],
        #     [0.78063547, -0.01],
        # ]),
        torch.tensor([
            [0.9283, -1.0534],
            [0.3060, -0.2324],
        ]),
        bounds=torch.tensor([
            [b.pos, b.neg],
            [b.pos, b.neg],
        ]),
        requires_optim=True,
    )
    
    # sigma
    sigma_bounds = [0.01,0.5]
    s_dist = torch.distributions.Uniform(*sigma_bounds)
    sigma_s = optim.Parameter(
        # s_dist.sample((2,2)),
        # torch.tensor(
        #     [[30.60939689, 31.54267749],
        #      [ 5.82356181, 11.72982061]]
        # )/575.0,  
        torch.tensor(
            [[0.2336, 0.0679],
             [ 0.2584, 0.3015]]
        ),
        bounds=torch.tensor(sigma_bounds),
    )
    ndim_s = 1 # 2 spatial dimensions
    
    # amplitude
    a_dist = torch.distributions.Normal(0.0,1.0)
    amplitude = optim.Parameter(
        # a_dist.sample().abs(),
        # torch.tensor(1.5973258580974314),
        torch.tensor(0.5666),
        bounds=torch.tensor(b.pos),
    )
    
    Ls = [2*L]*ndim_s # we want our model to be twice as long as actual data
    shape = tuple([101]*ndim_s)
    w_dims = []
    
    a_model = amd.SpatialSSNModel(W, sigma_s, ndim_s, w_dims=w_dims)
    grid = gridtools.Grid(Ls, shape, w_dims=w_dims, device=device)
    _, y_data_base, _, _ = load_exp_data('/home/hc3190/ken/spatial-model/data/baselines_new/base_by_dist.txt', normalize=L/2)
    r_star = torch.tensor([np.mean(y_data_base), 1.2*np.mean(y_data_base)], dtype=torch.float)
    r_star = optim.Parameter(r_star, requires_optim=False)
    
    solver_kwargs = None
    model = rmd.SteadyStateResponse(
        a_model, 
        grid, 
        r_star, 
        amplitude, 
        torch.tensor(0), 
        torch.tensor(0), 
        dr_rtol=1.0e-4, 
        dr_atol=1.0e-6, 
        max_t=500.0, 
        solver_kwargs=solver_kwargs
    )
    model.to(device)
    
    # Define regularizer
    regularizer = regularizers.WeightNormReg(lamb=0.001)
    regularizer.to(device)
    
    # Define criterion
    criterion = criteria.NormalizedLoss()
    criterion.to(device)
    
    # Define constraints
    constraints = [
        # con.SpectralRadiusCon(max_spectral_radius=0.99, trials=1),
        con.StabilityCon(max_instability=0.99),
        con.ParadoxicalCon(cell_type=1, min_subcircuit_instability=1.01),
    ]
    
    # Define optimizer
    optimizer = optim.Optimizer(model, criterion, constraints=constraints, callback=None, tol=1.0e-6, use_autograd=True, options={'maxiter': 100})
    
    # Define training data
    x_data, y_data_mean, y_data_sem, scale = load_exp_data('/home/hc3190/ken/spatial-model/data/space_resp/resp_geq500_min.txt', normalize=L/2)
    # y_data = np.random.normal(y_data_mean, y_data_sem) # sample data
    y_data = y_data_mean
    x_data = np.concatenate([x_data,np.zeros((len(x_data),ndim_s-1))],axis=-1)
    x_data = torch.tensor(x_data, dtype=torch.float, device=device)
    y_data = torch.tensor(y_data, dtype=torch.float, device=device)
    
    # Optimize
    success, loss = optimizer(x_data, y_data)

    return success, loss, optimizer.model, optimizer.state_dict()
    
def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    # logging.getLogger('fr_models.optimize').setLevel(logging.INFO)
    for _ in range(1000):
        success, loss, model, state_dict = train()
        if success:
            model_name = uuid.uuid4()
            path = pathlib.Path(f'/home/hc3190/ken/fr-models/test/data/trained_models/{model_name}')
            path.mkdir()
            torch.save(state_dict, f'{path}/state_dict.pth.tar')
            utils.io.save_config(f'{path}/meta.json', {'loss': loss})
            logger.info(f"Saving. Model name: {model_name}, loss: {loss}, state_dict:")
            logger.info(pprint.pformat(state_dict))
    
if __name__ == '__main__':
    main()