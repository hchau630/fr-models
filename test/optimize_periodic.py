import logging
import uuid
import pathlib
import argparse

import torch
import numpy as np

from fr_models import analytic_models as amd
from fr_models import response_models as rmd
from fr_models import optimize as optim
from fr_models import gridtools, criteria, regularizers
from fr_models import constraints as con
import utils

ENV = {
    "models_path": "/home/hc3190/ken/V1-perturb/data/models",
    "exp_path": "/home/hc3190/ken/V1-perturb/data/experiment",
}

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

def get_dataset():
    data_path = f'{ENV["exp_path"]}/space_resp/resp_geq500_min.txt'
    
    # Define training data
    x_data, y_data_mean, y_data_sem = load_exp_data(data_path)
    # y_data = np.random.normal(y_data_mean, y_data_sem) # sample data
    y_data = y_data_mean # for debugging, see github branch debug_2
    x_data = torch.tensor(x_data, dtype=torch.float)
    y_data = torch.tensor(y_data, dtype=torch.float)
    
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    return dataset

def get_model(device='cpu'):
    # length scale of model
    length_scale = torch.tensor(575.0) # 1.0 in model = 575.0 microns
    
    # Define parameters and model
    b = optim.Bounds(epsilon=1.0e-8)
    
    # W
    w_dist = torch.distributions.Normal(0.0,1.0)
    W = optim.Parameter(
        # torch.tensor([
        #     [w_dist.sample().abs(), -w_dist.sample().abs()],
        #     [w_dist.sample().abs(), -w_dist.sample().abs()],
        # ]),
        torch.tensor([
            [0.9283, -1.0534],
            [0.3060, -0.2324],
        ]), # for debugging, see github branch debug_2
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
        torch.tensor(
            [[0.2336, 0.0679],
             [ 0.2584, 0.3015]]
        ), # for debugging, see github branch debug_2
        bounds=torch.tensor(sigma_bounds),
    )
    
    # r_star
    _, y_data_base, _ = load_exp_data(f'{ENV["exp_path"]}/baselines/base_by_dist.txt')
    r_star = torch.tensor([1.0, 1.2], dtype=torch.float) * torch.tensor(np.mean(y_data_base), dtype=torch.float)
    r_star = optim.Parameter(r_star, requires_optim=False)
    
    # amplitude
    a_dist = torch.distributions.Normal(0.0,1.0)
    amplitude = optim.Parameter(
        # a_dist.sample().abs(),
        torch.tensor(0.5666), # for debugging, see github branch debug_2
        bounds=torch.tensor(b.pos),
    )
    
    ndim_s = 1
    Ls = [2*1.0]*ndim_s # we want our model to be twice the data length scale
    shape = tuple([100]*ndim_s)
    w_dims = [0]
    period = Ls[0]
    
    model = rmd.RadialSteadyStateResponse(
        amd.SpatialSSNModel(W, sigma_s, ndim_s, w_dims=w_dims, period=period), 
        gridtools.Grid(Ls, shape, w_dims=w_dims, device=device).cpu(), 
        r_star, 
        amplitude, 
        0, 
        0,
        length_scale,
        max_t=500.0,
        # method='static',
        # steady_state_kwargs=dict(
        #     dr_rtol=1.0e-4, 
        #     dr_atol=1.0e-6,
        # )
    )
    return model

def train(model, x, y, device='cpu'):
    # Move model and dataset to device
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    
    # Define regularizer
    regularizer = regularizers.WeightNormReg(lamb=0.001)
    regularizer.to(device)
    
    # Define criterion
    criterion = criteria.NormalizedLoss()
    criterion.to(device)
    
    # Define constraints
    constraints = [
        # con.SpectralRadiusCon(max_spectral_radius=0.99, trials=1),
        con.StabilityCon(max_instability=0.99, use_circulant=True),
        con.ParadoxicalCon(cell_type=1, min_subcircuit_instability=1.01, use_circulant=True),
    ]
    
    # Define optimizer
    optimizer = optim.Optimizer(
        model,
        criterion,
        constraints=constraints,
        callback=None,
        tol=1.0e-6,
        use_autograd=True,
        options={'maxiter': 100}
    )
    
    # Optimize
    success, loss = optimizer(x, y)

    return success, loss

def parse_args():
    parser = argparse.ArgumentParser(description='Data fitting')
    parser.add_argument('-l', '--log-level', type=str, choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO',
                        help="Logging level. Choices: ['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']. Default: 'INFO'")
    args = parser.parse_args()
    return args
    
def main():
    # Parse args
    args = parse_args()
    
    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(message)s')

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get model and dataset
    model = get_model(device=device)
    dataset = get_dataset()
    x, y = dataset.tensors
    
    # Train
    success, loss = train(model, x, y, device=device)
    
#     # Save results
#     if success:
#         model_name = uuid.uuid4()
#         path = pathlib.Path(f'{ENV["models_path"]}/{model_name}')
#         path.mkdir()

#         torch.save(model.state_dict(), f'{path}/state_dict.pt')
#         torch.save(dataset.tensors, f'{path}/dataset.pt')
#         utils.io.save_config(f'{path}/meta.json', {'loss': loss})

#         logger.info(f"Saving. Model name: {model_name}, loss: {loss}")
    
if __name__ == '__main__':
    print("Starting data fitting...")
    main()