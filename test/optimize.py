import torch
import numpy as np

from fr_models import analytic_models as amd
from fr_models import response_models as rmd
from fr_models import optimize as optim
from fr_models import gridtools
from fr_models import criteria

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define length scale for space
    L = 1.0
    
    # Define parameters and model
    b = optim.Bounds(epsilon=1.0e-8)
    W = optim.Parameter(
        torch.tensor([
            [1.0, 1.0],
            [1.0, 1.0],
        ]),
        bounds=torch.tensor([
            [b.pos, b.neg],
            [b.pos, b.neg],
        ]),
    )
    sigma_s = optim.Parameter(
        torch.tensor([
            [1.0, 1.0],
            [1.0, 1.0],
        ]),
        bounds=torch.tensor(b.pos),
    )
    ndim_s = 2 # 2 spatial dimensions
    amplitude = optim.Parameter(
        torch.tensor(1.0),
        bounds=torch.tensor(b.pos),
    )
    
    Ls = [L]*ndim_s
    shape = tuple([51]*ndim_s)
    w_dims = []
    
    a_model = amd.SpatialSSNModel(W, sigma_s, ndim_s, w_dims=w_dims)
    grid = gridtools.Grid(Ls, shape, w_dims=w_dims, device=device)
    _, y_data_base, _, _ = load_exp_data('/home/hc3190/ken/spatial-model/data/baselines_new/base_by_dist.txt', normalize=L/2)
    r_star = torch.tensor([np.mean(y_data_base), 1.2*np.mean(y_data_base)], dtype=torch.float)
    r_star = optim.Parameter(r_star, requires_optim=False)
    
    model = rmd.SteadyStateResponse(a_model, grid, r_star, amplitude, torch.tensor(0), torch.tensor(0))
    model.to(device)
    
    # Define criterion
    criterion = criteria.NormalizedMSELoss()
    criterion.to(device)
    
    # Define constraints
    constraints = []
    
    # Define optimizer
    optimizer = optim.Optimizer(model, criterion, constraints=constraints)
    
    # Define training data
    x_data, y_data_mean, y_data_sem, scale = load_exp_data('/home/hc3190/ken/spatial-model/data/space_resp/resp_geq500_min.txt', normalize=L/2)
    y_data = np.random.normal(y_data_mean, y_data_sem) # sample data
    x_data = np.concatenate([x_data,np.zeros((len(x_data),ndim_s-1))],axis=-1)
    x_data = torch.tensor(x_data, dtype=torch.float, device=device)
    y_data = torch.tensor(y_data, dtype=torch.float, device=device)
    
    # Optimize
    success, loss = optimizer(x_data, y_data)
    
    # Save results
    threshold = 0.8
    print(success, loss)
    if success and loss < threshold:
        pass # save data
    
if __name__ == '__main__':
    main()