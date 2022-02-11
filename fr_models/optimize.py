import functools 

import torch
import numpy as np
from scipy.optimize import minimize

class Bounds:
    def __init__(self, epsilon=0):
        self._epsilon = epsilon
        
    @property
    def epsilon(self):
        return self._epsilon
        
    @property
    def none(self):
        return [-np.inf, np.inf]
    
    @property
    def neg(self):
        return [-np.inf, -self.epsilon]
    
    @property
    def pos(self):
        return [self.epsilon, np.inf]

class Parameter(torch.nn.Parameter):
    """
    A subclass of torch.nn.Parameter that has an additional attribute
    requires_optim, which is a torch.BoolTensor that specifies
    which elements of the data should be optimized.
    If bounds is not None, bounds should be either be a tuple or
    a torch.Tensor with size (*data.shape, 2), which specifies the 
    lower and upper bound of either all elements or each individual
    element respectively.
    
    Admittedly, this design choice has the drawback that it
    slightly mixes model logic with optimization logic, since
    the decision of which parameters should be optimized is now
    made during the model construction step. As a result, it is
    a little less modular, in the sense that changing the optimization
    procedure changes the model, so there cannot be a single model
    shared across different optimization procedures. But I think it is
    more user-friendly this way, since you don't need to specify
    that something like index idx1, idx2 of module.submodule.parameter_A
    should be optimized. Also, the pytorch paradigm slightly mixes model
    with optimization logic anyway, due to the fact that nn.modules contain
    optimization initialization code inside them.
    """
    def __new__(cls, data, requires_optim=True, **kwargs):
        if isinstance(requires_optim, torch.Tensor):
            requires_grad = torch.any(requires_optim).item()
        else:
            requires_grad = requires_optim
        return super().__new__(cls, data, requires_grad)
    
    def __init__(self, data, requires_optim=True, bounds=None):
        if isinstance(requires_optim, bool):
            if requires_optim:
                requires_optim = torch.ones(data.shape, dtype=torch.bool)
            else:
                requires_optim = torch.zeros(data.shape, dtype=torch.bool)
        self.requires_optim = requires_optim
        if bounds is None:
            bounds = torch.stack([-torch.ones(data.shape)*np.inf, torch.ones(data.shape)*np.inf],dim=-1)
        bounds = bounds.expand((*data.shape,2)) # broadcast
        self.bounds = bounds
        
class Optimizer():
    def __init__(self, model, criterion, method=None, constraints=[], tol=None, callback=None, options=None):
        """
        model should be a torch.nn.Module. This function optimizes all optimizer.Parameter instances in model.
        parameters.bounds will also be used to ensure the parameter lies within bounds. 

        Parameters:
          model: a torch.nn.Module
          constraints: a list of callables that has the signature func: torch.nn.Module -> scalar

        """
        self.model = model
        self.criterion = criterion
        self.method = method
        self.constraints = list(map(lambda f: self.func_wrapper(f), constraints))
        self.tol = tol
        self.callback = self.func_wrapper(callback) if callback is not None else None
        self.options = options
    
    @property
    def params(self):
        """
        Returns a 1D-tensor of optimizable parameters
        """
        params = []
        for name, param in self.model.named_parameters():
            if isinstance(param, Parameter) and torch.any(param.requires_optim):
                params.append(param[param.requires_optim])
        return torch.cat(params)
    
    @property
    def params_grad(self):
        """
        Returns a 1D-tensor of optimizable parameters gradients
        """
        params_grad = []
        for name, param in self.model.named_parameters():
            if isinstance(param, Parameter) and torch.any(param.requires_optim):
                params_grad.append(param.grad[param.requires_optim])
        return torch.cat(params_grad)
    
    @params.setter
    def params(self, params):
        """
        Updates the optimizable parameters with params, which is a 1D-tensor.
        """
        assert len(params) == len(self.params)
        
        i = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if isinstance(param, Parameter) and torch.any(param.requires_optim):
                    N = param[param.requires_optim].numel()
                    param[param.requires_optim] = torch.tensor(params[i:i+N], dtype=param.dtype, device=param.device)
                    i += N
                
    @property
    def bounds(self):
        bounds = []
        for name, param in self.model.named_parameters():
            if isinstance(param, Parameter):
                bounds += param.bounds[param.requires_optim].tolist()
        return bounds
    
    def func_wrapper(self, f):
        
        @functools.wraps(f)
        def wrapped_f(params):
            self.params = params

            with torch.no_grad():
                return f(self.model)

        return wrapped_f
        
    def __call__(self, x, y):
        def fun(params):
            self.params = params

            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

            self.model.zero_grad()
            loss.backward()
            
            print(self.params)
            
            return loss.item(), self.params_grad.tolist()

        result = minimize(fun,
            self.params.tolist(),
            method=self.method,
            jac=True,
            bounds=self.bounds,
            constraints=self.constraints,
            tol=self.tol,
            callback=self.callback,
            options=self.options,
        )

        self.params = result.x
        loss = self.criterion(self.model(x), y).item()
        return result.success, loss