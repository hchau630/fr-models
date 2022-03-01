import functools
import logging
import time

import torch
import numpy as np
import scipy.optimize as sp_opt

from . import exceptions
from . import constraints as con
from . import timeout

logger = logging.getLogger(__name__)

def ndarray_to_tuple(f):
    @functools.wraps(f)
    def wrapped_f(x):
        return f(tuple(x.tolist()))
    return wrapped_f

def pformat(d, indent=0, spaces=4):
    output = ''
    for key, value in d.items():
        output += ' ' * spaces * indent + str(key) + ':' + '\n'
        if isinstance(value, dict):
            output += pformat(value, indent+1) + '\n'
        else:
            output += '\n'.join([' ' * spaces * (indent+1) + line for line in str(value).split('\n')]) + '\n'
    return output

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
    def __init__(self, model, criterion, regularizer=None, method=None, constraints=[], tol=None, options=None, use_autograd=True, timeout=1000, cache_size=128):
        """
        model should be a torch.nn.Module. This function optimizes all optimizer.Parameter instances in model.
        parameters.bounds will also be used to ensure the parameter lies within bounds. 

        Parameters:
          model: a torch.nn.Module
          constraints: a list of constraints.Constraint objects

        """
        self.model = model
        self.criterion = criterion
        self.regularizer = regularizer
        self.method = method
        self.tol = tol
        self.options = options
        self.use_autograd = use_autograd
        self.timeout = timeout
        self.cache_size = cache_size
        
        self.constraints = list(map(lambda c: {'type': c.type.value, 'fun': self.constraint_wrapper(c)}, constraints))
        self.constraint_names = [type(constraint).__name__ for constraint in constraints]
    
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
    
    def state_dict(self):
        state_dict = {}
        for name, param in self.model.named_parameters():
            if isinstance(param, Parameter):
                state_dict[name] = param.data
        return state_dict
    
    def constraint_wrapper(self, f):
        
        @ndarray_to_tuple
        @functools.lru_cache(maxsize=self.cache_size)
        def wrapped_f(params):
            self.params = params
            with torch.no_grad():
                result = f(self.model)
                return result

        return wrapped_f
    
    def make_loss_func(self, x, y):
        
        @ndarray_to_tuple
        @functools.lru_cache(maxsize=self.cache_size)
        def loss_func(params):
            with torch.set_grad_enabled(self.use_autograd):
                self.params = params
                
                try:
                    y_pred = self.model(x)
                    
                except (exceptions.NumericalModelError, exceptions.TimeoutError) as err:
                    logger.debug(f"Caught exception inside compute_loss: {err}")
                    
                    loss = torch.tensor(np.inf)
                    
                else:
                    loss = self.criterion(y_pred, y)
                
                return loss
            
        return loss_func
    
    def make_minimizer_func(self, x, y, loss_func):
        
        def minimizer_func(params):
            loss = loss_func(params)
            
            if self.regularizer is not None:
                loss += self.regularizer(self.model)
            
            loss_item = loss.item()
            
            if self.use_autograd:
                if loss_item == np.inf or loss.grad_fn is None: # either bad params or non-differentiable
                    grad = [np.nan for _ in range(len(params))]
                    
                else:
                    self.model.zero_grad()
                    loss.backward()
                    
                    grad = self.params_grad.tolist()
                    
                return loss_item, grad
            
            return loss_item
        
        return minimizer_func
        
    def make_callback(self, x, y, loss_func, hist):
        
        def callback(params):
            # Compute loss, hopefully result is in cache
            loss = loss_func(params)
            loss_item = loss.item()
            
            # Compute constraints, hopefully results are in cache
            all_satisfied = True
            constraint_values_dict = {}

            for constraint_name, constraint in zip(self.constraint_names, self.constraints):
                val = constraint['fun'](params)
                if constraint['type'] == con.Types.EQ.value:
                    satisfied = val == 0
                elif constraint['type'] == con.Types.INEQ.value:
                    satisfied = val >= 0
                else:
                    raise RuntimeError()

                constraint_values_dict[constraint_name] = val
                all_satisfied = all_satisfied and satisfied
                
            hist['loss'].append(loss_item)
            hist['satisfied'].append(all_satisfied)
            hist['params'].append(params)

            logger.info(f"Loss: {loss_item}, constraints: {constraint_values_dict}")
            logger.debug(f"Loss func cache info: {loss_func.__wrapped__.cache_info()}")
            for constraint_name, constraint in zip(self.constraint_names, self.constraints):
                logger.debug(f"{constraint_name} cache info: {constraint['fun'].__wrapped__.cache_info()}")
            logger.debug(self.params.detach().cpu())
            # logger.debug(pformat(self.state_dict()))
            
        return callback
        
    def __call__(self, x, y):
        logger.info("Started optimizing...")
        logger.debug(self.params.detach().cpu())
        # logger.debug(pformat(self.state_dict()))
        
        hist = {'loss': [], 'satisfied': [], 'params': []}
        
        loss_func = self.make_loss_func(x, y)
        minimizer_func = self.make_minimizer_func(x, y, loss_func)
        callback = self.make_callback(x, y, loss_func, hist)
        
        minimize = timeout.timeout(seconds=1000)(sp_opt.minimize)
        
        try:
            result = minimize(minimizer_func,
                self.params.tolist(),
                method=self.method,
                jac=self.use_autograd,
                bounds=self.bounds,
                constraints=self.constraints,
                tol=self.tol,
                callback=callback,
                options=self.options,
            )
            if not result.success:
                if result.message == 'Iteration limit reached':
                    raise exceptions.IterationStepsExceeded(result.message)
                raise exceptions.OptimizationError(result.message)
            
        except (exceptions.NumericalModelError, exceptions.OptimizationError, exceptions.TimeoutError) as err:
            logger.info(f"Optimization failed due to exception: {err}")
            
            logger.info(f"Loss hist: {hist['loss']}")
            logger.info(f"Satisfied hist: {hist['satisfied']}")
            
            if len(hist['loss']) == 0:
                logger.info("No result returned since len(hist['loss']) = 0.")
                return False, np.inf
            
            loss_hist = np.array(hist['loss'])
            satisfied_hist = np.array(hist['satisfied'])
            params_hist = np.array(hist['params'])
            
            satisfied_loss_hist = loss_hist[satisfied_hist] # get only the losses where constraint is satisfied
            satisfied_params_hist = params_hist[satisfied_hist]
    
            logger.info(f"Satisfied loss hist: {satisfied_loss_hist}")
        
            if len(satisfied_loss_hist) == 0:
                logger.info("No result returned since len(satisfied_loss_hist) = 0.")
                return False, np.inf
        
            idx = np.argmin(satisfied_loss_hist)
            
            self.params = satisfied_params_hist[idx]
            loss = satisfied_loss_hist[idx]
            
            logger.info(f"Returning result during optimization. Loss: {loss}.")
            logger.debug(self.params.detach().cpu())
            # logger.debug(pformat(self.state_dict()))
            return True, loss
        
        self.params = result.x
        loss = loss_func(result.x)
        
        logger.info(f"Finished optimization successfully. Loss: {loss}")
        logger.debug(self.params.detach().cpu())
        # logger.debug(pformat(self.state_dict()))
        return True, loss