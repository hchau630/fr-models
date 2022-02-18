import functools
import logging
import pprint
import time

import torch
import numpy as np
from scipy.optimize import minimize

from . import exceptions
from . import constraints as con

logger = logging.getLogger(__name__)

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
    def __init__(self, model, criterion, regularizer=None, method=None, constraints=[], tol=None, callback=None, options=None, use_autograd=True, timeout=1000):
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
        self.constraints = list(map(lambda c: {'type': c.type.value, 'fun': self.constraint_wrapper(c)}, constraints))
        self.constraint_names = [type(constraint).__name__ for constraint in constraints]
        self.tol = tol
        self.callback = callback
        self.options = options
        self.use_autograd = use_autograd
        self.timeout = timeout
    
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
        
        @functools.wraps(f)
        def wrapped_f(params, **kwargs):
            self.params = params
            with torch.no_grad():
                result = f(self.model, **kwargs)
                return result

        return wrapped_f
    
    def compute_loss(self, x, y):
        try:
            y_pred = self.model(x)
        except (exceptions.NumericalModelError, exceptions.TimeoutError) as err:
            logger.debug(f"Caught exception inside compute_loss: {err}")
            return torch.tensor(np.inf)

        logger.debug(f"y_pred, y: {y_pred}, {y}")
        return self.criterion(y_pred, y)
        
    def __call__(self, x, y):
        logger.info("Started optimizing...")
        logger.debug(pprint.pformat(self.state_dict()))
        start_time = time.time()
        
        loss_hist = []
        params_hist = []
        satisfied_hist = []
        
        def fun(params):
            with torch.set_grad_enabled(self.use_autograd):
                self.params = params
                
                loss = self.compute_loss(x, y)
                if self.regularizer is not None:
                    loss += self.regularizer(self.model)
            
            if self.use_autograd:
                if loss.item() == np.inf:
                    return loss.item(), [np.nan for _ in range(len(params))]
                self.model.zero_grad()
                loss.backward()
                
                return loss.item(), self.params_grad.tolist()
            return loss.item()
        
        def callback(params):
            if (time_taken := time.time() - start_time) > self.timeout:
                raise exceptions.TimeoutError(f"optimizer.optimize has been running for {time_taken} seconds and timed out.")
                
            with torch.no_grad():
                self.params = params
                
                # logger.debug(pprint.pformat(self.state_dict()))
                
                all_satisfied = []
                values_dict = {}
                
                for i, constraint in enumerate(self.constraints):
                    val = constraint['fun'](params)
                    if constraint['type'] == con.Types.EQ.value:
                        satisfied = val == 0
                    elif constraint['type'] == con.Types.INEQ.value:
                        satisfied = val >= 0
                    else:
                        raise RuntimeError()

                    values_dict[self.constraint_names[i]] = val
                    all_satisfied.append(satisfied)
                    
                is_all_satisfied = all(all_satisfied)
                
                loss = self.compute_loss(x, y)
                loss_hist.append(loss.item())
                params_hist.append(params)
                satisfied_hist.append(is_all_satisfied)
                
                logger.info(f"Loss: {loss.item()}, constraints: {values_dict}")
                
                if self.callback is not None:
                    self.callback(self, x, y)

        try:
            result = minimize(fun,
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
            # logger.debug("Details of the exception: ", exc_info=True)
            
            logger.info(f"Loss hist: {loss_hist}")
            logger.info(f"Satisfied hist: {satisfied_hist}")
            
            if len(loss_hist) == 0:
                logger.info("No result returned since len(loss_hist) = 0.")
                return False, np.inf
            
            loss_hist = np.array(loss_hist)
            params_hist = np.array(params_hist)
            satisfied_hist = np.array(satisfied_hist)
            
            satisfied_loss_hist = loss_hist[satisfied_hist] # get only the losses where constraint is satisfied
            satisfied_params_hist = params_hist[satisfied_hist]
    
            logger.info(f"Satisfied loss hist: {satisfied_loss_hist}")
        
            if len(satisfied_loss_hist) == 0:
                logger.info("No result returned since len(satisfied_loss_hist) = 0.")
                return False, np.inf
        
            idx, min_loss = np.argmin(satisfied_loss_hist), np.min(satisfied_loss_hist)
            self.params = satisfied_params_hist[idx]
            logger.info(f"Returning result during optimization. Loss: {min_loss}.")
            return True, min_loss
        
        with torch.no_grad():
            self.params = result.x
            loss = self.compute_loss(x, y).item()
        logger.info(f"Finished optimization successfully. Loss: {loss}")
        return True, loss