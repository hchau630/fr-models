import abc
from functools import partial

from torchdiffeq import odeint, odeint_event
import numpy as np
import torch

from . import _torch, gridtools, exceptions, timeout

ADAPTIVE_SOLVERS = ['dopri8', 'dopri5', 'bosh3', 'adaptive_heun']

class NumericalModel(abc.ABC, torch.nn.Module):
    def __init__(self, W):
        assert W.ndim == 2
        super().__init__()
        
        self.tau = 1.0  # always use tau = 1.0 for convenience
        self.N = W.shape[0]
        self.W = W
        
    @abc.abstractmethod
    def _drdt(self, t, r, h):
        return
        
    def forward(self, h, r0, t, event_fn=None, odeint_interface=odeint, **kwargs):
        """
        Solves an ODE with input h and initial state r0 at time stamps t.
        Args:
            h: a function that maps a 0-dimensional tensor t to a tensor with shape (*,N),
               or a tensor with shape (*,N) that is interpreted as constant input
            r0: a 1-dimensional tensor with shape (**,N) or a 0-dimensional tensor, which is interpreted
               to be constant across neurons
            t: If event_fn=None, t is a 1-dimensional tensor that specifies the time stamps
               at which the ODE is solved, and whose first element represents the initial time,
               or a 0-dimensional tensor that specifies the end time, with the initial time interpreted as 0.
               If event_fn is not None, t must be a 0-dimensional tensor that specifies start time.
            event_fn: Function that maps (t,r,h) to a Tensor. The solve terminates when
                      any element of event_fn(t,r,h) evaluates to zero.
            odenint_interface: either torchdiffeq.odeint or torchdiffeq.odeint_adjoint
               
        Returns:
            (Let *** denote the shape obtained by broadcasting * and **)
            r: If event_fn is None: 
                   If t is a 1-dimensional tensor with length L, then
                   r is a tensor with shape (L,***,N), with r[i] being the firing rates at t[i].
                   If t is a 0-dimensional tensor, then r is tensor with shape (***,N) 
                   that represents the firing rates at t.
               else:
                   r is tensor with shape (***,N) that represents the firing rates at event_t
            t: If event_fn is None:
                   t is the same as the input t
               else:
                   t is the event time.
                   
        """
        if 'options' not in kwargs or ('max_num_steps' not in kwargs['options'] and 'method' in ADAPTIVE_SOLVERS):
            kwargs['options'] = {'max_num_steps': 100} # by default, set max_num_steps to 100 if it is not provided and method is not a fixed solver.
            
        if event_fn is None:
            if t.ndim == 0: # if t is a scalar tensor
                _t = torch.Tensor([0,t]).to(t.device)
            elif t.ndim == 1:
                _t = t
            else:
                raise ValueError("t must be a one or zero dimensional tensor")
            t0 = _t[0]
        else:
            assert t.ndim == 0
            t0 = t
            
        if not callable(h):
            h0 = h
            h = lambda t, h=h: h
        else:
            h0 = h(t0)
            h = h
            
        assert h0.shape[-1] == self.N # asserts h has correct shape (*,N)
            
        r0, _ = torch.broadcast_tensors(r0, h0) # (***,N)

        drdt = partial(self._drdt, h=h)
        
        try:
            if event_fn is None:
                r = odeint_interface(drdt, r0, _t, **kwargs) # (L,***,N)

                if t.ndim == 0:
                    return r[-1], t # r - (***,N)
                
                return r, t # r - (L,***,N)

            else:
                event_fn = partial(event_fn, h=h)
                event_t, r = odeint_event(drdt, r0, t0, event_fn=event_fn, odeint_interface=odeint_interface, **kwargs) # r - (L,***,N)

                return r[-1], event_t # r - (***,N)
        except AssertionError as err:
            # Sometimes odeint raises an error AssertionError('underflow in dt ...') or AssertionError('max_num_steps exceeded ...')
            # This catches that error and re-raises the error with a more detailed 
            # explanation of that error message.
            if str(err).startswith('underflow in dt'):
                raise exceptions.RequiredStepSizeTooSmall(f'AssertionError in odeint: {err}. This is due to the adaptive solver needing to use a step size that is too small, which in turn implies the problem is likely too stiff. See https://github.com/rtqichen/torchdiffeq/blob/master/FAQ.md')
            if str(err).startswith('max_num_steps exceeded'):
                raise exceptions.IterationStepsExceeded(f'AssertionError in odeint: {err}. If you wish to allow the solver to run longer, set "max_num_steps" inside the options dict to something larger')
            raise
            
    def steady_state(self, h, r0, max_t, method='dynamic', **kwargs):
        if method == 'dynamic':
            return self._steady_state_dynamic(h, r0, max_t, **kwargs)
        elif method == 'static':
            return self._steady_state_static(h, r0, max_t, **kwargs)
        raise NotImplementedError("Only 'dynamic' and 'static' are valid options for the keyword argument 'method'.")
    
    def _steady_state_dynamic(self, h, r0, max_t, dr_rtol=1.0e-4, dr_atol=1.0e-6, epsilon_t=1.0e-1, solver_kwargs=None):
        if solver_kwargs is None:
            solver_kwargs = {} # mutable defaults are generally bad
            
        assert max_t > 0
        
        max_t = max_t*self.tau
        epsilon_t = epsilon_t*self.tau
        
        def event_fn(t, r, h, dr_rtol=dr_rtol, dr_atol=dr_atol, max_t=max_t):
            drdt = self._drdt(t, r, h) * self.tau # how much dr changes in self.tau time if we extrapolate from the infinitesimal change
            allclose = _torch.allclose(drdt, torch.zeros(r.shape, device=r.device), rtol=r*dr_rtol, atol=dr_atol)
            exceeded_max_t = t > max_t
            result = (1.0 - (allclose | exceeded_max_t).float()).unsqueeze(0)
            return result
            
        t0 = torch.tensor(0.0, device=max_t.device)
        r, t = self.forward(h, r0, t0, event_fn=event_fn, **solver_kwargs)

        if t >= max_t - epsilon_t:
            raise exceptions.SteadyStateNotReached(f"Failed to convergence to steady state with tolerance dr_rtol={dr_rtol}, dr_atol={dr_atol} within maximum time {max_t}.")
        if t == t0:
            raise exceptions.ToleranceTooLarge("numerical model returned steady state as the inital state, probably because dr_rtol and dr_atol are too big. Try reducing those numbers.") 
            
        return r, t
    
    def _steady_state_static(self, h, r0, max_t, steps=10, dr_rtol=1.0e-4, dr_atol=1.0e-6, epsilon_t=1.0e-1, solver_kwargs=None):
        """
        max_t - torch.Tensor scalar
        """
        if solver_kwargs is None:
            solver_kwargs = {} # mutable defaults are generally bad
            
        assert max_t > 0
        
        max_t = max_t*self.tau
        epsilon_t = epsilon_t*self.tau
        
        t = torch.linspace(0, max_t, steps=steps+1, device=max_t.device)
        
        r = r0
        for ti, tf in zip(t[:-1], t[1:]):
            r = self.forward(h, r, torch.stack([ti, tf]), **solver_kwargs)[0][-1]
            # compute drdt
            dr = self.forward(h, r, torch.stack([tf, tf+epsilon_t]), **solver_kwargs)[0][-1] - r
            
            if torch.all(dr == torch.zeros(dr.shape, device=dr.device)):
                raise RuntimeError("dr is exactly 0. Try increasing epsilon_t.")
                
            drdt = dr / epsilon_t * self.tau # how much dr changes in self.tau time if we extrapolate from the infinitesimal change
            is_steady_state = _torch.allclose(drdt, torch.zeros(r.shape, device=r.device), rtol=r*dr_rtol, atol=dr_atol)
            
            if is_steady_state:
                return r, tf
        
        raise exceptions.SteadyStateNotReached(f"Failed to convergence to steady state with tolerance dr_rtol={dr_rtol}, dr_atol={dr_atol} within maximum time {max_t}.")
        
class MultiDimModel(NumericalModel):
    def __init__(self, W, **kwargs):
        assert W.ndim % 2 == 0
        
        shape = W.shape[:W.ndim // 2]
        _W = W.reshape(np.prod(shape), np.prod(shape))
        
        super().__init__(_W, **kwargs)
        
        self.shape = shape
        self.ndim = W.ndim // 2
    
    @property
    def W_expanded(self):
        return self.W.reshape(*self.shape,*self.shape)
        
    def forward(self, h, r0, t, **kwargs):
        # h - (*,*shape) or Callable[float,(*,*shape)], r0 - (**,*shape) or (,)
                
        if t.ndim == 0:
            t0 = torch.tensor(0, device=t.device)
        else:
            t0 = t[0]
        
        if not callable(h):
            h0 = h # (*,*shape)
        else:
            h0 = h(t0) # (*,*shape)

        assert h0.shape[-self.ndim:] == self.shape # asserts h has correct shape
        h_batch_shape = h0.shape[:-self.ndim]
        
        if not callable(h):
            h = h.reshape(*h_batch_shape,self.N) # (*,N)
        else:
            h = lambda t, h=h, h_batch_shape=h_batch_shape: h(t).reshape(*h_batch_shape,self.N) # (*,N)
            
        r0, _ = torch.broadcast_tensors(r0, h0) # (***,*shape)
        batch_shape = r0.shape[:-self.ndim]
        r0 = r0.reshape(*batch_shape,self.N) # (***,N)

        r, t = super().forward(h, r0, t, **kwargs) # (L,***,N) or (***,N)

        if t.ndim == 0:
            r = r.reshape(*batch_shape,*self.shape) # (***,*shape)
        else:
            r = r.reshape(len(t),*batch_shape,*self.shape) # (***,*shape)
            
        return r, t
    
class TrivialVBModel(MultiDimModel):
    """
    A model with additional topological data. The model must be a trivial vector bundle E = B x F 
    over a manifold B that is expressable as an arbitrary, finite cartesian product of R^1 or S^1 
    with the usual metric. The fibre F is a vector space.
    This should be an abstract yet specific enough base class for any reasonable models of the cortex.
    Future improvement: This class not be necessary if we have a separate DiscretizedManifold class
    along with a VectorBundle class.
    The user then defines their own DiscretizedManifold.
    They can then get the model by discretizing the kernel on the DiscretizedManifold, which
    returns a VectorBundle which are the weights, which they can use to initialize a MultiDimModel.
    """
    def __init__(self, W, F_dim, w_dims, **kwargs):
        """
        Args:
          - W: The weight tensor, with the first F_dim dimensions corresponding to the fiber space F, 
               and the rest of the dimensions correspond to the base space B.
          - F_dim: Dimensionality of the fiber space. 
          - w_dims: a list of integers denoting the dimensions (within base space B) which are 'wrapped',
                    i.e. are S^1.
        """
        super().__init__(W, **kwargs)
        self.F_dim = F_dim
        self.B_dim = self.ndim - F_dim
        assert self.F_dim >= 0 and self.B_dim >= 0
        self.F_shape = self.shape[:F_dim]
        self.B_shape = self.shape[F_dim:]
        self.F_N = np.prod(self.F_shape)
        self.B_N = np.prod(self.B_shape)
        self.w_dims = w_dims
        assert all([w_dim < self.B_dim for w_dim in self.w_dims])
    
class MultiCellModel(TrivialVBModel):
    def __init__(self, W, w_dims, **kwargs):
        assert W.ndim >= 4
        super().__init__(W, F_dim=1, w_dims=w_dims, **kwargs)
                
class FRModel(NumericalModel):
    def __init__(self, W, f, **kwargs):
        """
        User should make sure that f is a function that is compatible with torch.Tensor and not np.ndarray
        """
        super().__init__(W, **kwargs)
        self.f = f
        
    def _drdt(self, t, r, h):
        # W - (N,N), r - (***,N), h - (*,N)
        result = self.f(torch.einsum('ij,...j->...i', self.W, r) + h(t)) - r
        return result
    
class MultiCellFRModel(MultiCellModel, FRModel):
    pass
    
class SSNModel(FRModel):
    def __init__(self, W, power, **kwargs):
        f = lambda x, power: torch.clip(x,0,None)**power
        super().__init__(W, f=partial(f, power=power), **kwargs)
        self.power = power
        
class MultiCellSSNModel(MultiCellModel, SSNModel):
    def linear_perturbed_model(self, r_star, share_mem=False, **kwargs):
        if not share_mem:
            W_expanded = self.W_expanded.clone()
            w_dims = self.w_dims.copy()
        else:
            W_expanded = self.W_expanded
            w_dims = self.w_dims
        return LinearizedMultiCellSSNModel(W_expanded, r_star=r_star, w_dims=w_dims, **kwargs)
    
    def nonlinear_perturbed_model(self, r_star, share_mem=False, **kwargs):
        if not share_mem:
            W_expanded = self.W_expanded.clone()
            w_dims = self.w_dims.copy()
        else:
            W_expanded = self.W_expanded
            w_dims = self.w_dims
        return PerturbedMultiCellSSNModel(W_expanded, r_star=r_star, w_dims=w_dims, power=self.power, **kwargs)
        
class LinearizedMultiCellSSNModel(MultiCellModel):
    def __init__(self, W, r_star, w_dims, **kwargs):
        assert torch.all(r_star >= 0)
        super().__init__(W, w_dims=w_dims, **kwargs) # f is defined dynamically
        assert r_star.shape == self.F_shape
        self._r_star = r_star # (*self.F_shape), i.e. (n)
        
    @property
    def _f_prime(self):
        return 2*self._r_star**0.5 # (*self.F_shape)
    
    @property
    def r_star(self):
        device = self._r_star.device
        return torch.einsum('i,...->i...', self._r_star, torch.ones(self.B_shape, device=device)) # (*self.shape)
        
    @property
    def f_prime(self):
        return 2*self.r_star**0.5 # (*self.shape)
    
    def spectral_radius(self, use_circulant=False):
        if use_circulant:
            FW = torch.einsum('i,i...->i...',self._f_prime, self.W_expanded) # (n,*self.B_shape,n,*self.B_shape)
            return _torch.linalg.eigvalsbnc(FW, self.B_dim).abs().max()
        else:
            F = torch.diag(self.f_prime.reshape(-1))
            return torch.linalg.eigvals(F @ self.W).abs().max()

    def instability(self, use_circulant=False):
        if use_circulant:
            FW = torch.einsum('i,i...->i...',self._f_prime, self.W_expanded) # (n,*self.B_shape,n,*self.B_shape)
            return _torch.linalg.eigvalsbnc(FW, self.B_dim).real.max()
        else:
            F = torch.diag(self.f_prime.reshape(-1))
            return torch.linalg.eigvals(F @ self.W).real.max()
        
    def response_matrix(self, use_circulant=False):
        # F = torch.diag(self.f_prime.reshape(-1))
        # if use_circulant:
        #     FW = torch.einsum('i,i...->i...',self._f_prime, self.W_expanded) # (n,*self.B_shape,n,*self.B_shape)
        # else:
        #     FW = torch.diag(self.f_prime.reshape(-1)) @ self.W
        # test_inputs = torch.eye(X
        F = torch.diag(self.f_prime.reshape(-1)) # (self.N,self.N)
        FW = F @ self.W # (self.N,self.N)
        I = torch.eye(FW.shape[0], device=FW.device) # (self.N,self.N)
        R = torch.linalg.inv(I - FW) @ F # (self.N,self.N)
        return R # (self.N,self.N)
    
    def _drdt(self, t, r, h):
        result = self.f_prime.reshape(-1) * (torch.einsum('ij,...j->...i', self.W, r) + h(t)) - r # (*,self.N)
        return result
    
    def forward(self, delta_h, delta_r0, t, **kwargs):
        delta_r, t = super().forward(delta_h, delta_r0, t, **kwargs)
        return delta_r, t
    
    def steady_state(self, delta_h, delta_r0=None, max_t=None, method='dynamic', **kwargs):
        if method == 'theory':
            assert not callable(delta_h) and delta_r0 is None
            assert delta_h.shape[-self.ndim:] == self.shape
            input_shape = delta_h.shape # (*,*shape)
            delta_r = torch.einsum('ij,...j->...i',self.response_matrix(), delta_h.reshape(-1,self.N)) # (prod(*),N)
            return delta_r.reshape(input_shape), np.inf
        else:
            return super().steady_state(delta_h, delta_r0, max_t, method=method, **kwargs)
        
class PerturbedMultiCellSSNModel(MultiCellSSNModel):
    def __init__(self, W, r_star, w_dims, power, **kwargs):
        assert torch.all(r_star >= 0)
        super().__init__(W, w_dims=w_dims, power=power, **kwargs)
        assert r_star.shape == self.F_shape
        self._r_star = r_star  # (*self.F_shape), i.e. (n)
        
    @property
    def r_star(self):
        device = self._r_star.device
        return torch.einsum('i,...->i...', self._r_star, torch.ones(self.B_shape, device=device)) # (*self.shape)
        
    @property
    def h_star(self):
        r_star = self.r_star.reshape(-1)
        return (r_star**0.5 - self.W @ r_star).reshape(self.shape) # (*self.shape)
    
    def forward(self, delta_h, delta_r0, t, **kwargs):
        if not callable(delta_h):
            h = self.h_star + delta_h # (*,*self.shape)
        else:
            h = lambda t: self.h_star + delta_h(t) # (*,*self.shape)
            
        r0 = self.r_star + delta_r0 # (**,*self.shape)
            
        r, t = super().forward(h, r0, t, **kwargs) # (***,*self.shape)
        
        delta_r = r - self.r_star
        
        return delta_r, t
    