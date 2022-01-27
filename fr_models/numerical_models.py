import abc

from scipy.integrate import solve_ivp
import numpy as np
import torch

class NumericalModel(abc.ABC, torch.nn.Module):
    def __init__(self, W, tau=1.0):
        assert W.ndim % 2 == 0
        super().__init__()
        
        self.shape = W.shape[:W.ndim // 2]
        self.N_neurons = np.prod(self.shape)
        self.tau = tau
        
        self.W = torch.nn.Parameter(W.reshape(self.N_neurons, self.N_neurons))
        
    @abc.abstractmethod
    def _drdt(self, t, r, h):
        return
        
    def simulate(self, T, h, r0, method='RK45', get_t=False, max_steps=1000):
        r0 = r0.reshape(-1)
        flat_h = lambda t: h(t).reshape(-1)
        
        result = solve_ivp(self._drdt, (0,T), r0, method=method, args=(flat_h,), max_step=max_steps)

        if result.status == -1:
            raise RuntimeError(f"Simulation failed: {result.message}")
        elif result.status == 1:
            raise RuntimeError(f"Simulation terminated: {result.message}")
            
        t = result.t
        r = result.y.T.reshape(-1, *self.shape)
        if get_t:
            return t, r
        return r
    
    def sim_ss_resp(self, h, T=100, threshold=1.0e-4, max_steps=np.inf):
        T = self.tau*T
        
        h = h.reshape(-1)
        h_func = lambda t: h
        func = lambda x: self._drdt(0, x, h_func)
        
        r0 = torch.zeros(self.N_neurons)
        r = self.simulate(T, h_func, r0, max_steps=max_steps)

        ss_resp = r[-1]
    
        error = (func(ss_resp.reshape(-1))**2).mean()**0.5
        if error < threshold:
            return ss_resp
        
        raise RuntimeError(f"Steady state not reached after T = {T}tau, error = {error}. Try increasing T.")
    
class MultiCellFRModel(NumericalModel):
    """
    A multi cell type firing rate model with nonlinearity.
    """
    def __init__(self, W, f, **kwargs):
        assert W.ndim >= 4
        super().__init__(W, **kwargs)
        self.f = f
        self.dim = W.ndim // 2 - 1
        self.nct_shape = self.shape[1:] # nct stands for non-cell-type
        self.n = self.shape[0]
        self.N = np.prod(self.nct_shape)
        
    def _drdt(self, t, r, h):
        return self.f(self.W @ r + h(t)) - r
    
    def linearize(self, f_prime):
        assert len(f_prime) == self.n
        f_prime_expanded = np.tensordot(f_prime, np.ones(self.nct_shape), axes=0).reshape(-1)
        f = lambda x: f_prime_expanded * x
        return FRModel(self.W.reshape((*self.shape, *self.shape)), f, self.tau)

class MultiCellSSNModel(MultiCellFRModel):
    """
    A multi cell type SSN model with rectified squared nonlinearity.
    """
    @staticmethod
    def is_valid_r_star(r_star):
        return np.all(r_star >= 0)
    
    def __init__(self, W, **kwargs):
        f = lambda x: np.clip(x,0,None)**2
        super().__init__(W, f, **kwargs)

    def taylor_expand(self, r_star, order, clip=False):
        assert SSN.is_valid_r_star(r_star)
        
        f_prime = 2*r_star**0.5
        if order == 1:
            return self.linearize(f_prime)
        
        elif order == 2:
            f_prime_expanded = np.tensordot(f_prime, np.ones(self.nct_shape), axes=0).reshape(-1)
            r_star_expanded = np.tensordot(r_star, np.ones(self.nct_shape), axes=0).reshape(-1)
            
            if clip:
                # this will be equivalent to the exact model
                def f(x):
                    x = np.clip(x, -r_star_expanded**0.5, None)
                    return f_prime_expanded*x + x**2
            else:
                f = lambda x: f_prime_expanded*x + x**2
                
            return FRModel(self.W.reshape((*self.shape, *self.shape)), f, self.tau)