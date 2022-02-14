import abc
import enum

import torch

class Types(enum.Enum):
    EQ = 'eq' # return value from constraint.forward must be = 0
    INEQ = 'ineq' # return value from constraint.forward must be >= 0

class Constraint(abc.ABC, torch.nn.Module):
    @property
    @abc.abstractmethod
    def type(self):
        pass
    
    @abc.abstractmethod
    def forward(self, r_model):
        pass

class SpectralRadiusCon(Constraint):
    def __init__(self, max_spectral_radius=0.99, **kwargs):
        super().__init__()
        self.max_spectral_radius = 0.99
        self.kwargs = kwargs
        
    @property
    def type(self):
        return Types.INEQ
        
    def forward(self, r_model):
        return self.max_spectral_radius - r_model.a_model.spectral_radius(r_model.r_star, **self.kwargs)

class StabilityCon(Constraint):
    def __init__(self, max_instability=0.99, **kwargs):
        super().__init__()
        self.max_instability = 0.99
        self.kwargs = kwargs
        
    @property
    def type(self):
        return Types.INEQ
        
    def forward(self, r_model):
        # result1 = self.max_instability - r_model.a_model.instability(r_model.r_star, **self.kwargs)
        lp_model = r_model.a_model.numerical_model(r_model.grid).linear_perturbed_model(r_model.r_star, share_mem=True)
        assert lp_model.W.is_cuda
        result2 = (self.max_instability - lp_model.instability()).item()
        # print(result1, result2)
        return result2
    
class ParadoxicalCon(Constraint):
    def __init__(self, cell_type, min_subcircuit_instability=1.01, **kwargs):
        super().__init__()
        self.cell_type = cell_type
        self.min_subcircuit_instability = min_subcircuit_instability
        self.kwargs = kwargs
        
    @property
    def type(self):
        return Types.INEQ
        
    def forward(self, r_model):
        # assert r_model.a_model.n == 2
        # result1 = (r_model.a_model.W[0,0]*2*r_model.r_star[0]**0.5).item() - self.min_subcircuit_instability
        subcircuit_cell_types = [i for i in range(r_model.a_model.n) if i != self.cell_type]
        subcircuit_model = r_model.a_model.sub_model(subcircuit_cell_types)
        subcircuit_r_star = r_model.r_star[torch.tensor(subcircuit_cell_types)]
        # result2 = subcircuit_model.instability(subcircuit_r_star, **self.kwargs) - self.min_subcircuit_instability
        sub_lp_model = subcircuit_model.numerical_model(r_model.grid).linear_perturbed_model(subcircuit_r_star, share_mem=True)
        assert sub_lp_model.W.is_cuda
        result3 = (sub_lp_model.instability() - self.min_subcircuit_instability).item()
        # print(result1, result2, result3)
        # assert torch.allclose(torch.tensor([result1]), torch.tensor([result3]), rtol=5.0e-1)
        return result3