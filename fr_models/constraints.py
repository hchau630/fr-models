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

# TODO: Modify the forward function to use numerical model's spectral radius
# class SpectralRadiusCon(Constraint):
#     def __init__(self, max_spectral_radius=0.99, **kwargs):
#         super().__init__()
#         self.max_spectral_radius = 0.99
#         self.kwargs = kwargs
        
#     @property
#     def type(self):
#         return Types.INEQ
        
#     def forward(self, r_model):
#         return self.max_spectral_radius - r_model.a_model.spectral_radius(r_model.r_star, **self.kwargs)

class StabilityCon(Constraint):
    def __init__(self, max_instability=0.99, **kwargs):
        super().__init__()
        self.max_instability = 0.99
        self.kwargs = kwargs
        
    @property
    def type(self):
        return Types.INEQ
        
    def forward(self, r_model):
        lp_model = r_model.a_model.numerical_model(r_model.grid).linear_perturbed_model(r_model.r_star, share_mem=True)
        
        assert lp_model.W.is_cuda
        result = (self.max_instability - lp_model.instability(**self.kwargs)).item()

        return result
    
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
        subcircuit_cell_types = [i for i in range(r_model.a_model.n) if i != self.cell_type]
        subcircuit_model = r_model.a_model.sub_model(subcircuit_cell_types)
        subcircuit_r_star = r_model.r_star[torch.tensor(subcircuit_cell_types)]
        sub_lp_model = subcircuit_model.numerical_model(r_model.grid).linear_perturbed_model(subcircuit_r_star, share_mem=True)
        
        assert sub_lp_model.W.is_cuda
        result = (sub_lp_model.instability(**self.kwargs) - self.min_subcircuit_instability).item()
        
        return result