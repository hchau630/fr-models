import abc
import dataclasses
import typing

import numpy as np

@dataclasses.dataclass
class AnalyticModel(abc.ABC):
    @abc.abstractmethod
    def numerical_model(self):
        pass
    
    def named_parameters(self):
        return dataclasses.asdict(self)
    
@dataclasses.dataclass
class SpatialSSNModel(AnalyticModel):
    W: np.ndarray
    sigma: np.ndarray
    
    def numerical_model(self):
        return 