import abc

import torch

from fr_models import _torch

class PCurve(abc.ABC, _torch.utils.TensorStruct):
    """
    A parametric curve in D-dimesional space
    """
    @property
    @abc.abstractmethod
    def D(self):
        pass
    
# class Hyperplane(_torch.utils.TensorStruct):
#     """
#     Define a M-dimensional hyperplane in a N-dimensional space as the set of all x such that Ax = b.
#     A has shape (N-M,N).
#     b has shape (N-M).
#     """
#     def __init__(self, A, b=None):
#         A = torch.as_tensor(A)
        
#         assert A.ndim == 2
        
#         if b is None:
#             b = torch.zeros(len(A), device=A.device)
#         else:
#             b = b.as_tensor(b)
        
#         assert b.ndim == 1
#         assert len(b) == len(A)
        
#         self.A = A
#         self.b = b
#         self.N = A.shape[1]
#         self.M = self.N - len(A)
    
class PLine(PCurve):
    def __init__(self, w, b=None, normalize=False):
        w = torch.as_tensor(w, dtype=torch.float)
        
        assert w.ndim == 1
        
        if b is None:
            b = torch.zeros(w.shape, device=w.device)
        else:
            b = torch.as_tensor(b, dtype=torch.float)
            
        assert b.shape == w.shape

        super().__init__()
        
        if normalize:
            w = w / w.norm()
        self.w = w
        self.b = b
        
    @property
    def D(self):
        return len(self.w)
        
    def __call__(self, t):
        return torch.outer(t,self.w) + self.b

class DPCurve(_torch.utils.TensorStruct):
    """
    A discrete parametric curve in D-dimensional space
    """
    def __init__(self, t, points):
        assert t.ndim == 1 and points.ndim == 2
        assert len(t) == points.shape[0]
        
        super().__init__()
        
        indices = torch.argsort(t)
        self.t = t[indices]
        self.points = points[tuple([indices,...])]
        self.D = self.points.shape[1]
        
    @classmethod
    def from_pcurve(cls, pcurve, t):
        return cls(t, pcurve(t))
