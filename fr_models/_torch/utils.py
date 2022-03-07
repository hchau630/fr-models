import torch

class TensorStruct:
    def __init__(self):
        self.__tensors = {}
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, torch.Tensor):
            if hasattr(self, '_TensorStruct__tensors'):
                self.__tensors[name] = value
            else:
                raise RuntimeError("TensorStruct must be initalized before setting tensor")
                
    def __delattr__(self, name):
        if hasattr(self, '_TensorStruct__tensors') and name in self.__tensors:
            del self.__tensors[name]
            
    def to(self, *args, **kwargs):
        for name, tensor in self.__tensors.items():
            setattr(self, name, tensor.to(*args, **kwargs))
            
    def cpu(self, *args, **kwargs):
        for name, tensor in self.__tensors.items():
            setattr(self, name, tensor.cpu(*args, **kwargs))
