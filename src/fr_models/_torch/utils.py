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
            
class TensorList(list):
    def to(self, *args, **kwargs):
        for i, x in enumerate(self):
            self[i] = x.to(*args, **kwargs)
        return self
    
    def cpu(self, *args, **kwargs):
        for i, x in enumerate(self):
            self[i] = x.cpu(*args, **kwargs)
        return self
    
    def numpy(self, *args, **kwargs):
        for i, x in enumerate(self):
            self[i] = x.numpy(*args, **kwargs)
        return self
    
    def tolist(self, *args, **kwargs):
        for i, x in enumerate(self):
            self[i] = x.tolist(*args, **kwargs)
        return self
    
    def __repr__(self):
        return f'{self.__class__.__name__}({super().__repr__()})'

class TensorDict(dict):
    def to(self, *args, **kwargs):
        for k, v in self.items():
            self[k] = v.to(*args, **kwargs)
        return self
    
    def cpu(self, *args, **kwargs):
        for k, v in self.items():
            self[k] = v.cpu(*args, **kwargs)
        return self
    
    def numpy(self, *args, **kwargs):
        for k, v in self.items():
            self[k] = v.numpy(*args, **kwargs)
        return self
    
    def tolist(self, *args, **kwargs):
        for k, v in self.items():
            self[k] = v.tolist(*args, **kwargs)
        return self
    
    def __repr__(self):
        return f'{self.__class__.__name__}({super().__repr__()})'