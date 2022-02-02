#  A module containing functions with pytorch backend that behave like numpy functions

import torch

def linspace(start, end, steps, endpoint=True, **kwargs):
    if endpoint:
        return torch.linspace(start, end, steps, **kwargs)
    else:
        return torch.linspace(start, end, steps+1, **kwargs)[:-1] # exclude endpoint
    
def tensor(data, **kwargs):
    """
    Mimics np.array in that this can take in a (nested) sequence of tensors and create a new tensor as such:
    
    >>> from fr_models import utils
    
    >>> A,B,C,D = [torch.ones((2,3))*i for i in range(4)]
    >>> X = utils._torch.tensor([[A,B,B,D],[C,D,A,B]])
    >>> print(X.shape)
    torch.Size([2, 4, 2, 3])
    
    This does not fully mimic np.array, since torch.Tensor does not support dtype=object.
    """
    try:
        return torch.tensor(data, **kwargs)
    except Exception as err:
        # try to recursively create tensors by stacking along the first dimension. 
        # stop recursing if the element is already a tensor.
        return torch.stack([elem if isinstance(elem, torch.Tensor) else tensor(elem) for elem in data], dim=0)
    
def atleast_0d(*args):
    result = tuple([arg if isinstance(arg, torch.Tensor) else torch.tensor(arg) for arg in args])
    return result[0] if len(result) == 1 else result