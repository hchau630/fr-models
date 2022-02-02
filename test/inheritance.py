import torch

class MyBase(object):
    def __init__(self, base):
        super().__init__()
        self.base = base
        print(f'Base initialized! base: {base}')

class A(MyBase, torch.nn.Module):
    def __init__(self, base, a, *args, **kwargs):
        base = base + 10
        super().__init__(base, *args, **kwargs)
        self.a = a
        print(f'A initialized! a: {a}')

class B(MyBase, torch.nn.Module):
    def __init__(self, base, b, *args, **kwargs):
        super().__init__(base, *args, **kwargs)
        print(f'B initialized! b: {b}')

class C(A, B):
    pass

c = C(1,2,3)
print(C.__mro__)