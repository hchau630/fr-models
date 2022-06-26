import functools

import torch

from fr_models import gridtools, _torch

def is_circulant(A, shift=1, rtol=1.0e-5, atol=1.0e-8):
    """
    Given A with shape (*,n,n), returns a boolean tensor with shape (*) where each element
    indicates whether the corresponding (n,n) matrix is circulant or not.
    (In the rest of the description below I omit the batch dimensions for clarity)
    Specifically, we check whether or not A[i,j] == A[(i+1)%n,(j+shift)%n] for all i,j.
    Equivalently, we check whether or not A[i,j] == A[(i+shift)%n,(j+1)%n] for all i,j.
    shift=1 is a circulant matrix by typical convention, whereas
    shift=-1 is known as a anti-circulant matrix. The two are equal if the matrix is hermitian.
    """
    assert isinstance(shift, int)
    
    is_square = (A.ndim >= 2) and (A.shape[-1] == A.shape[-2])
    
    if not is_square:
        return False
    
    n = A.shape[-1]
    device = A.device
    
    indices = gridtools.get_grid([n,n], method='arange', device=device) # (n,n,2)
    indices[...,1] = (indices[...,1] + shift * indices[...,0]) % n
    indices = tuple([...,*indices.moveaxis(-1,0)]) # (...,(n,n),(n,n))
    return _torch.isequal(A[indices], dim=-2, rtol=rtol, atol=atol).all(dim=-1) # A[indices] shape: (*,n,n), checks all equal vertically

def is_hermitian(A, rtol=1.0e-5, atol=1.0e-8):
    return torch.isclose(A, A.transpose(-1,-2).conj(), rtol=rtol, atol=atol).all(dim=-1).all(dim=-1)

def is_symmetric(A, rtol=1.0e-5, atol=1.0e-8):
    return torch.isclose(A, A.transpose(-1,-2), rtol=rtol, atol=atol).all(dim=-1).all(dim=-1)

def _eigvalsc(A, dims, hermitian=False, check=False):
    """
    A - (*,n,**,n,***) where the first n is at axis dims[0] and the second n is at axis dims[1] 
    returns - (*,**,***,n)
    """
    assert (isinstance(dims, tuple) or isinstance(dims, list)) and len(dims) == 2 and dims[0] < dims[1]
    
    dims = (dims[0] % A.ndim, dims[1] % A.ndim) # turns into positive indices, so that moving the dims[1] first doesn't affect moving dims[0]
    A = A.moveaxis(dims[1],-1).moveaxis(dims[0],-2)
    
    if check:
        assert is_circulant(A).all()
        if hermitian:
            assert torch.allclose(A, A.transpose(-1,-2).conj())
    
    signal = A[...,0,:] # (*,n)
    n = signal.shape[-1]
    
    if hermitian:
        eigvals = torch.fft.hfft(signal[...,:n//2+1],n=n,dim=-1)
    else:
        eigvals = torch.fft.fft(signal,dim=-1)

    return eigvals # (*,n)

def eigvalsc(A, hermitian=False, check=False):
    """
    Fast eigenvalue computation for circulant matrices (matrices with shift=1, see the function is_circulant).
    If A is hermitian, uses torch.fft.hfft instead of torch.fft.fft.
    By default, assumes A is circulant (and hermitian, if hermitian=True) without checking, unless one sets check=True.
    A has shape (*,n,n), returns shape (*,n)
    """
    return eigvalsnc(A, 1, hermitian=hermitian, check=check, keep_shape=False)

def eigvalsnc(A, ndim, hermitian=False, check=False, keep_shape=False):
    """
    Fast eigenvalues computation for n-circulant matrices.
    A has shape (*,n_1,n_2,...,n_ndim,n_1,n_2,...,n_ndim)
    If keep_shape, returns shape (*,n_1,n_2,...,n_ndim)
    otherwise, returns shape (*,n_1*n_2*...*n_ndim)
    """
    list_dims = [(-ndim-1,-i-1) for i in range(ndim)]
    # the shape of A after each step:
    #                           (*,n_1,n_2,...,n_{ndim-1},n_ndim,n_1,n_2,...,n_{ndim-1},n_ndim)
    # dims=(-ndim-1,-1)      -> (*,n_1,n_2,...,n_{ndim-1},       n_1,n_2,...,n_{ndim-1},n_ndim)
    # dims=(-ndim-1,-2)      -> (*,n_1,n_2,...,                  n_1,n_2,...,n_ndim,n_{ndim-1})
    # ...                    -> (*,n_1,n_2,                      n_1,n_2,n_ndim,n_{ndim-1},...)
    # dims=(-ndim-1,-ndim+1) -> (*,n_1,                          n_1,n_ndim,n_{ndim-1},...,n_2)
    # dims=(-ndim-1,-ndim)   -> (*,                              n_ndim,n_{ndim-1},...,n_2,n_1)
    eigvals = functools.reduce(functools.partial(_eigvalsc, hermitian=hermitian, check=check), list_dims, A)
    
    batch_ndim = eigvals.ndim - ndim
    if keep_shape:
        permute_indices = list(range(batch_ndim)) + list(range(batch_ndim,eigvals.ndim))[::-1]
        return eigvals.permute(permute_indices) # (*,n_1,n_2,...,n_ndim)
    else:
        batch_shape = eigvals.shape[:batch_ndim]
        return eigvals.reshape(*batch_shape,-1) # (*,n_1*n_2*...*n_ndim)

def eigvalsbc(A, hermitian=False, check=False, keep_shape=False):
    """
    Fast eigenvalues computation for block circulant matrices.
    A has shape (*,n,N,n,N), which represents a batch of nxn blocks of NxN circulant matrices.
    If keep_shape, returns shape (*,n,N)
    otherwise, returns shape (*,n*N)
    """
    return eigvalsbnc(A, 1, hermitian=hermitian, check=check, keep_shape=keep_shape)

def eigvalsbnc(A, ndim, hermitian=False, check=False, keep_shape=False, epsilon=1.0e-16):
    """
    Fast eigenvalues computation for block n-circulant matrices.
    A has shape (*,n,N_1,N_2,...,N_ndim,n,N_1,N_2,...,N_ndim)
    If keep_shape, returns shape (*,n,N_1,N_2,...,N_ndim)
    otherwise, returns shape (*,n*N_1*N_2*...*N_ndim)
    
    As an example of the massive speed improvement over torch.linalg.eigvals, below are the
    timing results of computing eigenvalues of W with shape (2,10,20,2,10,20) = (400x400)
    with GPU in a jupyter notebook
    
    >> %timeit torch.linalg.eigvals(W)
    71.8 ms ± 5.22 ms per loop
    
    >> %timeit _torch.linalg.eigvalsbnc(W_expanded, 2)
    702 µs ± 12.5 µs per loop
    
    >> %timeit _torch.linalg.eigvalsbnc(W_expanded, 2, hermitian=True)
    703 µs ± 9.68 µs per loop
    
    which shows a 100x speed increase. 
    
    This difference is even bigger for bigger matrices. For example, this is for matrix W 
    with shape (2,30,40,2,30,40) = (2400,2400)
    
    >> %timeit torch.linalg.eigvals(W)
    1.78 s ± 96.1 ms per loop
    
    >> %timeit _torch.linalg.eigvalsbnc(W_expanded, 2)
    2.67 ms ± 74.4 µs per loop
    
    >> %timeit _torch.linalg.eigvalsbnc(W_expanded, 2, hermitian=True)
    2 ms ± 20.2 µs per loop
    
    which shows a 1000x speed increase.
    """
    assert A.ndim >= ndim*2 + 2
    batch_ndim = A.ndim - (ndim*2 + 2)
    A = A.moveaxis(batch_ndim+1+ndim, batch_ndim+1) # (*,n,n,N_1,N_2,...,N_ndim,N_1,N_2,...,N_ndim)
    eigvals = eigvalsnc(A, ndim, hermitian=hermitian, check=check, keep_shape=keep_shape) # (*,n,n,**)
    eigvals = eigvals.moveaxis(batch_ndim+1,-1).moveaxis(batch_ndim,-2) # (*,**,n,n)
    eigvals = eigvals + torch.diag_embed(torch.normal(0.0, 1.0, eigvals.shape[:-1], device=eigvals.device))*epsilon # prevent 0 eigenvalues, which cause backward pass to fail
    eigvals = torch.linalg.eigvals(eigvals) # (*,**,n)
    
    if keep_shape:
        return eigvals.moveaxis(-1,batch_ndim) # (*,n,N_1,N_2,...,N_ndim)
    else:
        batch_shape = eigvals.shape[:batch_ndim]
        return eigvals.reshape(*batch_shape,-1) # (*,n*N_1*N_2*...*N_ndim)

# def _block_eigvals(A, batch_ndim=0, methods=None, list_kwargs=None):
#     A_shape = A.shape[batch_ndim:]
#     A_ndim = len(A_shape)
    
#     assert A_ndim % 2 == 0
#     N = A_ndim // 2
    
#     assert A_shape[:N] == A_shape[N:]
    
#     if methods is None:
#         methods = [torch.linalg.eigvals]*N
#     else:
#         assert len(methods) == N
    
#     if list_kwargs is None:
#         list_kwargs = [{}]*N
#     else:
#         assert len(list_kwargs) == N
      
#     # a recursive implementation may lead to memory problems, I'll rewrite it in terms of a for-loop if that becomes a problem
#     if N == 1:
#         return methods[0](A, **list_kwargs[0])
#     else:
#         A = A.moveaxis(batch_ndim+N,batch_ndim+1) # (*,n_1, n_1, n_2, ..., n_N, n_2, ..., n_N)
#         eigvals = block_eigvals(A, batch_ndim=batch_ndim+2, methods=methods[1:], list_kwargs=list_kwargs[1:]) # block_eigvals thinks it's processing a tensor with shape (**,n_2,...,n_N,n_2,...,n_N), now it returns (n_N,...,n_2,**) = (n_N,...,n_2,*,n_1,n_1)
#         eigvals = methods[0](eigvals, **list_kwargs[0]) # (n_N,...,n_2,*,n_1)
#         return eigvals.moveaxis(-1,N-1) # (n_N,...,n_2,n_1,*)
    
# def block_eigvals(A, batch_ndim=0, methods=None, list_kwargs=None):
#     """
#     Efficiently compute a batch of matrices with recursive block structure, where all blocks of same size commute with each other.
#     A is a tensor with shape (*, n_1, n_2, ..., n_N, n_1, n_2, ..., n_N)
#     which represents a batch of n_1 x n_1 block matrix of (n_2 x n_2 block matrix of ... (n_N x n_N matrix))
#     methods is a list of functions where function i compute the eigenvalues of the n_i x n_i matrix.
#     Each function must be able to compute eigvalues in batch, i.e. accepts input of the form (*,n,n).
#     If methods=None, uses torch.linalg.eigvals.
#     list_kwargs is a list of kwargs to provide to each of the functions.
    
#     Returns the eigenvalues in the shape (*,n_1,...,n_N)
#     """
#     eigvals = _block_eigvals(A, batch_ndim=batch_ndim, methods=methods, list_kwargs=list_kwargs) # (n_N,...,n_2,n_1,*)
    
#     N = eigvals.ndim - batch_ndim
#     eigvals_shape, batch_shape = eigvals.shape[:N], eigvals.shape[N:] # (n_N,...,n_1), (*)
    
#     eigvals = eigvals.reshape(*eigvals_shape,-1) # (n_N,...,n_1,M)
#     eigvals = eigvals.permute(list(range(N+1))[::-1]) # (M,n_1,...,n_N)
#     eigvals = eigvals.reshape(*batch_shape,*eigvals_shape[::-1]) # (*,n_1,...,n_N)
    
#     return eigvals