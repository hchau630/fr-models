def flatten_seq(s, depth=-1, dtypes=[list, tuple]):
    """
    Recursively flattens a sequence (defined as an instance
    of one of the DTYPES) up to depth DEPTH.
    Will try to cast the result to type(s) if possible, otherwise returns a list.
    depth=0 returns the sequence unchanged, depth=-1 returns a fully flattened sequence
    TypeError is raised if s is not an instance of any of the dtypes.
    
    >>> s = [[1,2,[3]],[],(4,5),6]
    >>> flatten_seq(s, depth=1)
    [1,2,[3],4,5,6]
    >>> flatten_seq(s, depth=-1)
    [1,2,3,4,5,6]
    >>> flatten_seq(tuple(s), depth=-1)
    (1,2,3,4,5,6)
    
    """
    if not any([isinstance(s, dtype) for dtype in dtypes]):
        raise TypeError(f"The arugment to flatten_list must be one of {dtypes}, not {type(s)}")
    flattened_s = []
    for elem in s:
        if depth == 0 or not any([isinstance(elem, dtype) for dtype in dtypes]):
            flattened_s.append(elem)
        else:
            flattened_s += flatten_seq(elem, depth=depth-1)
    try:
        return type(s)(flattened_s)
    except Exception:
        return flattened_s
    
def flatten_dict(d, depth=-1):
    """
    Recursively flattens dict up to depth DEPTH. Keys are concatenated into a tuple key.
    depth=0 returns the dictionary unchanged, depth=-1 returns a fully flattened dictionary
    depth=n means the first n layers of the dictionary is flattened, so a depth k dict becomes a depth k-1 dict.
    TypeError is raised if d is not a dict.
    
    >>> d = {'a': {'b': {'c': 'd'}}}
    >>> flatten_dict(d, depth=1)
    {('a','b'): {'c': 'd'}}
    >>> flatten_dict(d, depth=-1)
    {('a','b','c'):'d'}
    
    """
    if not isinstance(d, dict):
        raise TypeError(f"The arugment to flatten_dict must be a dict, not {type(d)}")
    flattened_dict = {}
    for k, v in d.items():
        if depth == 0 or not isinstance(v, dict) or v == {}: # if depth == 0 or v is leaf
            flattened_dict[k] = v
        else:
            for new_k, new_v in flatten_dict(v, depth=depth-1).items():
                flattened_dict[(k, *new_k)] = new_v
    return flattened_dict