import functools
import copy
import logging

import torch

import utils

logger = logging.getLogger(__name__)

def cache(fun):
    cached_state_dict = None
    cached_value = None
    
    @functools.wraps(fun)
    def wrapper(self):
        nonlocal cached_state_dict
        nonlocal cached_value
        state_dict = self.state_dict()
        if cached_state_dict is not None:
            equal = True
            for _, v1, v2 in utils.itertools.dict_zip(cached_state_dict, state_dict):
                equal = equal and (v1 == v2).all()

            if equal:
                logger.debug(f"Hit in cache for function {fun.__name__}, returning cached function value")
                return cached_value
        
        logger.debug(f"Miss in cache for function {fun.__name__}, computing function value.")
        cached_state_dict = copy.deepcopy(state_dict)
        cached_value = fun(self)
        return cached_value
    
    return wrapper