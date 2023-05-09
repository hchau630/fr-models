## Reference: https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish ##

import errno
import os
import signal
import functools

from . import exceptions

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise exceptions.TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator

def max_calls(max_n_calls):
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wrapper.n_calls += 1
            print(wrapper.n_calls, end=' ')
            if wrapper.n_calls > max_n_calls:
                raise exceptions.TimeoutError(f"Number of calls: {wrapper.n_calls} to the function {func} exceeded maximum number of calls permitted: {max_n_calls}")
                
            return func(*args, **kwargs)
        
        wrapper.n_calls = 0
        return wrapper
    
    return decorator