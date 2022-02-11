class NumericalModelError(Exception):
    pass

class SteadyStateNotReached(NumericalModelError):
    pass