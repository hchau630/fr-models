class NumericalModelError(Exception):
    pass

class SteadyStateNotReached(NumericalModelError):
    pass

class ToleranceTooLarge(NumericalModelError):
    pass

class RequiredStepSizeTooSmall(NumericalModelError):
    pass

class TimeoutError(Exception):
    pass

class IterationStepsExceeded(TimeoutError):
    pass

class OptimizationError(Exception):
    pass