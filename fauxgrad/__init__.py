from .value import Value
from .nn import Linear

def cross_entropy(pred, target):
    # this is a bit ugly, as it's at the expense of not implementing __sub__, etc.
    ret = [-p.log() if t == 1 else -(-p+1).log() for p, t in zip(pred, target)]
    return sum(ret)