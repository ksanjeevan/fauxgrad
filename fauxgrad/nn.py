
from .value import Value

def dot(a : list, b : list):  
  return sum([x*y for x, y in zip(a, b)])

class Linear:
  def __init__(self, in_dim : int, out_dim : int, activation=None):
    self._out = out_dim

    self.W = [[Value.rand() for _ in range(in_dim)] for _ in range(out_dim)]
    self.b = [Value.rand() for _ in range(out_dim)]

    self.f = {'relu'    : lambda x: x.relu(),
              'sigmoid' : lambda x: x.sigmoid()}.get(activation, lambda x:x)

  def __call__(self, x : list):
    return [self.f( dot(x, self.W[i]) + self.b[i] ) for i in range(self._out)]
      





