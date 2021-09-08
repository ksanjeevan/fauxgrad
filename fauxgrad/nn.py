from typing import List
from .value import Value

def dot(a : List[Value], b : List[Value]):  
  return sum([x*y for x, y in zip(a, b)])

def flatten(w):
  return [element for vector in w for element in vector]

class Linear:
  def __init__(self, in_dim : int, out_dim : int, activation=None):
    self._out = out_dim

    self.W = [[Value.rand() for _ in range(in_dim)] for _ in range(out_dim)]
    self.b = [Value(0) for _ in range(out_dim)]

    self.f = {'relu'    : lambda x: x.relu(),
              'sigmoid' : lambda x: x.sigmoid()}.get(activation, lambda x:x)

  def __call__(self, x : List[Value]):
    return [self.f( dot(x, self.W[i]) + self.b[i] ) for i in range(self._out)]
      
  def parameters(self):
    return flatten(self.W + [self.b])




