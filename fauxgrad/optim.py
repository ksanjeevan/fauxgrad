from .value import Value
from typing import List

class SGD:
  def __init__(self, parameters : List[Value], lr=1e-3):
    self.lr = lr
    self.parameters = parameters

  def step(self):
    for p in self.parameters:
      p.val = p.val -  self.lr * p.grad

  def zero_grad(self):
    for p in self.parameters:
      p.grad = 0