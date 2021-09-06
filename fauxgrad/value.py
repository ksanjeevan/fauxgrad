"""
A much, much worse autodiff engine written after reading
about micrograd and tinygrad ¯\\_(ツ)_//¯
"""

class Value:

  def __init__(self, val, parents=[]):

    self.val = val
    self.parents = parents
    
    self.grad = 0; self.diff = lambda grad: []

  def __repr__(self):
    return 'Value(%.2f; grad=%.2f)'%(self.val, self.grad)

  def _back(self):
    def traverse(node):
      if node in visit: return
      [traverse(p) for p in node.parents]
      visit.add(node); ret.append(node) # is this right?

    self.grad = 1.0 # implicit gradient creation
    visit = set(); ret = []
    traverse(self)
    return ret

  def backward(self):
    nodes = self._back()
    for node in reversed(nodes):
      gradients = node.diff(node.grad)
      for p, g in zip(node.parents, gradients):
        p.grad += g    

  def __add__(self, other):

    other = other if isinstance(other, Value) else Value(other)
    ret = Value(self.val + other.val, parents=[self, other])

    def diff(grad):
      return grad, grad

    ret.diff = diff
    return ret

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    ret = Value(self.val * other.val, parents=[self, other])

    def diff(grad):
      return grad * other.val, grad * self.val

    ret.diff = diff
    return ret

  def __pow__(self, expon):
    ret = Value(self.val ** expon, parents=[self])

    def diff(grad):
      return [grad * expon * self.val ** (expon-1)]

    ret.diff = diff
    return ret

  def relu(self):
    ret = Value(max(0, self.val), parents=[self])

    def diff(grad):
      return [grad * (self.val > 0)]

    ret.diff = diff
    return ret

  def __neg__(self):
    return self * (-1)
 
  def __radd__(self, other):
    return self + other

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return other + (-self)

  def __rmul__(self, other):
    return self * other

  def __truediv__(self, other):
    return self * other**-1

  def __rtruediv__(self, other):
    return other * self**-1


if __name__ == '__main__':

  from fauxgrad.test_value import graph1, graph2

  g = graph1()[-1]

  import networkx as nx
  fmt = lambda x: 'val(%.1f;%.1f)'%(x.val, x.grad) 

  G = nx.DiGraph()
  ret = [g]
  while len(ret) > 0:
    n = ret.pop()
    for p in n.parents:
      G.add_edge(fmt(p), fmt(n), weight=p.grad)
      ret.append(p)

  weights = [G[u][v]['weight'] for u,v in G.edges]
  maxw, minw = max(weights), min(weights)
  weights = [1 + 2*(w-minw)/(maxw - minw) for w in weights]


  import matplotlib.pyplot as plt
  nx.draw(G, width=weights, 
          with_labels=True, 
          font_weight='bold',
          font_size=10)
  plt.show()


