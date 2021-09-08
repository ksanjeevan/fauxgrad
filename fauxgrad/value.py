import random, math

class Value:

  def __init__(self, val, parents=[]):

    self.val = float(val)
    self.parents = parents
    
    self.grad = 0
    self.diff = lambda grad: []

  def __repr__(self):
    return 'Value(%.2f; grad=%.2f)'%(self.val, self.grad)

  def __neg__(self):
    return self * (-1)

  def __add__(self, other):

    other = other if isinstance(other, Value) else Value(other)
    ret = Value(self.val + other.val, parents=[self, other])

    def diff(grad):
      return grad, grad

    ret.diff = diff
    return ret

  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    ret = Value(self.val * other.val, parents=[self, other])

    def diff(grad):
      return grad * other.val, grad * self.val

    ret.diff = diff
    return ret
  
  @staticmethod
  def rand():
    return Value(random.uniform(-1,1))

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

  def relu(self):
    ret = Value(max(0, self.val), parents=[self])

    def diff(grad):
      return [grad * (self.val > 0)]

    ret.diff = diff
    return ret

  def sigmoid(self):
    _sigmoid = lambda x: 1/(1 + math.exp(-x))
    ret = Value(_sigmoid(self.val), parents=[self])

    def diff(grad):
      return [grad * (1-_sigmoid(self.val))*_sigmoid(self.val)]

    ret.diff = diff
    return ret

  def log(self):
    ret = Value(math.log(self.val), parents=[self])

    def diff(grad):
      return [grad * (1/self.val)]

    ret.diff = diff
    return ret

  # Trying to reduce bloat at the expense of usability

  # def __pow__(self, expon):
  #   ret = Value(self.val ** expon, parents=[self])

  #   def diff(grad):
  #     return [grad * expon * self.val ** (expon-1)]

  #   ret.diff = diff
  #   return ret

  # def __sub__(self, other):
  #   return self + (-other)

  # def __rsub__(self, other):
  #   return other + (-self)

  # def __rmul__(self, other):
  #   return self * other

  # def __truediv__(self, other):
  #   return self * other**-1

  # def __rtruediv__(self, other):
  #   return other * self**-1