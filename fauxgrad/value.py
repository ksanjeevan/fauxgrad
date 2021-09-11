import random, math

class Value:

  def __init__(self, val, parents=[]):
    self.val = float(val)
    self.parents = parents
    
    self.grad = 0
    self.diff = lambda grad: []

  def __repr__(self):
    return 'Value(%.2f; grad=%.2f)'%(self.val, self.grad)
  
  @staticmethod
  def rand():
    return Value(random.uniform(-1,1))

  def __neg__(self):
    return self * (-1)

  #----- Essential Operations -----
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    ret = Value(self.val + other.val, parents=[self, other])
    ret.diff = lambda grad: [grad, grad]
    return ret

  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    ret = Value(self.val * other.val, parents=[self, other])
    ret.diff = lambda grad: [grad * other.val, grad * self.val]
    return ret

  def __rmul__(self, other):
    return self * other
  #--------------------------------

  #-------- Backward Pass ---------
  def _rev_topo_sort(self):
    def traverse(node):
      if node in visit: return
      visit.add(node)
      for p in node.parents: traverse(p)
      ret.append(node)

    visit = set(); ret = []
    traverse(self)
    return reversed(ret)

  def backward(self):
    self.grad = 1.0 # implicit gradient creation
    for node in self._rev_topo_sort():
      gradients = node.diff(node.grad)
      for p, g in zip(node.parents, gradients): p.grad += g    
  #--------------------------------

  #------ Activation & Loss Functions ------ 
  def relu(self):
    ret = Value(max(0, self.val), parents=[self])
    ret.diff = lambda grad: [grad * (self.val > 0)]
    return ret

  def sigmoid(self):
    _sigmoid = lambda x: 1/(1 + math.exp(-x))
    ret = Value(_sigmoid(self.val), parents=[self])
    ret.diff = lambda grad: [grad * (1-_sigmoid(self.val))*_sigmoid(self.val)]
    return ret

  def log(self):
    ret = Value(math.log(self.val), parents=[self])
    ret.diff = lambda grad: [grad * (1/self.val)]
    return ret
  #-----------------------------------------