
import unittest, random
import micrograd.engine as gt
from fauxgrad.value import Value

def close(a, b, tol=1e-4):
  return abs(a-b) < tol

def graph1(ValueClass):
  a = ValueClass(-4.0)
  b = ValueClass(2.0)
  c = a * b
  g = c + c * 2
  g.backward()
  return [a, b, c, g]

def graph2(ValueClass):
  a = ValueClass(15.1)
  b = ValueClass(-.093)
  c = ValueClass(-13)

  c += a + b

  d = a * b
  e = (c + d).relu() * (-a) + 100.43
  g = a * 2
  g.backward()
  return [a,b,c,d,e,g]

def _forward(graph, ValueClass):
  get_data = lambda x:x.data if hasattr(x, 'data') else x.val
  return map(get_data, graph(ValueClass))

def _back(graph, ValueClass):
  return map(lambda x: x.grad, graph(ValueClass))


class TestValue(unittest.TestCase):

  def test_act(self):
    self.assertTrue( Value.rand().relu().val >= 0 )
    self.assertTrue( 0 <= Value.rand().sigmoid().val <= 1 )

  def test_add(self):
    self.assertTrue( (Value(3) + 4).val  == 7 )
    self.assertTrue( (Value(-5) + Value(3)).val  == -2 )

  def test_mul(self):

    self.assertTrue( (Value(3) * 4).val  == 12 )
    self.assertTrue( (-Value(3)).val  == -3 )
    self.assertTrue( (Value(-5) * Value(-3)).val  == 15 )

  def test_forward(self):
    for graph in [graph1, graph2]:
      with self.subTest(graph=graph):
        for x, y in zip(_forward(graph, gt.Value), 
                        _forward(graph, Value)):
          self.assertTrue( close(x, y) )

  def test_back(self):
    for graph in [graph1, graph2]:
      with self.subTest(graph=graph):
        for x, y in zip(_back(graph, gt.Value), 
                        _back(graph, Value)):
          self.assertTrue( close(x, y) )


if __name__ == '__main__':
  unittest.main()
  # pytest -v .