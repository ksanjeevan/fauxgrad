
import micrograd.engine as gt
from fauxgrad.value import Value

import unittest

def close(a, b, tol=1e-4):
  return abs(a-b) < tol

def graph1(ValueClass):
  a = ValueClass(-4.0)
  b = ValueClass(2.0)
  c = a * b
  g = c + 2*c
  g.backward()
  return [a, b, c, g]

def graph2(ValueClass):
  a = ValueClass(-4.0)
  b = ValueClass(2.0)
  c = a + b
  d = a * b + b**3
  c += c + 1
  c += 1 + c + (-a)
  d += d * 2 + (b + a).relu()
  d += 3 * d + (b - a).relu()
  e = c - d
  f = e**2
  g = f / 2.0
  g += 10.0 / f
  g.backward()
  return [a,b,c,d,e,f,g]

def _forward(graph, ValueClass):
  get_data = lambda x:x.data if hasattr(x, 'data') else x.val
  return map(get_data, graph(ValueClass))

def _back(graph, ValueClass):
  return map(lambda x: x.grad, graph(ValueClass))


class TestFaux(unittest.TestCase):

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