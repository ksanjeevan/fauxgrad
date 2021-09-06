
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(value):
  G = nx.DiGraph()
  fmt = lambda x: 'val(%.1f;%.1f)'%(x.val, x.grad) 

  ret = [value]
  while len(ret) > 0:
    n = ret.pop(0)
    for p in n.parents:
      G.add_edge(fmt(p), fmt(n), weight=p.grad)
      ret.append(p)

  weights = [G[u][v]['weight'] for u,v in G.edges]
  maxw, minw = max(weights), min(weights)
  weights = [1 + 2*(w-minw)/(maxw - minw) for w in weights]

  nx.draw(G, width=weights, 
          with_labels=True, 
          font_weight='bold',
          font_size=10)
  plt.show()

def get_example_graph():
  from fauxgrad import Value
  from test.test_value import graph1
  return graph1(Value)[-1]