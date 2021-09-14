# Nothing of use to be seen here!

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .value import Value

def plot_graph(value):
  # ugly...
  G = nx.DiGraph()
  fmt = lambda x: '%.1f\ngrad=%.1f'%(x.val, x.grad) 

  v_p = [(v, len(v.parents)) for v in value._rev_topo_sort()]
  nc = {fmt(v):"#918AEF" if np>0 else "#D6D4F2" for v, np in v_p}
  nc[fmt(v_p[0][0])] = "#18A32F"

  ret = [value]
  while len(ret) > 0:
    n = ret.pop(0)
    for p in n.parents:
      G.add_edge(fmt(n),fmt(p), weight=p.grad)
      ret.append(p)

  weights = [abs(G[u][v]['weight']) for u,v in G.edges]
  maxw, minw = max(weights), min(weights)
  weights = [1 + 2*(w-minw)/(maxw - minw) for w in weights]

  try:
    pos = nx.drawing.nx_agraph.graphviz_layout(G)
  except:
    print("`pygraphviz` is not installed. Using `random` layout")
    pos = None

  fig = plt.figure(figsize=(10,6))
  ax = fig.add_subplot(111)
  cmap = plt.cm.coolwarm
  
  nx.draw(G, pos=pos,
          width=3.0, 
          with_labels=True, 
          font_weight='bold',
          font_size=10,
          edge_color=weights,
          edge_cmap=cmap,
          alpha=0.7,
          node_color=[nc[n] for n in G.nodes()],
          ax=ax)

  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=minw, vmax=maxw))
  sm.set_array([])
  cbar = fig.colorbar(sm, fraction=0.1, pad=0.0, ticks=[])
  plt.show()


def get_example_graph():
  from test.test_value import graph1
  return graph1(Value)[-1]

# Generate toy 2D data of a donut inside another donut
def generate_circles(num_samples):
  from sklearn.datasets import make_circles
  X, Y = make_circles(n_samples=num_samples, factor=.4, noise=.11)
  return X, Y.reshape(-1,1)

# Visualize the contour plot of the neural net's predictions
def visualize_kamada_kawai(X, Y, m, title=None):
  resolution = 100
  xmin, xmax = X[:,0].min(), X[:,0].max()
  ymin, ymax = X[:,1].min(), X[:,1].max()

  x = np.linspace(xmin, xmax, resolution)
  y = np.linspace(ymin, ymax, resolution)
  xx, yy = np.meshgrid(x, y)

  d = np.stack([xx, yy], axis=2).reshape(-1,2)
  
  z = np.array([m([Value(d[0]), Value(d[1])])[0].val for d in d]).reshape(resolution, resolution)
  
  #plt.contourf(x,y,z, alpha=0.4, cmap='RdYlBu')
  #plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y), cmap='bwr_r', s=30, alpha=0.6)
  plt.contourf(x,y,z, alpha=0.4, cmap='plasma')
  plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y), cmap='binary', s=30, alpha=0.6)

  ax = plt.gca()
  ax.set_facecolor((0.95, 0.95, 0.95))
  if title is not None: ax.set_title(title)
  plt.axis('off')
  plt.show()