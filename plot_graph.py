from fauxgrad.test_value import graph1
from fauxgrad import Value
import networkx as nx


G = nx.Graph()
fmt = lambda x: 'val(%.1f;%.1f)'%(x.val, x.grad) 

G = nx.DiGraph()
ret = [graph1(Value)[-1]]
while len(ret) > 0:

  n = ret.pop()

  for p in n.parents:
    if p.grad != 0:
      G.add_edge(str(p)[5:], str(n)[5:])
      ret.append(p)
    print(fmt(p), fmt(n), p.grad)
    G.add_edge(fmt(p), fmt(n), weight=p.grad)
    ret.append(p)


for n1,n2,attr in G.edges(data=True):
    print(n1,n2,attr)

weights = [G[u][v]['weight'] for u,v in G.edges]
maxw, minw = max(weights), min(weights)
weights = [1 + 2*(w-minw)/(maxw - minw) for w in weights]


import matplotlib.pyplot as plt
nx.draw(G, with_labels=True, font_weight='bold')
nx.draw(G, width=weights, 
        with_labels=True, 
        font_weight='bold',
        font_size=10)
plt.show()