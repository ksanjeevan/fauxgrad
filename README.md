# *fauxgrad*

<p align="center">
  <img src="https://user-images.githubusercontent.com/12011058/132263990-4463a85e-a2ef-4b18-b1fb-e9f4ffc831b1.png" width="550px"/>
</p>

<p align="center">
<img src="https://github.com/ksanjeevan/fauxgrad/actions/workflows/unit.yaml/badge.svg" />

<a href="https://colab.research.google.com/github/ksanjeevan/fauxgrad/blob/master/fauxgrad_walkthrough.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</p>

-----------------------------------------

There's plenty of excellent ([tinygrad](https://github.com/geohot/tinygrad)) and minimalist ([micrograd](https://github.com/karpathy/micrograd)) built-from-scratch, deep learning frameworks out there, so the goal of `fauxgrad` is to sacrifice some of the full functionality, and focus on the general idea and building blocks for writing your own.

The walkthrough/tutorial can be found in this [notebook](https://colab.research.google.com/github/ksanjeevan/fauxgrad/blob/master/fauxgrad_walkthrough.ipynb).

### Installation

```
pip install fauxgrad
```

### Examples
Calculating some gradients:

```python
from fauxgrad import Value
a = Value(2.3)
b = Value(-1)
c = (-a * b).log()
l = -(c.sigmoid() + b) + a
l.backward()
print('The derivative that we computed before, dl/da:', a.grad)
>>> 0.91
```

Plotting the backward pass graph:


```python
from fauxgrad.utils import plot_graph
plot_graph(l) # green node is l, light blue nodes have no parents
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/12011058/133201721-0825f8a2-819e-42b8-be6d-ed91e0007523.png" width="850px"/>
</p>

