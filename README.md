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
a = Value(5)
b = Value(-3)
c = a * b
d = a + c
e = d * 2
e.backward()

print(f'The derivative that we computed before, de/da:', a.grad)
>>> -4.0
```

Plotting the backward pass graph:


```python
from fauxgrad.utils import plot_graph
plot_graph(e)
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/12011058/133004903-d165a145-a6e7-4cc6-91fe-8facab302345.png" width="450px"/>
</p>


