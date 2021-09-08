
import unittest, random
from fauxgrad import Value, Linear
from fauxgrad.utils import generate_circles
from fauxgrad.optim import SGD

class MLP:
    
    def __init__(self):
        self.l1 = Linear(2, 15, 'relu')
        self.l2 = Linear(15, 1, 'sigmoid')
        
    def __call__(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

def nll_loss(pred, target):
    ret = [-p.log() if t == 1 else -(-p+1).log() for p, t in zip(pred, target)]
    return sum(ret)


class TestValue(unittest.TestCase):

  def test_train(self):
    X, Y = generate_circles(1600)
    data = X.reshape(100, 16, 2) 
    labels = Y.reshape(100, 16)
    mlp = MLP()
    opt = SGD(mlp.parameters(), lr=0.01)

    for epoch in range(5):
      acc = []
      for i in range(100):
          x = [[Value(d[0]), Value(d[1])] for d in data[i]]

          opt.zero_grad()
          yhat = [mlp(datum)[0] for datum in x]

          loss = nll_loss(yhat, labels[i])
          loss.backward()
          
          acc.extend([int(y.val>0.5)==l for y, l in zip(yhat, labels[i])])
          opt.step()

      avg_acc = sum(acc)/len(acc)
      if avg_acc > 0.95:
        self.assertTrue(True)
        break
    else:
      self.assertTrue(False, f'[ACC THRESHOLD NOT MET] {avg_acc}')