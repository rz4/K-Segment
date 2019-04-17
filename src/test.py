import numpy as np
from k_segment import regress_ksegments

# Params
k = 10
size = 100

if __name__ == '__main__':

  # Data
  x = np.random.rand((size))
  print(x)

  # Fit
  y = regress_ksegments(series=x, weights=np.ones(x.shape), k=k)
  print(y)
