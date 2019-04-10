# K-Segmentation For Sequence Data
Author: Rafael Zamora

## Overview

This project contains a Python implementation of the Bellman k-segmentation
algorithm as described in this [blog](http://homepages.spa.umn.edu/~willmert/science/ksegments/).
The algorithm generates a segmented constant-line fit to a data series, which
is useful in finding unique clusters along sequentially ordered data. A quick
exploration of using this on different kinds of data can be found in the project's
[notebook](notebook/research.ipynb).

## Importing Algorithm
The actual code is very short and can be found under `src/k_segment.py`.

To run k-segmentation on data do:

`test.py`
```python
from k_segment import regress_ksegments

# Params
k = 10
size = 100

if __name__ == '__main__':

  # Data
  x = np.rand((size))
  print(x)

  # Fit
  y = regress_ksegments(series=x, weights=np.ones((size)), k=k)
  print(y)

```
