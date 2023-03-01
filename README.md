# [Dynamic Time Warping](https://en.wikipedia.org/wiki/Dynamic_time_warping) Python Module

[![Build Status](https://travis-ci.org/pierre-rouanet/dtw.svg?branch=master)](https://travis-ci.org/pierre-rouanet/dtw)

Dynamic time warping is used as a similarity measured between temporal sequences. This package provides two implementations:

* the basic version (see [here](https://en.wikipedia.org/wiki/Dynamic_time_warping)) for the algorithm
* an accelerated version which relies on scipy cdist (see https://github.com/pierre-rouanet/dtw/pull/8 for detail)

```python

import numpy as np

# We define two sequences x, y as numpy array
# where y is actually a sub-sequence from x
x = np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
y = np.array([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)

from dtw import dtw

manhattan_distance = lambda x, y: np.abs(x - y)

d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)

print(d)
>>> 2.0 # Only the cost for the insertions is kept

# You can also visualise the accumulated cost and the shortest path
import matplotlib.pyplot as plt

plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.show()

```
Result of the accumulated cost matrix and the shortest path (in white) found:
![Acc cost matrix and shortest path](./acc.png)
