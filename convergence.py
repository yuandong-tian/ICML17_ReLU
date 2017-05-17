import numpy as np
import math
from utils import compute_gradient
import matplotlib.pylab as plt

d = 100
n_node = 50
nIter = 10000

ws_star = np.random.randn(d, n_node)
#ws_star[:n_node,:n_node] = np.eye(n_node)
ws_star_norm = np.linalg.norm(ws_star, axis=0)

ratio = 0.5
ws0 = np.copy(ws_star) + np.random.randn(d, n_node) * ratio

nIter = 1000

ws = np.copy(ws0)
ws_all = np.zeros((d, n_node, nIter))

eta = 0.01

for t in range(nIter):
    ws_all[:,:,t] = ws
    grad = compute_gradient(ws, ws_star)
    ws += eta * grad

errs_per_w = np.sum(np.power(ws_all - ws_star[:,:,None], 2), axis=0)
errs = np.sum(errs_per_w, axis=0)
plt.plot(errs_per_w.T)
plt.show()
plt.plot(errs)
plt.show()
print(errs[0])
print(errs[-1])

