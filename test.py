
import numpy as np

x=[1,2,3,4,5,6]
y=[1,2,3,4,5,6]
f=np.matrix([[1,1/2,1/3,1/4,1/5,1/6],[1/2,1/6,1/12,1/20,1/30,1/42],[1/3,1/12,1/30,1/60,1/105,1/168],[1/4,1/20,1/60,1/140,1/280,1/504],[1/5,1/30,1/105,1/280,1/630,1/1260],[1/6,1/42,1/168,1/504,1/1260,1/2772]])
def vandermonde2d(x, y):
    """
    Computes the Vandermonde matrix.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    shape = (len(x), len(y))
    xy = np.broadcast_arrays(x[:, None], y[None, :])[0]
    return np.power(xy, range(len(xy.T)))


print(np.linalg.solve(vandermonde2d(x, y), f))
