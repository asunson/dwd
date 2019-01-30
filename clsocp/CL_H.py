import numpy as np 
from .FB_smoother import FB_smoother

def CL_H(x, s, mu, A, b, inds):
    n = len(inds)
    m = A.shape[0]
    k = A.shape[1]
    out = np.repeat(0.0, m + k + n)
    out[0:m] = (b - np.array([np.matmul(A, x)]).T).flatten()
    out[m:m + k] = FB_smoother(x, mu, s, inds)
    out[m + k: m + k + n] = mu
    return out

