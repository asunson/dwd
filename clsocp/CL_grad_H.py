import numpy as np
from .jordan import jordan_square, jordan_sqrt, jordan_identity, jordan_sym

def CL_grad_H(x, y, s, mu, A, inds):
    m = A.shape[0]
    k = A.shape[1]
    n = len(inds)
    
    out = np.zeros((m+k+n, m+k+n))
    
    out[0:m, 0:k] = -1 * A
    
    N = np.zeros((k, k))
    P = np.zeros((k, n))
    M = np.zeros((k, k))
    
    for i in range(n):
        ki = len(inds[i])
        w1bar = (x[inds[i]] + mu[i] * s[inds[i]].T).flatten()
        w2bar = (mu[i] * x[inds[i]] + s[inds[i]].T).flatten()
        wbar = jordan_sqrt(jordan_square(w1bar) + jordan_square(w2bar) + 2 * mu[i]**2 * jordan_identity([ki]))
        Lwinv = np.linalg.inv(jordan_sym(wbar)) 
        M[np.ix_(inds[i], inds[i])] = (1 + mu[i]) * np.eye(ki, dtype = float) - \
                                        np.matmul(Lwinv, (jordan_sym(w1bar) + mu[i] * jordan_sym(w2bar)))
        N[np.ix_(inds[i], inds[i])] = (1 + mu[i]) * np.eye(ki, dtype = float) - \
                                        np.matmul(Lwinv, (mu[i] * jordan_sym(w1bar) + jordan_sym(w2bar)))
        P[inds[i], i] = x[inds[i]] + s[inds[i]].T.flatten() - \
                            np.matmul(Lwinv, (np.matmul(jordan_sym(w1bar), s[inds[i]]).flatten() + \
                            np.matmul(jordan_sym(w2bar), x[inds[i]]) + \
                            2 * mu[i] * jordan_identity([ki])))
    out[m:m + k, 0:k] = M
    out[m:m + k, k:(k + m)] = -1 * np.matmul(N, A.T)
    out[m:m + k, k + m:k + m + n] = P
    out[m + k:m + k + n, m + k:m + k + n] = np.eye(n, dtype = float)
    
    return out