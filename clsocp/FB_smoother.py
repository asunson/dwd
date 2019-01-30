import numpy as np
from .jordan import jordan_square, jordan_sqrt, jordan_identity

def FB_smoother(x, mu, s, inds):
    k = len(x)
    phi = np.repeat(0.0, k)
    for i in range(len(inds)):
        phi[inds[i]] = (1 + mu[i]) * (x[inds[i]] + s[inds[i]].T) - \
                        jordan_sqrt(jordan_square((x[inds[i]] + mu[i] * s[inds[i]].T).flatten()) + \
                                    jordan_square((mu[i] * x[inds[i]] + s[inds[i]].T).flatten() + \
                                                  2 * mu[i]**2 * jordan_identity([len(inds[i])])))
    return phi

