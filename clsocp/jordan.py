import numpy as np

def jordan_identity(kvec):
    temp = []
    for val in kvec:
        if val > 1:
            temp.append(np.append(1, np.repeat(0.0, val - 1)))
        else:
            temp.append(np.array([1]))
    return np.concatenate(temp)

def jordan_square(x):
    out = lambda1(x)**2 * u1(x) + lambda2(x)**2 * u2(x)
    return out

def jordan_sqrt(x):
    out = np.sqrt(lambda1(x)) * u1(x) + np.sqrt(lambda2(x)) * u2(x)
    return out

def jordan_sym(x):
    n = len(x)
    out = np.zeros((n, n))
    out[0, :] = x
    out[:, 0] = x
    np.fill_diagonal(out, x[0])
    return out

def lambda1(x):
    out = x[0] - np.linalg.norm(x[1:])
    return out

def lambda2(x):
    out = x[0] + np.linalg.norm(x[1:])
    return out 

def u1(x):
    nrm = np.linalg.norm(x[1:])
    n = len(x)
    if n > 1:
        if (nrm < 1e-12):
            out = np.append(.5, -.5 * np.repeat(np.sqrt(1/(n - 1)), n - 1))
        else:
            out = np.append(.5, -.5/nrm * x[1:])
    else: 
        out = np.array([.5])
    return out

def u2(x):
    nrm = np.linalg.norm(x[1:])
    n = len(x)
    if n > 1:
        if (nrm < 1e-12):
            out = np.append(.5, .5 * np.repeat(np.sqrt(1/(n - 1)), n - 1))
        else:
            out = np.append(.5, .5/nrm * x[1:])
    else:
        out = np.array([.5])
    return out
