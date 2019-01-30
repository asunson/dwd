import numpy as np
from .clsocp.socp import socp
from sklearn.metrics.pairwise import euclidean_distances

# These functions are adapted from the MATLAB functions written by J.S. Marron
# available at https://genome.unc.edu/pubsup/dwd

# These functions are again adapted from the R functions written by jcrudy
# available at https://github.com/jcrudy/CONOR

def DWD1SM(Xp, Xn, threshfact = 100):
    Np = Xp.shape[1]
    Nn = Xn.shape[1]
    vpwdist2x = euclidean_distances(Xp.T, Xn.T)
    medianpwdist2 = np.median(np.square(vpwdist2x))

    penalty = threshfact / medianpwdist2

    sepelimout = sepelimdwd(Xp, Xn, penalty)
    w = sepelimout['w']
    flag = sepelimout['flag']
    if (flag == -1):
        print("Inaccurate Solution!")
    if (flag == -2):
        print("Infeasible or unbounded optimization problem!")
        
    dirvec = w/easy_norm(w)

    return dirvec

# helper function
def easy_norm(w):
    svs = np.linalg.svd(w.reshape((len(w), 1)), compute_uv = False)
    max_sv = svs[0]
    return max_sv

def sepelimdwd(Xp, Xn, penalty):
    flag = 0

    Dp = Xp.shape[0]
    Np = Xp.shape[1]

    Dn = Xn.shape[0]
    Nn = Xn.shape[1]

    assert (Dn == Dp), "The dimensions are incompatible"
    d = Dp

    XpnY = np.concatenate((Xp, -Xn), axis = 1)
    XpnY11 = XpnY[0, 0]

    n = Np + Nn

    if (d > n):
        Q, RpnY = np.linalg.qr(XpnY)
        dnew = n
    else:
        RpnY = XpnY
        dnew = d

    yp = np.array([[1] for i in range(Np)])
    yn = np.array([[-1] for i in range(Nn)])

    y = np.concatenate((yp, yn)) # should be a column vector
    ym = y[1:n]

    nv = 1 + dnew + 4 * n
    nc = 2 * n

    blk = [['q', np.concatenate( ([dnew + 1], 3 * np.ones(n)) )],
           ['l', n]]

    Avec = []
    A = np.zeros((nc, nv - n))
    col1 = RpnY[:, 0]

    A[0:n - 1, 1:dnew + 1] = (RpnY[:, 1:n] - (col1 * ym).T).T
    A[0:n - 1, list(range(dnew + 4, dnew + 3 * n, 3))] = -1 * np.diag(np.ones(n - 1))
    A[0:n - 1, list(range(dnew + 5, dnew + 1 + 3 * n, 3))] = np.diag(np.ones(n - 1))
    A[0:n - 1, dnew + 1] = ym.T
    A[0:n - 1, dnew + 2] = -1*ym.T
    A[n - 1, 0] = 1
    A[n:n+n, list(range(dnew+3,dnew+2+3*n,3))] = np.diag(np.ones(n))

    Avec.append(A.T)
    Avec.append(np.concatenate(
                    (np.concatenate((-1 * ym, 
                                     np.diag(np.ones(n-1))), axis = 1), 
                     np.zeros((n + 1, n)))))
    b = np.concatenate((np.zeros((n-1, 1)), np.ones((n + 1, 1))))

    C = []
    c = np.zeros((nv - n, 1))
    c[list(range(dnew + 1, dnew + 1 + 3 * n, 3)), 0] = 1
    c[list(range(dnew + 2, dnew + 2 + 3 * n, 3)), 0] = 1

    C.append(c)
    C.append(penalty * np.ones((n,1)))

    # solve SOCP problem
    CL_K  = np.append(blk[0][1], blk[1][1])
    CL_qlen = sum(blk[0][1])
    CL_llen = blk[1][1]
    CL_type = ['q' for i in range(len(blk[0][1]))]
    CL_type.append('l')
    CL_A = np.concatenate((Avec[0].T, Avec[1]), axis = 1)
    CL_c = np.concatenate((C[0], C[1]))

    soln = socp(CL_A, b, CL_c, CL_K, CL_type, gamma_fac = .3, sigma0 = .1)
    
    X1 = soln['x'][0:int(CL_qlen)]
    X2 = soln['x'][int(CL_qlen):int(CL_qlen + CL_llen)]
    lambday = soln['y']

    barw = X1[1:(dnew + 1)]
    if (d > n):
        w = np.matmul(Q, barw)
    else:
        w = barw
    beta = X1[dnew + 1] - X1[dnew + 2] - X2[0] - np.matmul(col1.T, barw)
    normw = easy_norm(w)
    if (normw < 1 - 1e-3):
        print(normw)
    normwm1 = 0
    if (normw > 1 - 1e-3):
        w = w/normw
        normwm1 = easy_norm(w) - 1
        beta = beta/normw
        
    # Compute the minimum of the supposedly postive
    # and the maximum of the supposedly negative residuals 
    # Refine the primal solution and print its objective value

    residp = np.matmul(Xp.T, w) + beta
    residn = np.matmul(Xn.T, w) + beta
    minresidp = min(residp)
    maxresidn = max(residn)
    res = np.matmul(XpnY.T, w) + beta * y.flatten()
    rsc = 1/np.sqrt(penalty)
    xi = -1 * res + rsc
    xi = xi.clip(min = 0)
    totalviolation = sum(xi)
    minresidpmod = min(residp + xi[0:Np])
    maxresidnmod = max(residn - xi[Np:n])
    minxi = min(xi)
    maxxi = max(xi)
    resn = res + xi
    rresn = 1/resn
    primalobj = penalty * sum(xi) + sum(rresn)

    #Compute the dual solution alp and print its objective value.

    alp = np.zeros((n, 1))
    lambda1 = lambday[0:(n - 1)]
    alp[0] = -1 * np.matmul(ym.T, lambda1)
    alp[1:n] = lambda1.reshape((len(lambda1), 1))
    alp = alp.clip(min = 0)
    sump = sum(alp[0:Np])
    sumn = sum(alp[Np:n])
    sum2 = (sump + sumn)/2
    alp[0:Np] = (sum2/sump) * alp[0:Np]
    alp[Np:n] = (sum2/sumn) * alp[Np:n]
    maxalp = max(alp)
    if (maxalp > penalty or maxxi > 1e-3):
        alp = (penalty/maxalp) * alp
    minalp = min(alp)
    p = np.matmul(RpnY, alp)
    eta = -1*easy_norm(p)
    gamma = 2*np.sqrt(alp)
    dualobj = eta + sum(gamma)
     
    dualgap = primalobj - dualobj

    if (dualgap > 1e-4):
        flag = -1
        
    output = {'w': w,
              'beta': beta,
              'residp': residp,
              'residn': residn,
               'alp': alp,
              'totalviolation': totalviolation,
              'dualgap': dualgap,
              'flag': flag}
    return output
    