import numpy as np
from socp import socp
penalty = 10

# Xp = np.random.rand(6, 2)
# Xn = np.random.rand(6, 2)

Xp = np.array([[5.489619,  3.543807,  2.624454],
               [13.500492,  9.179315,  7.246465],
               [4.003831,  1.390721, 41.585086],
               [13.363206, 13.554107,  5.448799],
               [3.101640, 15.480200,  3.896443]])
Xn = np.array([[3.731537,  1.0037370,  0.2492515],
               [2.092342,  1.8907049, 11.2378587],
               [0.480463,  0.3494742, 19.6087692],
               [4.266316,  3.2186105, 15.4405558],
               [7.388343, 19.4201739,  7.2613174]])


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
ym = y[1:10]

nv = 1 + dnew + 3 * n + n
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

soln = socp(CL_A, b, CL_c, CL_K, CL_type)