import numpy as np
from .jordan import jordan_identity
from .CL_H import CL_H
from .CL_grad_H import CL_grad_H
from .convert_l import convert_l
from .make_inds import make_inds

def socp(A, b, c, kvec, 
         t, 
         use_sparse = True, 
         gamma_fac = .95, 
         delta0 = .75, 
         sigma0 = .25, 
         mu0 = 0.01, 
         zero_tol = .000001, 
         max_iter = 100, 
         min_progress = .00000001):
    
    #A is m x sum(kvec)
    #b is a sum(kvec) vector
    #kvec = c(k1, ..., kn)

    #Set return code
    code = -1
    #Codes are: 
    #0 - Full success, 
    #1 - Singularity or other error occured,
    #2 - Lack of progress,
    #3 - Maximum number of iterations reached

    #Change linear constraints to SOC
    kvec, t = convert_l(kvec, t)

    # if(use_sparse):
    #    A = A.tosparse() #not correct syntax
    #     # convert A to a sparse matrix 

    # Dimensions
    n = len(kvec)
    m = A.shape[0]
    k = A.shape[1]

    # Create the SOC index list
    inds = make_inds(kvec)

    #Step 0
    delta = delta0
    sigma = sigma0
    mu = np.repeat(mu0, n)
    zbar = np.append(np.repeat(0.0, k + m), mu)
    x = jordan_identity(kvec)
    y = np.repeat(0.0, m)
    s = c - np.array([np.matmul(A.T, y)]).T
    H = CL_H(x, s, mu, A, b, inds)
    nrm_H = np.linalg.norm(H)
    oldnrm_H = nrm_H + 10 * min_progress
    gamma = gamma_fac * min(1, 1/nrm_H)
    iteration = 0
    lklast = 0
    
    while True:
        # Step 1
        if (nrm_H < zero_tol):
            code = 0
            print("Solution achieved within tolerance for SOCP")
            break
        elif (oldnrm_H - nrm_H < min_progress):
            code = 2
            print("Minimum progress not achieved for SOCP")
            break
        elif (iteration >= max_iter):
            code = 3
            print("Maximum number of iterations reached for SOCP")
            break
        else:
            rho = gamma * nrm_H * min(1, nrm_H)

        # Step 2
        try:
            delta_z = np.linalg.solve(CL_grad_H(x, y, s, mu, A, inds), rho * zbar - H)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                code = -1
                print("Singularity or other error occured for SOCP. Solution could be inaccurate")
                break
            else:
                raise 

        # Steps 3 and 4
        lk = lklast
        signswitch = False
        newx = x + (delta ** lk) * delta_z[0:k]
        newy = y + (delta ** lk) * delta_z[k:k + m]
        newmu = mu + (delta ** lk) * delta_z[k + m:k + m + n]
        news = c - np.array([np.matmul(A.T, newy)]).T
        newH = CL_H(newx, news, newmu, A, b, inds)
        newnrm_H = np.linalg.norm(newH)
        oldsign = newnrm_H <= (1 - sigma * (1 - gamma * mu0) * (delta ** lk)) * nrm_H

        done = False

        while not done:
            if oldsign:
                if lk == 0:
                    break
                lk = lk - 1
            else:
                lk = lk + 1

            newx = x + (delta ** lk) * delta_z[0:k]
            newy = y + (delta ** lk) * delta_z[k:k + m]
            newmu = mu + (delta ** lk) * delta_z[k + m:k + m + n]
            news = c - np.array([np.matmul(A.T, newy)]).T
            newH = CL_H(newx, news, newmu, A, b, inds)
            newnrm_H = np.linalg.norm(newH)
            sign = newnrm_H <= (1 - sigma * (1 - gamma * mu0) * (delta ** lk)) * nrm_H

            if sign and not oldsign:
                break

            oldsign = sign
        lklast = max(lk - 1, 0)

        x = newx
        y = newy
        mu = newmu
        s = news
        H = newH
        oldnrm_H = nrm_H
        nrm_H = newnrm_H
        iteration = iteration + 1
        print("Iteration ", iteration, " complete. ||H|| = ", nrm_H, ".")
    
    output = {'x': x,
              'y': y, 
              's': s, 
              'obj': sum(np.multiply(c.flatten(), x)),
              'code': code, 
              'mu': mu, 
              'iteration': iteration}
    
    return output