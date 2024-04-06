from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

def getK(X, gamma):
    return rbf_kernel(X, gamma=gamma)

def get_KMM(X, gamma):
    N = len(X)
    if N <= 10000:
        return np.mean(getK(X, gamma))
    else:
        s = 0.0
        step = 10000 ** 2 // N
        start = 0
        while start < N:
            s += np.sum(rbf_kernel(X[start:(start+step)], X, gamma=gamma))
            start = min(start + step, N)
        return s / (N ** 2)
    #return np.mean(rbf_kernel(X, gamma=gamma))

def getKMrM(X, Y, gamma):
    cKM = rbf_kernel(X, Y, gamma=gamma)
    M, N = len(X), len(Y)
    if N * M <= 10000 ** 2:
        return np.mean(cKM, axis=1)
    else:
        sums = np.zeros(M)
        step = 10000 ** 2 // M
        start = 0
        while start < N:
            sums += np.sum(rbf_kernel(X, Y[start:(start+step)], gamma=gamma), axis=1)
            start = min(start + step, N)
        return sums / N