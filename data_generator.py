import numpy as np


def laplacian_noise(loc, sigma, shape=None):
    return np.random.laplace(loc, sigma, shape)

def gaussian_noise(N, L2_sensitivity, epsilon, delta):
    variance = 2*np.log(1.25 / delta) * L2_sensitivity**2 / epsilon**2
    mean = 0
    gaussian_noise = np.random.normal(mean, variance, (N,1))
    return gaussian_noise


def exponential_mechanism(N, L2_sensitivity, epsilon, delta):
    return "none"

    
# See Wassermann&Zhou paper chapter 4.
def WZ_Lap_noise(epsilon, amt, dim=1):
    return laplacian_noise(0, 8/epsilon**2, (amt, dim))


def get_superregular_rw(n, noise=[]):
    if len(noise)==0 and n>0:
        noise = laplacian_noise(0,np.log(n), (n,))
    Z = eval_psis(n, noise)
    res = []
    summe = 0
    for i in range(0, n):
        summe += Z[i]
        res.append(summe)
    return np.array(res)
        

def eval_psis(n, noise):
    #ts = np.arange(1,n+1,0)
    base_vals = np.ones(n)
    DW_n = noise[0] * 1/n * base_vals
    DW_n = DW_n + noise[1] * 1/n * base_vals
    if n == 1: return DW_n
    maxpow = int(np.ceil(np.log2(n)))
    for j in range(2,maxpow):
        two_to_j = 2**j
        psi_j_of_t = base_vals*two_to_j/(2*n)
        interval_length = np.ceil(n/two_to_j).astype(int)
        for i in range(2*two_to_j):
            index_from = i*interval_length
            index_until = (i+1)*interval_length
            sign = 1
            if i%2 != 0:
                sign = -1
            psi_j_of_t[index_from:index_until] = psi_j_of_t[index_from:index_until] * (sign)
        
        psi_j_of_t = psi_j_of_t * noise[j]
        DW_n += psi_j_of_t
    return DW_n


def plot_haar_basis_functions(j, n, ts):
    j = 2**np.ceil(np.log2(j))
    print(j)
    j = int(j)
    psi_j = np.ones(len(ts))
    if j == 1:
        return psi_j*1/n
    elif j == 2:
        half = np.ceil(len(ts)/2).astype(int)
        psi_j[0:half] = psi_j[0:half]*1/n
        psi_j[half:] = -psi_j[half:]*1/n
        return psi_j
    
    psi_j =   psi_j * j/(2*n)
    interval_length = np.ceil(len(ts)/j).astype(int)
    for i in range(j):
        index_from = i*interval_length
        index_until = (i+1)*interval_length
        sign = 1
        if i%2 != 0:
            sign = -1
        else:
            psi_j[index_from] = np.NaN
            index_from += 1
        psi_j[index_from:index_until] = psi_j[index_from:index_until] * (sign)
        #        index_from = index_until
    return psi_j


def schoel_balog_tostik_sample(amt, dim=0):
    #np.random.multivariate_normal
    C = 10
    D = 1
    var_formeans = 200
    var_fordata = 30
    mu_means = 100.0 * np.ones(D)
    mus = np.random.normal(loc=mu_means, scale=var_formeans, size=(C, D))
    ws = np.array([1.0 / n for n in range(1, C+1)])
    ws /= np.sum(ws)
    memberships = np.random.choice(range(C), size=amt, p=ws)
    X_private = []
    for n in range(amt):
        mu = mus[memberships[n]]
        sample = np.random.normal(loc=mu, scale=var_fordata)
        X_private.append(sample)
    X_private = np.array(X_private)
    return X_private


def binary_binomial_sample(amt, dim=0):
    p = np.random.uniform(0,1,(1,))
    return np.random.binomial(1, p, (amt, dim))