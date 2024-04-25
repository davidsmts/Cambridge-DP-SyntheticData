# Note that this file contains metrics but also non-metric similarity measurements.

import ot as pot
import numpy as np
import numpy.linalg as linalg
import scipy.stats as stats
import itertools as it


def W1(sample1, sample2, w1=[], w2=[]):
    mu_1 = sample1/np.sum(sample1)
    mu_2 = sample2/np.sum(sample2)
    if len(w1) == 0:
        n = len(sample1)
        w1 = np.full((n,),1/n)
    else:
        # rescale weights to make sure they sum to 1
        # because this is not guaranteed for perturbed weights
        w1 = w1/np.sum(w1)
    if len(w2) == 0:
        n = len(sample2)
        w2 = np.full((n,),1/n)
    else:
        w2 = w2/np.sum(w2)

    return pot.wasserstein_1d(u_values=sample1, v_values=sample2, u_weights=w1, v_weights=w2, require_sort=True)


def multivW1(sample1, sample2, w1=[], w2=[], metric="euclidean"):

    #mu_1 = mu_1.reshape((len(mu_1), 1))
    #mu_2 = mu_2.reshape((len(mu_2), 1))
    if len(w1) == 0:
        w1 = pot.unif(len(sample1))
    else:
        w1 = w1/np.sum(w1)

    if len(w2) == 0:
        w2 = pot.unif(len(sample2))
    else:
        w2 = w2/np.sum(w2)

    c = pot.dist(sample1, sample2, metric=metric, p=1)
    # Empty list translates to uniform weights
    res = pot.emd2(a=w1, b=w2, M=c, numItermax=1e7)
    #print("W1 ends")
    return res

# return the difference on the marginal with the worst performance
def wrt_marginals(test_functions, sample1, sample2, dim=1):
    n = sample1.shape[0]
    k = sample2.shape[0]
    maxdiff = 0
    for f in test_functions:
        sum1 = 0
        for x_i in sample1:
            val = np.product(np.where(f-x_i == 0, np.ones(dim), np.zeros(dim)))*1/n
            sum1 += val

        sum2 = 0
        for z_i in sample2:
            val = np.product(np.where(f-z_i == 0, np.ones(dim), np.zeros(dim)))*1/k
            sum2 += val
        # print("SUMMEN")
        # print(sum1)
        # print(sum2)

        diff = np.abs(sum1 - sum2)
        if diff > maxdiff:
            maxdiff = diff

    return maxdiff


def mean_utility(X, Y):
    return np.linalg.norm(np.mean(X) - np.mean(Y))


def KS(F1, F2, x):
    diff = np.abs(F1.cdf(x) - F2.cdf(x))
    return max(diff)


def smartKS(X, Y, giveEvalPoints=False):
    rv1 = stats.rv_histogram(np.histogram(X, bins=int(np.sqrt(len(X)))))
    rv2 = stats.rv_histogram(np.histogram(Y, bins=int(np.sqrt(len(Y)))))
    upper_boundary = max(max(X), max(Y))
    lower_boundary = min(min(X), min(Y))
    # Evaluate on 100 points per unit length
    points_for_eval_of_KS = np.linspace(lower_boundary, upper_boundary, int(100*(upper_boundary-lower_boundary)))
    if giveEvalPoints:
        return KS(rv1, rv2, points_for_eval_of_KS), points_for_eval_of_KS
    else:     
        return KS(rv1, rv2, points_for_eval_of_KS)


def L2(F1, F2, x, dim=1):
    x1 = np.linspace(0,1,20)
    multidim = x1
    for _ in range(1, dim):
        multidim = np.array(list(it.product(multidim, x1)))
    print(multidim.shape)
    #print(F1.evaluate(x[0]))
    diff = np.abs(F1.pdf(multidim.T) - F2.pdf(multidim.T))
    return linalg.norm(diff,ord=2)**2    


def smartL2_hypercube(probabilities1, probabilities2, bin_amt):
    vol_of_bin = 1/bin_amt
    diff = (probabilities1 - probabilities2)**2
    L2norm = vol_of_bin * np.sum(diff)
    return L2norm


def smartKS_hypercube(probabilities1, probabilities2, bin_amt):
    diff = np.abs(probabilities1 - probabilities2)
    maximum = np.max(diff)
    return maximum


def Renyi(mu, nu):
    return 1
