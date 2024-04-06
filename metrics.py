import ot as pot
import numpy as np
import numpy.linalg as linalg
import scipy.stats as stats

def W1(sample1, sample2, w1=[], w2=[]):
    mu_1 = sample1/sum(sample1)
    mu_2 = sample2/sum(sample2)
    print("vals")
    print(mu_1)
    print(mu_2)
    print("vs.")
    print(sample1)
    print(sample2)
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

    return pot.wasserstein_1d(u_values=sample1, v_values=sample2, u_weights=w1, v_weights=w2)


def multivW1(sample1, sample2, w1=[], w2=[]):
    mu_1 = sample1/len(sample1)
    mu_2 = sample2/len(sample2)
    c = pot.dist(mu_1, mu_2)
    # Empty list translates to uniform weights
    return pot.emd2([], [], c)

# return the difference on the marginal with the worst performance
def wrt_marginals(test_functions, sample1, sample2):
    n = sample1.shape[0]
    k = sample2.shape[0]
    maxdiff = 0
    for f in test_functions:
        sum1 = 0
        for x_i in sample1:
            sum1 += sum1 + f.dot(x_i)

        sum2 = 0
        for z_i in sample2:
            sum2 += sum2 + f.dot(z_i)
        
        sum1 *= 1/n
        sum2 *= 1/k
        diff = np.abs(sum1 - sum2)
        if diff > maxdiff:
            maxdiff = diff

    return maxdiff



def KS(F1, F2, x):
    diff = np.abs(F1.pdf(x) - F2.pdf(x))
    return max(diff)

def L2(F1, F2, x):
    diff = np.abs(F1.pdf(x) - F2.pdf(x))
    return linalg.norm(diff,ord=2)**2    

def Renyi(mu, nu):
    return 1
