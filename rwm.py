import data_generator as data
import numpy as np
import metrics
import scipy.stats as stats

def asymptotic_rwm(Ns, x, database, epsilon):
    KS_dists = []
    L2_dists = []
    W1_dists = []
    for n in Ns:
        KS, L2, W1 = rwm(n, x, database[:n], epsilon)
        KS_dists.append(KS)
        L2_dists.append(L2)
        W1_dists.append(W1)

    return KS_dists, L2_dists, W1_dists

def hist_rwm(n, x, database, epsilon):
    n = len(database)
    alpha = epsilon*n
    perturbed_weights = perturb_weights(n, alpha)
    print(database.shape)
    print(perturbed_weights.shape)
    # get histograms
    hist_of_database = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(n)), density=False))
    hist_of_SD = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(n)), weights=perturbed_weights), density=False)
    return hist_of_database, hist_of_SD

def rwm(n, x, database, epsilon):
    n = len(database)
    alpha = epsilon*n
    perturbed_weights = perturb_weights(n, alpha)
    uniform_weights = np.full((n,),1/n)
    # get histograms
    hist_of_database = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(n)), density=False))
    hist_of_SD = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(n)), weights=perturbed_weights), density=False)
    # get distances between histograms
    KS = metrics.KS(hist_of_database, hist_of_SD, x)
    L2 = metrics.L2(hist_of_database, hist_of_SD, x)
    W1 = metrics.W1(database, database, uniform_weights, perturbed_weights)
    return KS, L2, W1

def perturb_weights(n, alpha):
    random_walk = data.get_superregular_rw(n)
    U = random_walk*2/alpha
    #U = random_walk
    uniform_weights = np.full((n,),1/n)
    perturbed_weights = uniform_weights + U
    return perturbed_weights


def asymptotic_ub_acc(Ns, epsilon=1):
    Ns = np.array(Ns)
    alpha = epsilon*Ns
    return np.log(Ns)**(3/2)/alpha