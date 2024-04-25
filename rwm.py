import data_generator as data
import numpy as np
import metrics
import scipy.stats as stats
import scipy.optimize as sciopt
import cvxpy as cp
import time
from histogram import Histogram

def asymptotic_rwm(Ns, database, epsilon, dim=1):
    KS_dists = []
    L2_dists = []
    W1_dists = []
    times = []
    TF_dists = []
    for n in Ns:
        if epsilon == 0:
            epsilon = 1
        #epsilon = n
        n_time, KS, L2, W1, TF = rwm(n, database[:n], epsilon, dim=dim)
        KS_dists.append(KS)
        L2_dists.append(L2)
        W1_dists.append(W1)
        times.append(n_time)
        TF_dists.append(TF)

    return times, KS_dists, L2_dists, W1_dists, TF_dists

def hist_rwm(n, x, database, epsilon):
    n = len(database)
    #database = database / np.max(database)
    #database = np.sort(database, axis = 0)
    alpha = epsilon*n
    perturbed_weights = perturb_weights(n, alpha)
    #perturbed_weights = perturbed_weights.reshape((n,))
    #print(perturbed_weights)
    #pmeasure = find_closest_pmeasure_5(database, perturbed_weights)
    pmeasure = find_closest_pmeasure_5(np.sort(database, axis = 0).reshape((n,)), perturbed_weights.reshape((n,)))
    #print(pmeasure)
    #pmeasure = pmeasure.x[n:]
    #print(pmeasure)
    #print(database.shape)
    # get histograms
    hist_of_database = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(n)), density=True))
    hist_of_SD = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(n)), weights=pmeasure))
    return hist_of_database, hist_of_SD

def rwm(n, database, epsilon=1, dim=1):
    start_time = time.time()
    alpha = epsilon*n
    #database = np.sort(database, axis = 0)
    perturbed_weights = perturb_weights(len(database), alpha, dim=dim)
    #uniform_weights = np.full((n,),1/n)
    pmeasure = find_closest_pmeasure_5(np.sort(database, axis = 0).reshape((len(database),)), perturbed_weights.reshape((len(database),)))
    #print("alpha = " + str(alpha))
    #print("pmeasure: ")
    #print(pmeasure)

    # get histograms
    #k = int(np.sqrt(n))
    k = n
    hist_of_database = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(len(database))), density=True))
    hist_of_SD = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(len(database))), weights=pmeasure.reshape((len(database),1)), density=True))
    #kde_of_SD = stats.gaussian_kde(database.reshape((len(database),)), weights=pmeasure.reshape((len(database),)))
    synthetic_data = hist_of_SD.rvs(size=(k,dim))
    stop_time = time.time() - start_time
    # get distances between histograms
    W1 = metrics.multivW1(database, database, w2=pmeasure)
    # INITIALISE HISTOGRAMS TO COMPUTE L2 AND KS DISTANCES and initialise m
    m = int(n**(dim/(2+dim)))
    Hist_DB = Histogram(database, bin_amt=m, dim=dim, delta=0)
    Hist_SD = Histogram(synthetic_data, bin_amt=m, dim=dim, delta=0)
    # COMPUTE L2 DISTANCE
    L2 = metrics.smartL2_hypercube(Hist_DB.probabilities, Hist_SD.probabilities, m)
    # COMPUTE KS DISTANCE
    KS = metrics.smartKS_hypercube(Hist_DB.probabilities, Hist_SD.probabilities, m)
    # TF
    test_functions = data.get_binary_testfunctions_upto(dimension=dim, max_order=False)
    TF = metrics.wrt_marginals(test_functions=test_functions, sample1=database[:n], sample2=synthetic_data, dim=dim)

    return stop_time, KS, L2, W1, TF


def perturb_weights(n, alpha, dim=1):
    rws = []
    for _ in range(dim):
        random_walk = data.get_superregular_rw(n)
        U = random_walk*2/alpha
        rws.append(U)
    
    #print("maximum of rw: " + str(np.max(rws)))
    uniform_weights = np.full((dim, n),1/n)
    perturbed_weights = uniform_weights + rws
    #perturbed_weights = np.clip(perturbed_weights, 0, 1)
    return perturbed_weights


def asymptotic_ub_acc(Ns, epsilon=1):
    Ns = np.array(Ns)
    alpha = epsilon*Ns
    return np.log(epsilon * Ns)**(3/2)/(epsilon*Ns)


def find_closest_pmeasure_5(database, signed_measure, dim=1):
    n = len(database)
    database = database.reshape((n,))
    #signed_measure = signed_measure.reshape((n,))
    newdb = np.zeros(n+1)
    newdb[:n] = database
    newdb[n] = 1
    nu_hat = cp.Variable(n, nonneg=True)
    signed_measure = signed_measure / np.sum(signed_measure)
    
    #print("diff")
    #print(np.diff(newdb, axis=0))
    #print("cumsum")
    #print(np.cumsum(signed_measure))
    
    objective = cp.Minimize(np.diff(newdb, axis=0).reshape((n,)) @ cp.abs((cp.cumsum(nu_hat).reshape((n,)) - cp.cumsum(signed_measure).reshape((n,)))))
    #objective = cp.Minimize(cp.norm(nu_hat.reshape((n,)) - signed_measure.reshape((n,))))
    constraints = [cp.sum(nu_hat) == 1]
    problem = cp.Problem(objective, constraints)
    sol = problem.solve(verbose=False, solver="ECOS")
    #print("solution of minW1:" + str(sol))
    #print(metrics.multivW1(signed_measure, nu_hat.value))
    return nu_hat.value
    '''
    guess = np.array([1/n for _ in range(n)])
    bnds = [(0,1) for _ in range(n)]
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    print("start optimization")
    res = sciopt.minimize(w1, guess, tol=1e-5, bounds=bnds, constraints=cons, method="Nelder-Meas")
    print("stop optimization")
    return res
    '''
    

