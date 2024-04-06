import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools
import kernel_functions as kernel
from bisect import bisect
import data_generator as data
import metrics


def eq_kernel(x1, x2, gamma):
    return np.exp(-gamma*np.linalg.norm(x1-x2, ord=2)**2)

def SBT_RKHS_algo(database, public=True, Ms=[100], epsilons=[1], delta=1e-6):
    # read the dimension and sample size
    N, D = np.shape(database)
    public_data = []

    # set gamma in the way defined in the schoelkopf paper
    gamma = 1/(100**2 * D)

    # get kernel mean
    K_mean = kernel.get_KMM(database, gamma=gamma)
    print("got kernel mean")
    #

    # execute algo for different Ms
    # Notice that we have done the previous calculations for the largest M
    # and now only have to select the appropriate subspaces
    dists_by_M = []
    for M in Ms:
        print(M)
        # take a random subsample (wlog the first m points) or create new sample from same space
        if public:
            public_data = database[:M]
        else:
            #random_indices = (np.random.uniform(0,1,(M,1))*N).astype(int)
            #subsample = database[random_indices]
            # note that this is kind of weird.
            # if the database is passed into the function there is
            # no obvious way to reliably know its distribution right here
            public_data = data.schoel_balog_tostik_sample(M,1)

        eKME = kernel.getKMrM(public_data, database, gamma)
        K = kernel.getK(public_data, gamma)
        # get basis from 
        print("computing basis")
        basis, par = compute_basis(K)
        basis = np.array(basis)
        # project eKME onto the base
        projection = basis.dot(eKME)
        # compute length of the subspace relative to the basis
        dims = bisect(par, M)
        projection_M = projection[:dims]
        basis_M = basis[:dims, :M]
        K_M = K[:M, :M]
        eKME_M = eKME[:M]

        dists = []
        W1_dists = []
        for (i,eps) in zip(range(len(epsilons)), epsilons):
            proj, beta, weights, dist = privatize(K_mean, eKME_M, K_M, basis_M, projection_M, N, eps, delta)
            dists.append(dist)
            '''
            print("Kmean:")
            print(K_M)
            print(weights)
            print(eKME)
            print("")
            '''
            #w1 = metrics.W1()
        dists_by_M.append(dists)
    
    return dists_by_M 

def privatize(K_mean, eKME, K, basis, proj, N, epsilon, delta):
    L2_sensitivity = 2/N
    noise = data.gaussian_noise(len(proj), L2_sensitivity, epsilon, delta)
    noise = noise.reshape((len(noise)))
    beta = proj + noise

    # get weights
    weights = beta.dot(basis)

    # Compute RKHS distance
    dist = np.sqrt(weights.dot(K.dot(weights)) - 2*weights.dot(eKME) + K_mean)
    return proj, beta, weights, dist

# Compute the orthonormal basis of H_M
# K contains the kernel functions k(z_m, .) that span H_m
def compute_basis(K, epsilon=1e-6):
    N = len(K)
    A = np.zeros((N,N))
    sqnorm = np.zeros(N)
    K_A = np.zeros((N,N))

    # iterate over functions in K. applying Gram schmidt
    for f in range(N):
        #if f%10 == 0:
            #print("gram-schmidt:"+str(f))
        A[f][f] = 1
        for j in range(f):
            denominator = sqnorm[j]
            if denominator > epsilon:
                numerator = A[f].dot(K_A[j])
                A[f] -= numerator / denominator * A[j]
            
            # project
            K_A[f] = K.dot(A[f])
            # get sq norm
            sqnorm[f] = A[f].dot(K.dot(A[f]))

    # norm vectors to unit length, check for zero values and create final basis
    basis = []
    par = []    # log location of selected basis vectors
    for f in range(N):
        if sqnorm[f] > epsilon:

            A_f = A[f] / np.sqrt(sqnorm[f])
            basis.append(A_f)
            par.append(f+1)
    
    return basis, par



def show(amt=10000, dim=1):
    # decide for which epsilons and deltas the expirement should be done
    epsilons = [1, 0.1, 0.01]
    delta = 1e-6 # see pg.17 of schoelkopf paper
    database = data.schoel_balog_tostik_sample(amt, dim)
    # decide for which sizes of the synthetic database the experiment should be done
    #maxM = int(np.sqrt(amt))
    Ms = [int(np.exp(e)) for e in np.linspace(np.log(10), np.log(400), 100)]
    Ms = [e for e, _ in itertools.groupby(Ms)]
    print(Ms)
    dists = SBT_RKHS_algo(database, public=False, Ms=Ms, epsilons=epsilons, delta=delta)
    dists = np.array(dists).T
    fig=plt.figure(figsize=(7,6))
    for (i,dist_by_eps) in zip(range(len(dists)),dists):
        plt.loglog(Ms, dist_by_eps, label="eps="+str(epsilons[i]))

    plt.legend()
    fig.savefig('./imgs/algos/kme/kme_basic_nopub.png', dpi=300, bbox_inches='tight')
    plt.show()

def asymptotic_kme(Ns, dim=1, epsilon=1):
    # decide for which epsilons and deltas the expirement should be done
    delta = 1e-6 # see pg.17 of schoelkopf paper
    database = data.schoel_balog_tostik_sample(max(Ns), dim)
    # decide for which sizes of the synthetic database the experiment should be done
    all_dists = []
    for n in Ns:
        Ms = [int(0.01*n)]
        dists = SBT_RKHS_algo(database[:n], public=True, Ms=Ms, epsilons=[epsilon], delta=delta)
        dists = np.array(dists).T
        all_dists.append(dists)
    return all_dists
        