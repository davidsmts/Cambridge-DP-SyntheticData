import numpy as np
import scipy.stats as stats
import data_generator as data
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import metrics
import cvxpy as cp
import itertools
from histogram import Histogram
import time

# Instead of the optimization problem described in the vershynin stat framework paper. This function minimizes the 
# wasserstein distance over the weighted sets
'''
def minw1(database, Omega_star, sigma):
    n = len(database)
    noise = data.laplacian_noise(0,4,shape=(n,))
    database = database - noise

    def w1(h):
        return wasserstein_distance(database, Omega_star, v_weights=h)

    guess = [1/len(Omega_star) for _ in range(len(Omega_star))]
    bnds = [(0,1) for _ in enumerate(Omega_star)]
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    print("start optimization")
    res = minimize(w1, guess, tol=1e-4, bounds=bnds, constraints=cons)
    print("stop optimization")
    return res
'''
    
# this function makes sure that the weights always sum to 1
def make_one(weights, m):
    dist_from_1 = np.sum(weights)-1
    if np.abs(dist_from_1) > 1e-2:
        print("Weights are too inaccurate!! : " + str(np.abs(dist_from_1)))
        return False
    else:
        random_index = np.random.choice(np.arange(0,m,1))
        print("dumping rest of weight onto index:"+str(random_index))
        for i, weight in enumerate(weights):
            weights[i] = weight - dist_from_1/m
        #weights[int(random_index)] = weights[int(random_index)] - dist_from_1
        return weights


def new_reweight(database, test_functions, Omega_star, noise, dim=1):
    n = len(database)
    m = len(Omega_star)
    # Weight matrix for the objective function.
    # Note that we're only optimizing for the newly introduced variable that upper bounds everything.
    # Thus only that one is 1 and the rest is 0.
    c = np.array([1] + [0 for _ in range(m)])

    # Create inequality constraints
    A_ub = []
    b_ub = []
    for j, f in enumerate(test_functions):
        # construct the matrix for the inequality constraints
        row1_ub = [-1]  # negative sign
        row2_ub = [-1]  # positive sign
        for z_i in Omega_star:
            #print(z_i-f)
            row1_ub.append(float(np.product(np.where(f-z_i==0, np.ones(dim), np.zeros(dim)))))
            row2_ub.append(float(-np.product(np.where(f-z_i==0, np.ones(dim), np.zeros(dim)))))
        A_ub.append(row1_ub)
        A_ub.append(row2_ub)

        # construct the b_ub for the inequality constraints
        b_ub_sum = 0
        for x_i in database:
            b_ub_sum += np.product(np.where(f-x_i, np.ones(dim), np.zeros(dim))/n)
        #print("sum: "+str(b_ub_sum))
        # noise[j] = 0
        b_ub.append(-b_ub_sum-noise[j])
        b_ub.append(b_ub_sum+noise[j])

    # Want to work with numpy arrays
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    # Create equality constraints
    A_eq = np.array([[0] + [1 for _ in range(m)]])
    b_eq = np.array([1])
    #A_eq = np.array([[0] + [0 for _ in range(m)]])
    #b_eq = np.array([0])

    # Constraints for the individual variables:
    individual_bounds = [(0,None)]
    for _ in range(m):
        individual_bounds.append((0,1))

    # solve linear program:
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=individual_bounds)

    return result

def reweight(database, test_functions, Omega_star, noise):
    n = len(database)
    m = len(Omega_star)
    # Weight matrix for the objective function.
    # Note that we're only optimizing for the newly introduced variable that upper bounds everything.
    # Thus only that one is 1 and the rest is 0.
    c = np.array([1] + [0 for _ in range(m)])

    # Create inequality constraints
    A_ub = []
    b_ub = []
    for j, f in enumerate(test_functions):
        # construct the matrix for the inequality constraints
        row1_ub = [-1]  # negative sign
        row2_ub = [-1]  # positive sign
        for z_i in Omega_star:
            row1_ub.append(float(-f.dot(z_i)))
            row2_ub.append(float(f.dot(z_i)))
        A_ub.append(row1_ub)
        A_ub.append(row2_ub)

        # construct the b_ub for the inequality constraints
        b_ub_sum = 0
        for x_i in database:
            b_ub_sum += f.dot(x_i)/n
        print("sum: "+str(b_ub_sum))
        # noise[j] = 0
        b_ub.append(-b_ub_sum-noise[j])
        b_ub.append(b_ub_sum+noise[j])

    # Want to work with numpy arrays
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    # Create equality constraints
    A_eq = np.array([[0] + [1 for _ in range(m)]])
    b_eq = np.array([1])
    #A_eq = np.array([[0] + [0 for _ in range(m)]])
    #b_eq = np.array([0])
    print("shape")
    print(A_ub.shape)
    # Constraints for the individual variables:
    individual_bounds = [(0,None)]
    for _ in range(m):
        individual_bounds.append((0,1))

    # solve linear program:
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=individual_bounds)

    return result


# Draw the synthetic dataset of size k from our subspace \Omega^* according to the reweighted distribution h^*
def bootstrap(h_weights, Omega_star, k):
    #select bin_amt based on sqrt rule
    #bin_amt = int(len(Omega_star))
    #Omega_star = Omega_star.tolist()
    #Omega_star = [0] + Omega_star
    #Y = stats.rv_histogram((h_weights, Omega_star), density=True)
    #Y = stats.rv_histogram(np.histogram(Omega_star), density=True)
    # Try KDE instead of simply using a histogram
    #Y = stats.gaussian_kde(Omega_star, weights=h_weights)
    #return Y.resample(size=int(k))
    #return Y.rvs(size=int(k))
    d = Omega_star.shape[1]
    n = Omega_star.shape[0]
    chosen_indices = np.random.choice([i for i in range(n)], size=k, p=h_weights)
    return Omega_star[chosen_indices]

def subsample(database, m):
    d = 0
    if len(database.shape) > 1:
        d = database.shape[1]
    else:
        d=1
    #Omega_star = np.random.uniform(low=min(database),high=max(database),size=(m, d))
    Omega_star = data.schoel_balog_tostik_sample(m,1)
    #kde = stats.gaussian_kde(database)
    #Omega_star = kde.resample(size=m)[0]
    #Omega_star = Omega_star.reshape(len(Omega_star,))
    #Omega_star = np.random.choice(database, size=m)
    return Omega_star

def subsample_binary(database, m):
    d = database.shape[1]
    Omega_star = np.random.binomial(1, 1/2, size=(m, d))
    return Omega_star


def correct_binary_len(arr, d):
    diff = len(arr) - d
    if diff > 0:
        arr = arr + [0 for _ in range(diff)]
    return arr

# returns marginals up to order 2
def get_testfunctions_binary(d):
    F = []
    # add p-dimensional marginals
    for k in range(-1,d,1):
        for l in range(k+1, d):
            decimal_num = 0
            if k != -1:
                decimal_num += 2**k
            decimal_num += 2**l
            binary_str = np.binary_repr(decimal_num, d)
            binary_num_arr = list(binary_str)
            #binary_num_arr = correct_binary_len(binary_num_arr, d)
            F.append(np.array(binary_num_arr).astype(int))
    return F

#
def get_binary_testfunctions_upto(dimension=1, order=3, max_order=False):
    if max_order:
        order = dimension
    else:
        order = 3
    F = [[1 for _ in range(dimension)], [0 for _ in range(dimension)]]
    for i in range(1, order):
        to_permute = [1 for _ in range(i)] + [0 for _ in range(dimension - i)]
        marginals_of_order_i = list(itertools.permutations(to_permute))
        without_reps = np.unique(marginals_of_order_i, axis=0)
        F = np.concatenate((F, without_reps), axis=0)
        #print("F:")
        #print(to_permute)
    return np.array(F)
    '''
    F = []
    for i in range(1<<dimension):
            s=bin(i)[2:]
            s='0'*(dimension-len(s))+s
            F.append(list((map(int,list(s)))))
    return F
    '''

def get_testfunctions_binary_3(d):
    F = []
    # add p-dimensional marginals
    for k in range(-1,d,1):
        for l in range(k+1, d):
            decimal_num = 0
            if k != -1:
                decimal_num += 2**k
            decimal_num += 2**l
            binary_str = np.binary_repr(decimal_num, d)
            binary_num_arr = list(binary_str)
            #binary_num_arr = correct_binary_len(binary_num_arr, d)
            F.append(np.array(binary_num_arr).astype(int))
    return F

def get_testfunctions(database):
    d = 0
    if len(database.shape)>1:
        d = database.shape[1]
    else:
        d = 1
    F = []
    for k in range(0,d):
        #for l in range(k+1, d):
        decimal_num = 0
        if k != -1:
            decimal_num += 2**k
        #decimal_num += 2**l
        binary_str = np.binary_repr(decimal_num, d)
        binary_num_arr = list(binary_str)
        F.append(np.array(binary_num_arr).astype(int))    
    F.append(np.ones(d))
    return F



def subsampling_mechanism(database, test_functions, delta, gamma, x, k, sigma, m):
    noise = data.laplacian_noise(0, sigma, (len(test_functions), 1))
    
    # drawing uniform from Omega
    Omega_star = subsample(database, m)
    res_of_minproblem = reweight(database, test_functions, Omega_star, noise)
    h_weights = res_of_minproblem.x[1:]
    # Finally get our synthetic data
    y = bootstrap(h_weights, Omega_star, k)
    y = y.reshape((len(y),))
    # Create a histogram of it and plot this against the initial histogram of the database
    hist_of_SD = stats.rv_histogram(np.histogram(y, bins=int(2*np.sqrt(len(y))), density=True))
    hist_of_database = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(len(database))), density=True))
    
    # Calculate distances between the different distributions/histograms
    KS = metrics.KS(hist_of_SD, hist_of_database, x)
    L2 = metrics.L2(hist_of_SD, hist_of_database, x)
    W1 = metrics.multivW1(database, y)
    TF = metrics.wrt_marginals(test_functions, database, y)
    
    return KS, L2, W1, TF

def subsampling_mechanism_bin(database, test_functions, delta, gamma, k, sigma, m, dim=1):
    d = database.shape[1]
    n = database.shape[0]
    start_time = time.time()
    noise = data.laplacian_noise(0, sigma, (len(test_functions), 1))
    
    # drawing uniform from Omega
    Omega_star = subsample_binary(database, m)
    #print("Reweighting.")
    res_of_minproblem = new_reweight(database, test_functions, Omega_star, noise)
    #print("Reweighting done.")
    h_weights = res_of_minproblem.x[1:]

    # Finally get our synthetic data
    #print("Bootstrap")
    y = bootstrap(h_weights, Omega_star, k)
    #print("Bootstrap done")
    stop_time = time.time() - start_time
    #print("Create Hists")
    Hist_DB = Histogram(database, bin_amt=int(np.sqrt(n)), dim=dim, delta=0)
    Hist_SD = Histogram(y, bin_amt=int(np.sqrt(n)), dim=dim, delta=0)
    #print("Create Hists DONE")

    # Calculate distances between the different distributions/histograms
    W1 = metrics.multivW1(database, y)
    L2 = metrics.smartL2_hypercube(Hist_DB.probabilities, Hist_SD.probabilities, m)
    KS = metrics.smartKS_hypercube(Hist_DB.probabilities, Hist_SD.probabilities, m, dim=dim)
    TF = metrics.wrt_marginals(test_functions, database, y, dim=dim)
    return stop_time, KS, L2, W1, TF


def check_guarantees(n, epsilon, delta, gamma, k, F_len, m, K=10):
    lb_kn = delta**(-2) * np.log(F_len / gamma)
    if lb_kn > n or lb_kn > k: 
        print("ERROR - ACCURACY VIOLATION: n,k")
        return False
    lb_m = delta**(-2) * K * F_len / gamma
    if lb_m > m:
        print("ERROR - ACCURACY VIOLATION: m")
        print("m="+str(m)+"; lb = " + str(lb_m))
    min_n = 2*1/(epsilon*delta)*F_len*np.log(F_len/gamma)
    #print(min_n)
    if n < min_n:
        print("ERROR - PRIVACY VIOLATION")
        return False

    return True


def asymptotic_subsampling_bin(Ns, database, epsilon, delta, gamma, dim=1):
    all_W1_dists = []
    all_L2_dists = []
    all_KS_dists = []
    all_TF_dists = []
    all_times = []
    for n in Ns:
        print(n)
        sample = database[:n]

        test_functions = data.get_binary_testfunctions_upto(dim, max_order=False)
        #print(test_functions)
        #print("TF Len: " + str(len(test_functions)))
        #print(test_functions)
        F_len = len(test_functions) 
        delta = np.sqrt(1/n*np.log(F_len/gamma))-1e-4
        #gamma = F_len/n**(delta**2)
        print("delta: "+ str(delta))
        sigma = delta/np.log(F_len/gamma)     # see Thm 2.3
    
        
        # The size of the "much smaller" Omega_star doesn't necessarily need to grow with n
        # But we make it do this to hope and see a convergence.
        K = 10 # this is based in the Renyi divergence. Just a guessed upper bound! TODO: Calculate this
        #m = np.sqrt(n).astype(int)
        m = int(np.ceil(max(delta**(-2)*K*F_len/gamma, np.sqrt(n))))
        # We want the size of the synthetic dataset to grow with n but also to satisfy the guarantees 
        # see Thm 2.3 in vershynin paper, with the assumption that n>k
        #k = (1/4 * m).astype(int)
        k = n   # notice that if k > n, then the synthetic data must contain duplicates
        # Check if the privacy and accuracy guarantees are fulfilled. Otherwise return
        check_guarantees(n, epsilon, delta, gamma, k, F_len, m, K)
        #
        Time, KS, L2, W1, TF = subsampling_mechanism_bin(sample, test_functions, delta, gamma, k, sigma, m, dim=dim)
        all_times.append(Time)
        all_W1_dists.append(W1)
        all_L2_dists.append(L2)
        all_KS_dists.append(KS)
        all_TF_dists.append(TF)

    return all_times, all_KS_dists, all_L2_dists, all_W1_dists, all_TF_dists


def asymptotic_subsampling(Ns, database, epsilon, delta, gamma, x):
    all_KS_dists = []
    all_L1_dists = []
    all_W1_dists = []
    all_TF_dists = []
    for n in Ns:

        sample = database[:n]
        test_functions = get_testfunctions(sample)

        F_len = len(test_functions)
        sigma = delta/np.log(F_len/gamma)     # see Thm 2.3
        
        # The size of the "much smaller" Omega_star doesn't necessarily need to grow with n
        # But we make it do this to hope and see a convergence.
        K = 10 # this is based in the Renyi divergence. Just a guessed upper bound! TODO: Calculate this
        m = max(delta**(-2)*K*F_len/gamma, np.sqrt(n))
        m = np.ceil(m).astype(int)
        m = np.sqrt(n).astype(int)
        m = int(n/2)
        # We want the size of the synthetic dataset to grow with n but also to satisfy the guarantees 
        # see Thm 2.3 in vershynin paper, with the assumption that n>k
        k = max(delta**(-2)*np.log(F_len/gamma), np.sqrt(m))
        k = np.ceil(k).astype(int)
        k = int(m/4)
        # Check if the privacy and accuracy guarantees are fulfilled. Otherwise return
        check_guarantees(n, epsilon, delta, gamma, k, F_len)
        #
        KS, L1, W1, TF = subsampling_mechanism(sample, test_functions, delta, gamma, x, k, sigma, m)
        all_KS_dists.append(KS)
        all_L1_dists.append(L1)
        all_W1_dists.append(W1)
        all_TF_dists.append(TF)

    return all_KS_dists, all_L1_dists, all_W1_dists, all_TF_dists




# deprecated?
# delta should be in (0, 1/4]
# gamma should be in (0, 1/2]
# K should be based on Renyi divergence
# this implementation of the algorithm samples Omega_star from Omega with a std normal
def show_subsampling_mechanism(database, test_functions, delta, gamma):
    n = len(database)
    sigma = delta/np.log(len(test_functions)/gamma)     # see Thm 2.3
    #m = np.sqrt(n).astype(int)
    m = n
    k = n
    #
    noise = data.laplacian_noise(0, sigma, (m, 1))
    noise = noise.reshape((m,))
    #noise = np.zeros((m,1))
    # drawing uniform from Omega
    #Omega_star = subsample(database, m)
    Omega_star = subsample_binary(database, m)
    #res_of_minproblem = cvxopt_reweight(database, test_functions, Omega_star, noise)
    res_of_minproblem = new_reweight(database, test_functions, Omega_star, noise, dim=1)
    print("RESULT:")
    print(res_of_minproblem.fun)
    print(res_of_minproblem.x[1:])
    h_weights = res_of_minproblem.x[1:]
    # Finally get our synthetic data
    y = bootstrap(h_weights, Omega_star, k)
    TF = metrics.wrt_marginals(test_functions, database, y, dim=1)
    print("TF = " + str(TF))

    x = np.linspace(-3/4, 7/4, 300)
    # Create a histogram of it and plot this against the initial histogram of the database
    Y = stats.rv_histogram((np.histogram(y, bins=2, density=True)[0], [-1/2,1/2,3/2]), density=True)
    gmm_hist = stats.rv_histogram((np.histogram(database, bins=2, density=True)[0], [-1/2,1/2,3/2]), density=True)
    star_hist = stats.rv_histogram((np.histogram(Omega_star, bins=2, density=True)[0], [-1/2,1/2,3/2]), density=True)
    
    #
    fig=plt.figure(figsize=(7,6))
    plt.plot(x, gmm_hist.pdf(x), label="database")
    plt.plot(x, star_hist.pdf(x), label="Omega*")
    plt.plot(x, Y.pdf(x), label="h*")
    plt.legend()
    fig.savefig('./imgs/algos/subsampling/example_hist.png', dpi=200, bbox_inches='tight')
    plt.show()