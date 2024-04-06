import numpy as np
import scipy.stats as stats
import data_generator as data
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import metrics

# Instead of the optimization problem described in the vershynin stat framework paper. This function minimizes the 
# wasserstein distance over the weighted sets
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


def naive_reweight(Omega_star, database, noise):
    n = len(database)
    m = len(Omega_star)
    c = np.array([1] + [0 for _ in range(len(Omega_star))])
    A_ub = []
    b_ub = []
    row1_ub = [-1]  # negative sign
    row2_ub = [-1]  # positive sign
    for z_i in Omega_star:
        row1_ub.append(-z_i)
        row2_ub.append(z_i)
    #print(row1_ub)
    #print("")
    A_ub.append(row1_ub)
    A_ub.append(row2_ub)
    # construct the b_ub for the inequality constraints
    #noise[j] = 0
    b_ub.append(-np.mean(database))
    b_ub.append(np.mean(database))

    A_eq = np.array([[0] + [1 for _ in range(m)]])
    b_eq = np.array([1])

    individual_bounds = [(0,None)]
    for _ in range(m):
        individual_bounds.append((0,1))

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
    print(Omega_star.shape)
    print(h_weights.shape)
    chosen_indices = np.random.choice([i for i in range(n)], size=k, p=h_weights)
    return Omega_star[chosen_indices]

def subsample(database, m):
    d = database.shape[1]
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

def get_testfunctions_binary_3():
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
    d = database.shape[1]
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
    return F



def subsampling_mechanism(database, test_functions, delta, gamma, x, k, sigma, m):
    noise = data.laplacian_noise(0, sigma, (len(test_functions), 1))
    
    # drawing uniform from Omega
    Omega_star = subsample(database, m)
    res_of_minproblem = reweight(database, test_functions, Omega_star, noise)
    h_weights = res_of_minproblem.x[1:]
    # Finally get our synthetic data
    y = bootstrap(h_weights, Omega_star, k)
    # Create a histogram of it and plot this against the initial histogram of the database
    hist_of_SD = stats.rv_histogram(np.histogram(y, bins=int(2*np.sqrt(len(y))), density=True))
    hist_of_database = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(len(database))), density=True))
    
    # Calculate distances between the different distributions/histograms
    KS = metrics.KS(hist_of_SD, hist_of_database, x)
    L2 = metrics.L2(hist_of_SD, hist_of_database, x)
    W1 = metrics.multivW1(database, y)
    
    return KS, L2, W1

def subsampling_mechanism_bin(database, test_functions, delta, gamma, k, sigma, m):
    d = database.shape[1]
    n = database.shape[0]
    noise = data.laplacian_noise(0, sigma, (len(test_functions), 1))
    
    # drawing uniform from Omega
    Omega_star = subsample_binary(database, m)
    res_of_minproblem = reweight(database, test_functions, Omega_star, noise)
    h_weights = res_of_minproblem.x[1:]
    # Finally get our synthetic data
    y = bootstrap(h_weights, Omega_star, k)
    print("yshape")
    print(y.shape)
    print(Omega_star.shape)
    # Calculate distances between the different distributions/histograms
    W1 = metrics.W1(database, y)
    
    return W1


def check_guarantees(n, epsilon, delta, gamma, k, F_len):
    if k > n: 
        print("ERROR - ACCURACY VIOLATION: k>n")
        return False
    min_n = 2*1/(epsilon*delta)*F_len*np.log(F_len/gamma)
    #print(min_n)
    if n < min_n:
        print("ERROR - PRIVACY VIOLATION")
        return False

    return True


def subsampling_bin(Ns, database, epsilon, delta, gamma):
    all_W1_dists = []
    for n in Ns:
        sample = database[:n]

        d = database.shape[1]
        print("dim: "+str(d))
        test_functions = get_testfunctions_binary(d)

        F_len = len(test_functions)
        sigma = delta/np.log(F_len/gamma)     # see Thm 2.3
        
        # The size of the "much smaller" Omega_star doesn't necessarily need to grow with n
        # But we make it do this to hope and see a convergence.
        K = 10 # this is based in the Renyi divergence. Just a guessed upper bound! TODO: Calculate this
        m = np.sqrt(n).astype(int)
        # We want the size of the synthetic dataset to grow with n but also to satisfy the guarantees 
        # see Thm 2.3 in vershynin paper, with the assumption that n>k
        k = (1/4 * m).astype(int)
        # Check if the privacy and accuracy guarantees are fulfilled. Otherwise return
        check_guarantees(n, epsilon, delta, gamma, k, F_len)
        #
        W1 = subsampling_mechanism_bin(sample, test_functions, delta, gamma, k, sigma, m)
        all_W1_dists.append(W1)

    return all_W1_dists


def subsampling(Ns, database, epsilon, delta, gamma, x):
    all_KS_dists = []
    all_L1_dists = []
    all_W1_dists = []
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
        # We want the size of the synthetic dataset to grow with n but also to satisfy the guarantees 
        # see Thm 2.3 in vershynin paper, with the assumption that n>k
        k = max(delta**(-2)*np.log(F_len/gamma), np.sqrt(m))
        k = np.ceil(k).astype(int)
        k = (m/4).astype(int)
        # Check if the privacy and accuracy guarantees are fulfilled. Otherwise return
        check_guarantees(n, epsilon, delta, gamma, k, F_len)
        #
        KS, L1, W1 = subsampling_mechanism(sample, test_functions, delta, gamma, x, k, sigma, m)
        all_KS_dists.append(KS)
        all_L1_dists.append(L1)
        all_W1_dists.append(W1)

    return all_KS_dists, all_L1_dists, all_W1_dists




# deprecated?
# delta should be in (0, 1/4]
# gamma should be in (0, 1/2]
# K should be based on Renyi divergence
# this implementation of the algorithm samples Omega_star from Omega with a std normal
def show_subsampling_mechanism(database, test_functions, delta, gamma):
    n = len(database)
    sigma = delta/np.log(1000/gamma)     # see Thm 2.3
    #m = np.sqrt(n).astype(int)
    m = int(0.1*n)
    k = int(m/2)
    #
    noise = data.laplacian_noise(0, sigma, (m, 1))
    noise = noise.reshape((m,))
    # drawing uniform from Omega
    Omega_star = subsample(database, m)
    res_of_minproblem = reweight(database, test_functions, Omega_star, noise)
    print(res_of_minproblem.fun)
    print(sum(res_of_minproblem.x))
    h_weights = res_of_minproblem.x[1:]
    # Finally get our synthetic data
    y = bootstrap(h_weights, Omega_star, k)
    x = np.linspace(min(database), max(database), 1000)
    # Create a histogram of it and plot this against the initial histogram of the database
    Y = stats.rv_histogram(np.histogram(y, bins=int(np.sqrt(len(y)))))
    gmm_hist = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(len(database)))))
    star_hist = stats.rv_histogram(np.histogram(Omega_star, bins=int(np.sqrt(len(Omega_star)))))
    #
    plt.plot(x, gmm_hist.pdf(x), label="database")
    plt.plot(x, star_hist.pdf(x), label="Omega*")
    plt.plot(x, Y.pdf(x), label="h*")
    plt.legend()
    plt.show()