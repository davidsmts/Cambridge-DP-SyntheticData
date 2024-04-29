import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import data_generator as data
import ot as pot
import scipy.linalg as linalg
from perturbed_histogram import PerturbedHistogram
import metrics
import itertools as it
import time

linestyles = ["solid", "dotted", "dashed", "dashdot"]
colors = ["blue", "green", "orange", "red"]
# Smooth histogram method

class Histogram:
    def __init__(self, data, bin_amt=-1, dim=1, delta=1/2) -> None:
        self.data = data
        self.bin_amt = bin_amt
        self.dim = dim
        self.delta = delta
        self.get_counts()
        pass

    def evaluate(self, x):
        multidim = np.array(list(it.product(x)))
        indices = []
        for d in range(self.dim):
            column = x[:]
            id_col = np.digitize(column, self.boundaries[d], right=False)
            id_col = np.clip(id_col, 0, self.bin_amt)
            indices.append(id_col.tolist())
        indices = np.array(indices[0])
        vals = []
        for index in indices:
            vals.append(self.counts[index-1])
        indices_small = (x < np.min(self.boundaries)).nonzero()
        indices_large = (x > np.max(self.boundaries)).nonzero()
        for index in indices_small[0]:
            vals[index] = 0
        for index in indices_large[0]:
            vals[index] = 0
        return vals
    
    #
    def pdf(self, x):
        return self.evaluate(x)

    #
    def get_counts(self):
        bin_amts = np.full(self.dim, fill_value=self.bin_amt)
        #print("np hsitdd")
        # this can take a lot of time if there are too many bins
        true_counts, edges = np.histogramdd(self.data, bins=bin_amts, density=True) 
        #print("DONE histdd")
        self.boundaries = np.array(edges)
        self.counts = true_counts
        self.probabilities = true_counts / np.sum(true_counts)

    #
    def sample(self, amt, dim=1):
        initial_shape = self.probabilities.shape
        flat_p = self.probabilities.flatten()
        choices = np.random.choice(np.arange(len(flat_p)), size=amt, p=flat_p)
        ddim_indices = self.get_indices_new(choices, flat_p)
        points = self.get_points_from_indices(ddim_indices)
        return points
    
    #
    def get_indices(self, choices):
        indices = []
        for choice in choices:
            index_k = []
            for k in range(0, self.dim):
                index_k.append(int(choice/(self.bin_amt**k))%self.bin_amt)
            #index_k.append(choice%self.bin_amt)
            indices.append(index_k)
        return indices
    
    #
    def get_indices_new(self, choices, flat_p):
        indices = []
        index_tensor = np.reshape(np.arange(len(flat_p)), newshape=self.probabilities.shape)
        for choice in choices:
            index_of_choice = np.where(index_tensor==choice)
            indices.append(index_of_choice)

        return indices
    
    #
    def get_points_from_indices(self, indices):
        points = []
        for j, index in enumerate(indices):
            lb = []
            ub = []
            for i, ind_dim in enumerate(index):
                lb.append(self.boundaries[i][ind_dim][0])                    
                ub.append(self.boundaries[i][ind_dim+1][0])

            #print(lb.shape)
            # draw random point from that histogram bin!!!
            point = np.random.uniform(lb, ub, size=(1,self.dim))
            points.append(list(point)[0])
        return np.array(points)

    #
    def set_smooth(self):
        self.counts = (1-self.delta)*self.counts + self.delta
        #self.counts = hist_vals
        newprobs = (1-self.delta)*self.probabilities + self.delta
        self.probabilities = newprobs / np.sum(newprobs)
        #return (hist_vals, self.boundaries[0])

    #
    def set_perturbed(self, noise=[]):
        if len(noise) == 0 or len(np.where(noise < 0, noise, noise))==len(noise):
            noise = np.random.laplace(0, 8, [self.bin_amt for _ in range(self.dim)])
        #print("NOISE")
        #print(noise)
        D = self.counts + noise.reshape(self.counts.shape)
        #print("D")
        #print(D)
        D[D<0] = 0
        #print(D)
        if np.sum(D) == 0:
            print("SUM IS ZERO")
        elif np.sum(D) == np.inf:
            print("SUM IS INF")

        p = D / np.sum(D)
        self.noisy_counts = D
        self.probabilities = p

    
    #
    def set_rwm_perturbed(self, noise=[]):
        if len(noise) == 0:
            noise = data.get_superregular_rw(self.bin_amt**self.dim)
        D = self.counts + noise.reshape(self.counts.shape)
        D[D<0] = 0
        p = D / np.sum(D)
        self.noisy_counts = D
        self.probabilities = p


# one dimensional histogram on the interval [a,b]
def histogram(x, data, bin_amt, closed_interval=(0,1)):
    a = closed_interval[0]
    b = closed_interval[1]
    n = len(data)
    bin_width = (b-a)/bin_amt
    
    # Determine the bin that x lands in
    bin_num_of_x = 0
    for j in range(bin_amt):
        if a+j*bin_width <= x < a+(j+1)*bin_width:
            bin_num_of_x = j
    
    # The following case is needed because the for loop doesn't account for the case x==b 
    if bin_num_of_x == 0 and x == b:
        bin_num_of_x = bin_amt

    # Count number of sample points in bin j
    bin_count = 0   # C_j in the Wassermann & Zhou paper
    for x_i in data:
        if a+bin_num_of_x*bin_width <= x_i <= a+(bin_num_of_x+1)*bin_width:
            bin_count += 1

    # Number of sample points in bin_of_x divided by amount of sample points
    p_hat_j = 1/n * bin_count
    
    return p_hat_j/bin_width


def getfull_histogram(data, bin_amt, closed_interval=(0,1)):
    a = closed_interval[0]
    b = closed_interval[1]
    n = len(data)
    bin_width = (b-a)/bin_amt
    
    # Count number of sample points in bin j
    bin_counts = []   # C_j in the Wassermann & Zhou paper
    boundaries = []
    for j in range(bin_amt):
        bin_count = 0
        for x_i in data:
            if a+j*bin_width <= x_i <= a+(j+1)*bin_width:
                bin_count += 1
        bin_counts.append(bin_count)
        boundaries.append(a+(j+1)*bin_width)
    probabs = np.array(bin_counts)/(n*bin_width)
    return (boundaries, probabs)


def smooth_histogram(data, delta=0.1, closed_interval=(0,1), for_dist="L2"):
    n = len(data)
    r = 1
    m = n**(r/(2*r+3))
    #print("m="+str(m))
    m = int(m)
    k = n**((r+2)/(2*r+3))
    #delta = n**(-1/(r+3))
    #print("delta="+str(delta))
    alpha = k*np.log(1+((1-delta)*m)/(n*delta))

    hist = np.histogram(data, bins=m, density=False)
    hist_vals = (1-delta)*hist[0] + delta
    return (hist_vals, hist[1])


def perturbed_histogram(x, data, bin_amt, closed_interval=(0,1), noise=[]):
    a = closed_interval[0]
    b = closed_interval[1]
    n = len(data)
    bin_width = (b-a)/bin_amt
    
    # Determine the bin that x lands in
    bin_num_of_x = 0

    # Count number of sample points in bin j and find the bin of x
    bin_counts = []   # C_j in the Wassermann & Zhou paper
    for j in range(bin_amt):
        # check for the bin of the input variable x
        if a+j*bin_width <= x < a+(j+1)*bin_width:
                bin_num_of_x = j
        # count amount of sample points in this bin
        bin_count = 0
        for x_i in data:
            if a+bin_num_of_x*bin_width <= x_i <= a+(bin_num_of_x+1)*bin_width:
                bin_count += 1
        bin_counts.append(bin_count)

    # The following case is needed because the for loop doesn't account for the case x==b 
    if bin_num_of_x == 0 and x == b:
        bin_num_of_x = bin_amt-1

    # perturb the histogram
    D_j = bin_count + noise[bin_num_of_x-1]
    D_j = D_j[0]
    D_j_tilde = max(D_j, 0)
    p_tilde_j = D_j_tilde / sum(bin_counts)
    
    return p_tilde_j/bin_width


def show_regular_histogram():
    x_vals = np.linspace(-5,5,1000)
    sample = np.random.normal(0, 1, (sample_size, dim_of_data))
    y_vals = histogram(x, sample, 40, closed_interval=(-5,5))
    plt.plot(x_vals, y_vals, label="hist with 40 bins")
    plt.plot(x_vals, stats.norm.pdf(x_vals), label="std. normal")
    plt.legend()
    plt.show()

def show_smooth(type="std", delta=0):
    interval = (0,0.999)
    sample_sizes = [1000, 10000,100000]
    dim_of_data = 1
    y_vals_by_ssize = []
    regular_y_vals_by_ssize = []
    for sample_size in sample_sizes:
        y_vals = [] 
        x_vals = np.linspace(0,0.99,100)
        sample = np.random.uniform(0,1,(sample_size, dim_of_data))
        regular_y_vals = []
        for x in x_vals:
            y_vals.append(smooth_histogram(x, sample, delta=delta, closed_interval=interval))
            m = int(sample_size**(1/5))
            regular_y_vals.append(histogram(x, sample, bin_amt=m, closed_interval=interval))
        y_vals_by_ssize.append(y_vals)
        regular_y_vals_by_ssize.append(regular_y_vals)

    for (i, y_vals) in zip(range(len(y_vals_by_ssize)), y_vals_by_ssize):    
        plt.plot(x_vals, y_vals, label="n="+str(sample_sizes[i]), linestyle=linestyles[0], color=colors[i])

    for (i, y_vals) in zip(range(len(regular_y_vals_by_ssize)), regular_y_vals_by_ssize):    
        plt.plot(x_vals, y_vals, label="non-smoothed n="+str(sample_sizes[i]), linestyle=linestyles[1], color=colors[i])

    y_vals = np.ones(len(x_vals))
    plt.plot(x_vals, y_vals, label="std.")

    plt.legend()
    plt.show()


'''
def show_perturbed(epsilon=1, sample_size=50):
    sample = np.random.uniform(0,1,(sample_size, 1))
    x_vals = np.linspace(0,1,100)

    y_vals_byamt = []
    bin_amts = [40]
    regular_hist_y_vals = []
    for bin_amt in bin_amts:
        # Get the laplacian noise
        noise = np.random.laplace(0, 8/epsilon**2, (bin_amt, 1))
        y_vals = [] 
        for x in x_vals:
            regular_hist_y_vals.append(histogram(x, sample, bin_amt, closed_interval=(-5,5)))
            val = perturbed_histogram(x, sample, bin_amt, closed_interval=(0,1), noise=noise)
            y_vals.append(val)
        print(y_vals)
        y_vals_byamt.append(y_vals)

    for (i, y_vals) in zip(range(len(y_vals_byamt)), y_vals_byamt):    
        plt.plot(x_vals, y_vals, label="perturbed hist with bin size="+str(bin_amts[i]))
    
    # plot the regular histogram (without perturbation)
    plt.plot(x_vals, regular_hist_y_vals, label="histogram wout perturbation")

    # plot the constant 1 function
    y_vals = np.ones(len(x_vals))
    plt.plot(x_vals, y_vals, label="f(x)=1")

    # reveal plot and legend
    plt.legend()
    plt.show()
'''

#
# Note that this already computes the average of 20 runs.
def smooth_on_normal(average_over=10):
    sample_sizes = [200, 1000, 10000, 50000, 100000] 
    x = np.linspace(0,3,100)

    all_KS_dists = []
    all_L2_dists = []
    all_W1_dists = []

    for i in range(average_over):
        #print(i)
        #generate data
        database = data.schoel_balog_tostik_sample(max(sample_sizes),1)
        #x = np.linspace(min(database),max(database),1000)

        KS_dists = []
        L2_dists = []
        W1_dists = []
        for size in sample_sizes:
            sample = database[:size]
            closed_interval = (min(sample),max(sample))
            m = int(size**(1/5))    # note that this is the m for L_2
            F = stats.rv_histogram(np.histogram(sample[:size], bins=m))
            histog = smooth_histogram(sample[:size], closed_interval=closed_interval)
            smooth_hist = stats.rv_histogram(histog)
            k = int(size**(3/5))    # note that this is the k for L_2
            Y = smooth_hist.rvs(size=k)
            F_Y = stats.rv_histogram(np.histogram(Y, bins=k))
            diff = np.abs(F_Y.cdf(x) - F.cdf(x))
            #plt.plot(x, diff, label="n="+str(size))
            KS_dists.append(max(diff))
            L2_dists.append(linalg.norm(diff,ord=2)**2)
            a = pot.unif(size)
            b = pot.unif(k)
            mu_1 = sample[:size]/sum(sample[:size])
            mu_1 = mu_1.reshape((len(mu_1)))
            mu_2 = Y/sum(Y)
            W1_dists.append(pot.emd2_1d(mu_1, mu_2))
        all_KS_dists.append(KS_dists)
        all_L2_dists.append(L2_dists)
        all_W1_dists.append(W1_dists)
    
    fig=plt.figure(figsize=(7,6))
    plt.loglog(sample_sizes, np.mean(all_KS_dists, axis=0), label="KS distances", linestyle="dashed", color=colors[0])
    plt.loglog(sample_sizes, np.mean(all_L2_dists, axis=0), label="L2 distances", linestyle="dotted", color=colors[1])
    plt.loglog(sample_sizes, np.mean(all_W1_dists, axis=0), label="W1 distances", linestyle="solid", color=colors[2])
    #plt.loglog(sample_sizes, np.power(sample_sizes, -1/2), label="n^(-1/2)", linestyle="dashed", color=colors[0])
    #plt.loglog(sample_sizes, 1/np.array(sample_sizes), label="n^-1", linestyle="dotted", color=colors[1])
    # display legend, save image and display image
    plt.legend()
    fig.savefig('./imgs/smooth-dpsd-loglog.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_perturbed():
    n = 10000
    database = data.schoel_balog_tostik_sample(n, 1)
    sizes = [int(np.exp(i)) for i in range(int(np.ceil(np.exp(1))), int(np.log(n)))]
    epsilons = [1, 0.1, 0.01, 0.001]
    for i,epsilon in enumerate(epsilons):
        all_KS = []
        all_L2 = []
        all_W1 = []
        for size in sizes:
            KS, L2, W1 = get_perturbed_accuracy(database[:size], epsilon)
            all_KS.append(KS)
            all_L2.append(L2)
            all_W1.append(W1)

        plt.loglog(sizes, all_KS, label="KS,eps="+str(epsilon), linestyle="solid", color=colors[i])
        #plt.loglog(sizes, all_L2, label="L2,eps="+str(epsilon), linestyle="dashed", color=colors[i])
        #plt.loglog(sizes, all_W1, label="W1,eps="+str(epsilon), linestyle="dotted", color=colors[i])

    plt.legend(loc='upper right')
    plt.show()

def asymptotic_perturbed(Ns, database=[], epsilon=1, dim=1, rwm=False):
    all_KS = []
    all_L2 = []
    all_W1 = []
    times = []
    all_TF = []
    for n in Ns:
        n_time, KS, L2, W1, TF = get_perturbed_accuracy(database[:n], epsilon, dim=dim, rwm=rwm)
        all_KS.append(KS)
        all_L2.append(L2)
        all_W1.append(W1)
        times.append(n_time)
        all_TF.append(TF)
        
    return times, all_KS, all_L2, all_W1, all_TF

def asymptotic_smooth(Ns, database=[], epsilon=1, dim=1):
    all_KS = []
    all_L2 = []
    all_W1 = []
    times = []
    all_TF = []
    for n in Ns:
        n_time, KS, L2, W1, TF = get_smooth_accuracy(database[:n], dim=dim)
        all_L2.append(L2)
        all_W1.append(W1)
        all_KS.append(KS)
        times.append(n_time)
        all_TF.append(TF)

    return times, all_KS, all_L2, all_W1, all_TF

def get_perturbed_accuracy(database, epsilon, x=[], dim=1, rwm=False):
    start_time = time.time()
    #
    n = len(database)
    m = int(np.ceil(n**(dim/(2+dim))))   # don't need a case distinction for m here because they're the same for KS and L2
    #epsilon = np.sqrt(n)
    # Get empirical distribution frim the data. This will be used to measure the accuracy below.
    #F = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(len(database)))))
    # The following is the underlying space over which the perturbed histogram will be constructed
    # The distances between the distributions will be calculate at those points
    #if len(x) == 0:
    #    x = np.linspace(min(database), max(database), 1000)
    # choose bin amount based on sqrt rule

    if rwm:
        histogram = Histogram(data=database, bin_amt=m, dim=dim)
        histogram.set_rwm_perturbed()
    else:
        # Generate noise based on wasserstein&zhou paper
        #LapNoise = data.WZ_Lap_noise(epsilon, m, dim=dim)
        LapNoise = np.random.laplace(0, 8/epsilon**2, tuple([m for _ in range(dim)]))
        # Create a perturbed histogram
        histogram = Histogram(data=database, bin_amt=m, dim=dim)
        histogram.set_perturbed(LapNoise)
    # Set k to .. based on Wasserstein&Zhou paper
    k = len(database)
    # Sample k points from our histogram
    synthetic_data = histogram.sample(amt=k, dim=dim)

    # create a density estimation of this via .. 
    #F_Y = stats.rv_histogram(np.histogram(synthetic_data, bins=k))
    #f_Y = stats.gaussian_kde(synthetic_data.T)
    #f = stats.gaussian_kde(database.T)
    
    #
    Data_Hist = Histogram(data=database, bin_amt=int(m), dim=dim, delta=0)
    SD_Hist = Histogram(data=synthetic_data, bin_amt=int(m), dim=dim, delta=0)
    
    #
    stop_time = time.time() - start_time
    
    W1 = metrics.multivW1(database, synthetic_data)
    L2 = metrics.smartL2_hypercube(Data_Hist.probabilities, SD_Hist.probabilities, m)
    KS = metrics.smartKS_hypercube(Data_Hist.probabilities, SD_Hist.probabilities, m, dim=dim)
    # TF
    test_functions = data.get_binary_testfunctions_upto(dimension=dim, max_order=False)
    TF = metrics.wrt_marginals(test_functions=test_functions, sample1=database, sample2=synthetic_data, dim=dim)
    
    return stop_time, KS, L2, W1, TF


def get_smooth_accuracy(database, dim=1, distance="L2", epsilon=1):
    start_time = time.time()
    n = len(database)
    #
    k_KS = int((n**(4/(6+dim))))
    m_KS = int(n**(dim/(2+dim)))
    delta_KS = m_KS*k_KS/(n*epsilon)
    #
    delta_L2 = n**(-1/(dim+3))
    k_L2 = int(n**((dim+2)/(2*dim+3)))
    m_L2 = int(n**(dim/(2*dim+3)))
    # Get empirical distribution frim the data. This will be used to measure the accuracy below.
    #F_KS = stats.rv_histogram(np.histogram(database, bins=int(m_KS)))
    #F_L2 = stats.rv_histogram(np.histogram(database, bins=int(m_L2)))
    # The following is the underlying space over which the perturbed histogram will be constructed
    # The distances between the distributions will be calculate at those points
    #if len(x) == 0:
    #    x = np.linspace(min(database), max(database), 1000)
    # create smooth histogram
    # draw k points from smooth histogram
    smooth_hist_KS = Histogram(data=database, bin_amt=int(m_KS), dim=dim, delta=delta_KS)
    smooth_hist_L2 = Histogram(data=database, bin_amt=int(m_L2), dim=dim, delta=delta_L2)
    smooth_hist_KS.set_smooth()
    smooth_hist_L2.set_smooth()
    synthetic_data_KS = smooth_hist_KS.sample(k_KS)
    synthetic_data_L2 = smooth_hist_L2.sample(k_L2)
    #
    Data_Hist_KS = Histogram(data=database, bin_amt=int(m_KS), dim=dim, delta=0)
    Data_Hist_L2 = Histogram(data=database, bin_amt=int(m_L2), dim=dim, delta=0)
    SD_Hist_KS = Histogram(data=synthetic_data_KS, bin_amt=int(m_KS), dim=dim, delta=0)
    SD_Hist_L2 = Histogram(data=synthetic_data_L2, bin_amt=int(m_L2), dim=dim, delta=0)
    
    stop_time = time.time() - start_time

    # Calculate the difference between the two distributions for various metrics
    W1 = metrics.multivW1(database, synthetic_data_KS)
    L2 = metrics.smartL2_hypercube(SD_Hist_L2.probabilities, Data_Hist_L2.probabilities, m_L2)
    KS = metrics.smartKS_hypercube(SD_Hist_KS.probabilities, Data_Hist_KS.probabilities, m_KS, dim=dim)
    #print("--------KS---------")
    #print(synthetic_data_KS)
    #print(KS)
    # TF
    test_functions = data.get_binary_testfunctions_upto(dimension=dim, max_order=False)
    TF = metrics.wrt_marginals(test_functions=test_functions, sample1=database, sample2=synthetic_data_L2, dim=dim)

    return stop_time, KS, L2, W1, TF

def display_smooth_histogram(display = True):
    d = 1
    database = data.schoel_balog_tostik_sample(10000,1)
    database = database / np.max(database)
    n = 1000
    k = int(np.sqrt(n))
    database = database[:n]
    x = np.linspace(min(database), max(database), 1000)
    F1 = stats.rv_histogram(np.histogram(database, bins=k), density=True)
    delt = 10
    F2 = Histogram(database, bin_amt=k, dim=1, delta=1/delt)
    F3 = stats.rv_histogram((F2.probabilities, F2.boundaries[0]), density=True)
    bins, edges = np.histogramdd(database, bins=k, density=True)
    F4 = stats.rv_histogram(F2.get_smooth(), density=True)
    fig=plt.figure(figsize=(7,6))
    plt.plot(x, F1.pdf(x), label="regular")
    #plt.plot(x, F2.pdf(x), label="custom2")
    plt.plot(x, F4.pdf(x), label="smooth")
    plt.xlabel("sample space")
    plt.ylabel("histogram values")
    plt.legend()
    fig.savefig('./imgs/algos/hist/smooth_delta_'+str(delt)+'.png', dpi=300, bbox_inches='tight')
    if display:
        plt.show()


def display_perturbed_accuracy(display = True):
    d = 1
    Ns = [int(np.exp(i)) for i in np.linspace(4.5, int(np.log(20000)), 6)]
    print(Ns)
    database = data.getACS(amt=max(Ns), dim=d)
    epsilon = 1
    reps = 5
    all_RKHS_kme = []
    all_W1_rwm = []
    all_W1_phist = []
    all_L2_phist = []
    all_KS_phist = []
    all_W1_shist = []
    all_W1_pmm = []
    all_TF_subs = []
    for rep in range(reps):
        if rep%3 == 0: print(rep)
        time, KS, L2, W1_phist, TF = asymptotic_perturbed(Ns, database, epsilon, dim=d)

        all_W1_phist.append(W1_phist)
        all_L2_phist.append(L2)
        all_KS_phist.append(KS)

    all_W1_phist = np.mean(all_W1_phist, axis=0)
    all_L2_phist = np.mean(all_L2_phist, axis=0)
    all_KS_phist = np.mean(all_KS_phist, axis=0)
    L2_bnd = [3*n**(-2/3) for n in Ns]
    # , np.log(n)/n**(2/3), np.sqrt(np.log(n)/n)
    KS_bnd = [min(np.sqrt(np.log(n)/n), np.log(n)/n**(2/3)) for n in Ns]
    fig=plt.figure(figsize=(7,6))
    plt.loglog(Ns, all_W1_phist, label="W1", linestyle="solid")
    plt.loglog(Ns, all_L2_phist, label="L2", linestyle="dotted")
    plt.loglog(Ns, L2_bnd, label="n^(-2/5)", color=(0.8, 0.439, 0), alpha=0.9, linestyle="dotted")
    plt.loglog(Ns, all_KS_phist, label="KS", linestyle="dashed")
    plt.loglog(Ns, KS_bnd, label="min(sqrt(log(n)/n), log(n)/n**(2/3))", color="green", alpha=0.9, linestyle="dashed")
    plt.xlabel("database size")
    plt.ylabel("distance between true and synthetic data")
    plt.legend()
    fig.savefig('./imgs/algos/hist/perturbed_on_acm_loglog.png', dpi=300, bbox_inches='tight')
    if display:
        plt.show()


def display_smooth_accuracy(display=True):
    d = 1
    Ns = [int(np.exp(i)) for i in np.linspace(6, int(np.log(30000)), 10)]
    database = data.schoel_balog_tostik_sample(amt=max(Ns), dim=d)
    print(Ns)
    x = np.linspace(0, 1, 300)
    epsilon = 1
    reps = 10

    all_W1_shist = []
    all_L2_shist = []
    all_KS_shist = []

    for rep in range(reps):
        if rep%3 == 0: print(rep)
        #KS, L2, W1_phist = asymptotic_perturbed(Ns, x, database, epsilon, dim=d)
        time, KS, L2, W1, TF = asymptotic_smooth(Ns, database, epsilon, dim=d)
        all_W1_shist.append(W1)
        all_L2_shist.append(L2)
        all_KS_shist.append(KS)

    all_W1_shist = np.mean(all_W1_shist, axis=0)
    all_L2_shist = np.mean(all_L2_shist, axis=0)
    all_KS_shist = np.mean(all_KS_shist, axis=0)
    L2_bnd = [1/5*n**(-2/5) for n in Ns]
    KS_bnd = [1/20*np.sqrt(np.log(n))*n**(-2/7) for n in Ns]

    fig=plt.figure(figsize=(7,6))
    plt.loglog(Ns, all_W1_shist, label="W1", linestyle="solid")
    plt.loglog(Ns, all_L2_shist, label="L2", linestyle="dotted")
    plt.loglog(Ns, L2_bnd, label="n^(-2/5)", color=(0.8, 0.439, 0), alpha=0.9, linestyle="dotted")
    plt.loglog(Ns, all_KS_shist, label="KS", linestyle="dashed")
    plt.loglog(Ns, KS_bnd, label="(logn)^(1/2)/n^(2/7)", color="green", alpha=0.9, linestyle="dashed")
    plt.legend()
    plt.xlabel("database size")
    plt.ylabel("distance between true and synthetic data")
    fig.savefig('./imgs/algos/hist/smooth_on_acm_loglog.png', dpi=300, bbox_inches='tight')
    if display:
        plt.show()