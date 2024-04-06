import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import data_generator as data
import ot as pot
import scipy.linalg as linalg
from perturbed_histogram import PerturbedHistogram
import metrics

linestyles = ["solid", "dotted", "dashed", "dashdot"]
colors = ["blue", "green", "orange", "red"]
# Smooth histogram method


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
        print(i)
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

def asymptotic_perturbed(Ns, x, database, epsilon):
    all_KS = []
    all_L2 = []
    all_W1 = []
    for n in Ns:
        KS, L2, W1 = get_perturbed_accuracy(database[:n], epsilon, x=x)
        all_KS.append(KS)
        all_L2.append(L2)
        all_W1.append(W1)
    
    return all_KS, all_L2, all_W1

def asymptotic_smooth(Ns, x, database, epsilon):
    all_KS = []
    all_L2 = []
    all_W1 = []
    for n in Ns:
        delta = 1/np.sqrt(n)
        KS, L2, W1 = get_smooth_accuracy(database[:n], delta, x=x)
        all_KS.append(KS)
        all_L2.append(L2)
        all_W1.append(W1)
    
    return all_KS, all_L2, all_W1

def get_perturbed_accuracy(database, epsilon, x=[]):
    n = len(database)
    # Get empirical distribution frim the data. This will be used to measure the accuracy below.
    F = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(len(database)))))
    # The following is the underlying space over which the perturbed histogram will be constructed
    # The distances between the distributions will be calculate at those points
    if len(x) == 0:
        x = np.linspace(min(database), max(database), 1000)
    # choose bin amount based on sqrt rule
    bin_amt = int(len(database)**(1/2))
    # Generate noise based on wasserstein&zhou paper
    LapNoise = data.WZ_Lap_noise(epsilon, bin_amt)
    # Create a perturbed histogram
    histogram = PerturbedHistogram(database, LapNoise)
    # Set k to .. based on Wasserstein&Zhou paper
    k = int(len(database)/2)
    # Sample k points from our histogram
    Y = histogram.sample(k)
    print(Y)
    
    # create a density estimation of this via .. 
    F_Y = stats.rv_histogram(np.histogram(Y, bins=k))

    # Calculate the difference between the two distributions for various metrics
    KS = metrics.KS(F, F_Y, x)
    L2 = metrics.L2(F, F_Y, x)
    W1 = metrics.W1(database[:n], Y)

    return KS, L2, W1

def get_smooth_accuracy(database, delta, x=[]):
    n = len(database)
    # Get empirical distribution frim the data. This will be used to measure the accuracy below.
    F = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(len(database)))))
    # The following is the underlying space over which the perturbed histogram will be constructed
    # The distances between the distributions will be calculate at those points
    if len(x) == 0:
        x = np.linspace(min(database), max(database), 1000)
    # create smooth histogram
    histog = smooth_histogram(database, closed_interval=(min(x), max(x)), delta=delta)
    smooth_hist = stats.rv_histogram(histog)
    # draw k points from smooth histogram
    k = int(n**(3/5))    # note that this is the k for L_2
    Y = smooth_hist.rvs(size=k)
    # create a density estimation of this via .. 
    F_Y = stats.rv_histogram(np.histogram(Y, bins=k))

    # Calculate the difference between the two distributions for various metrics
    KS = metrics.KS(F, F_Y, x)
    L2 = metrics.L2(F, F_Y, x)
    W1 = metrics.W1(database, Y)

    return KS, L2, W1