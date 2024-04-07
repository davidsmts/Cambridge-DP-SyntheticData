import sys
import numpy as np
import matplotlib.pyplot as plt
import histogram as hist
import kme_algo as kme
import scipy.stats as stats
import scipy.linalg as linalg
import ot as pot
import data_generator as data
import subsampling_method as vershynin
import rwm
import pmm

np.random.seed(42)


colors=["blue", "orange", "green"]

# Reading and interpreting the command line arguments in the following code
arguments = sys.argv
if len(arguments) < 2:
    print("Too few arguments. Specify the test you want to do. E.g. 'histogram'")
    sys.exit()


test = str(arguments[1])
print(test)
if test == "show_regular":
    hist.show_histogram()

elif test == "show-kme":
    kme.show()

elif test == "show_smooth":
    delta = 0.1
    if len(arguments) > 2:
        delta = float(arguments[2])
    hist.show_smooth()

elif test == "smooth-samples":
    sample_sizes = [100, 1000, 10000, 50000, 100000] 
    sample = np.random.uniform(0,1,(max(sample_sizes), 1))
    x = np.linspace(0,0.99,100)
    for size in sample_sizes:
        histog = hist.smooth_histogram(sample[:size])
        hist_dist = stats.rv_histogram(histog)
        plt.plot(x, hist_dist.pdf(x), label="n="+str(size))
    
    plt.legend()
    plt.show()

elif test == "s-hist-on-normal":
    hist.smooth_on_normal()

elif test == "p-hist-on-normal":
    hist.show_perturbed()

elif test == "show-subsampling":
    delta = 0.2
    gamma = 0.2
    epsilon = 1
    # Based on Theorem 2.2 in the vershynin paper
    # n_from_paper = np.ceil(2/(epsilon*delta) * len(test_functions)*np.log(len(test_functions)/gamma)).astype(int)
    d = 1
    n = 10000
    database = data.schoel_balog_tostik_sample(n, d)
    #database = database.reshape((n,d))
    test_functions = vershynin.get_testfunctions_binary(d)
    print("|F|="+str(len(test_functions)))
    #
    vershynin.show_subsampling_mechanism(database, test_functions, delta, gamma)

elif test == "asymptotic-subsampling-bin":
    delta = 0.2
    gamma = 0.2
    epsilon = 1
    # Based on Theorem 2.2 in the vershynin paper
    # n_from_paper = np.ceil(2/(epsilon*delta) * len(test_functions)*np.log(len(test_functions)/gamma)).astype(int)
    Ns = [1000, 5000, 10000, 40000, 60000, 100000]
    dims = [1]
    dists = []
    reps = 5
    for d in dims:
        dists.append([])
    for _ in range(reps):
        for i, d in enumerate(dims):
            database = data.binary_binomial_sample(max(Ns), d)
            W1 = vershynin.subsampling_bin(Ns, database, epsilon, delta, gamma)
            dists[i].append(W1)

    fig=plt.figure(figsize=(7,6))
    for i, w1 in enumerate(dists):
        plt.plot(Ns, np.mean(w1, axis=0), label="dim="+str(dims[i]))
    #plt.yticks([10**k for k in range(-5,1)])
    plt.legend()
    fig.savefig('./imgs/algos/subsampling/bin_multidim.png', dpi=300, bbox_inches='tight')
    plt.show()

elif test == "asymptotic-subsampling":
    delta = 0.2
    gamma = 0.2
    epsilon = 1
    d = 1
    # Based on Theorem 2.2 in the vershynin paper
    # n_from_paper = np.ceil(2/(epsilon*delta) * len(test_functions)*np.log(len(test_functions)/gamma)).astype(int)
    Ns = [1000, 5000, 10000, 40000, 60000, 100000]
    #database = data.schoel_balog_tostik_sample(max(Ns),1)
    database = data.schoel_balog_tostik_sample(max(Ns), d)
    #database = database.reshape((len(database),))
    x = np.linspace(min(database), max(database), 1000)
    KS, L2, W1 = vershynin.subsampling(Ns, database, epsilon, delta, gamma, x)
    plt.loglog(Ns, KS, label="KS")
    plt.loglog(Ns, L2, label="L2")
    plt.loglog(Ns, W1, label="W1")
    plt.legend()
    plt.show()

elif test == "asymptotic-subsampling-bydim":
    delta = 0.2
    gamma = 0.2
    epsilon = 1
    # Based on Theorem 2.2 in the vershynin paper
    # n_from_paper = np.ceil(2/(epsilon*delta) * len(test_functions)*np.log(len(test_functions)/gamma)).astype(int)
    Ns = [1000, 5000, 10000, 40000, 60000, 100000]
    dims = [2,3,4,5,6]
    dists = []
    m = 2
    for d in dims:
        dists.append([])
    for _ in range(m):
        for i, d in enumerate(dims):
            database = data.schoel_balog_tostik_sample(max(Ns), d)
            x = np.linspace(min(database), max(database), 1000)
            KS, L2, W1 = vershynin.subsampling(Ns, database, epsilon, delta, gamma, x)
            dists[i].append(W1)

    fig=plt.figure(figsize=(7,6))
    for i, w1 in enumerate(dists):
        plt.loglog(Ns, np.mean(w1, axis=0), label="dim="+str(dims[i]))
    plt.legend()
    fig.savefig('./imgs/algos/subsampling/multidim.png', dpi=300, bbox_inches='tight')
    plt.show()


    

elif test == "show-SBT-data":
    database = data.schoel_balog_tostik_sample(1000,1)
    x = np.linspace(min(database),max(database),1000)
    gmm_hist = stats.rv_histogram(np.histogram(database, bins=100))
    plt.plot(x, gmm_hist.pdf(x))
    plt.show()

elif test == "smooth-dpsd-unitsquare":
    sample_sizes = [200, 1000, 10000, 50000, 100000, 1000000] 
    x = np.linspace(0,0.999,100)

    all_KS_dists = []
    all_L2_dists = []
    all_W1_dists = []
    for i in range(20):
        print(i)
        sample = np.random.uniform(0,1,(max(sample_sizes), 1))
        KS_dists = []
        L2_dists = []
        W1_dists = []
        for size in sample_sizes:
            m = int(size**(1/5))    # note that this is the m for L_2
            F = stats.rv_histogram(np.histogram(sample[:size], bins=m, range=(0,1)))
            histog = hist.smooth_histogram(sample[:size])
            smooth_hist = stats.rv_histogram(histog)
            k = int(size**(3/5))    # note that this is the k for L_2
            Y = smooth_hist.rvs(size=k)
            F_Y = stats.rv_histogram(np.histogram(Y, bins=k, range=(0,1)))
            diff = F_Y.cdf(x) - F.cdf(x)
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
    plt.loglog(sample_sizes, np.mean(all_L2_dists, axis=0), label="L_2 distances", linestyle="dotted", color=colors[1])
    plt.loglog(sample_sizes, np.mean(all_W1_dists, axis=0), label="W1 distances", linestyle="solid", color=colors[2])
    plt.loglog(sample_sizes, np.power(sample_sizes, -1/2), label="n^(-1/2)", linestyle="dashed", color=colors[0])
    plt.loglog(sample_sizes, 1/np.array(sample_sizes), label="n^-1", linestyle="dotted", color=colors[1])
    # display legend, save image and display image
    plt.legend()
    fig.savefig('./imgs/smooth-dpsd-loglog.png', dpi=300, bbox_inches='tight')
    plt.show()


elif test == "superregular-rw":
    Ns = [10, 100, 500, 1000, 5000, 10000, 30000, 50000, 70000, 100000]
    # repeat m times to get the mean of the reps
    m = 100 
    mean_maxs = []
    for n in Ns:
        mean_res = np.zeros(n)
        all_res = []
        mean_max = 0
        for i in range(m):
            #if i%20 == 0: print(i)
            noise = data.laplacian_noise(0,np.log(n), shape=(n,))
            res = data.get_superregular_rw(n,  noise)
            all_res.append(res)
            mean_res = mean_res + res/m
            mean_max += np.max(res)/m
        mean_maxs.append(mean_max)
    
    #logn = lambda x: np.log(n)
    x = np.arange(n)
    fig=plt.figure(figsize=(7,6))
    plt.plot(Ns, mean_maxs, label="superregular rw")
    plt.plot(Ns, [np.log(n)**2 for n in Ns], label="(log(n))^2")
    plt.legend()
    fig.savefig('./imgs/algos/rwm/rw.png', dpi=300, bbox_inches='tight')
    plt.show()

elif test == "maxsuperregular-rw":
    Ns = [10, 100, 500, 1000, 5000, 10000, 30000, 50000, 70000, 100000]
    # repeat m times to get the mean of the reps
    m = 10 
    all_maxs = []
    for n in Ns:
        mean_res = np.zeros(n)
        all_res = []
        mean_max = 0
        for i in range(m):
            noise = data.laplacian_noise(0,np.log(n), shape=(n,))
            res = data.get_superregular_rw(n,  noise)
            all_res.append(np.max(res))
        all_maxs.append(all_res)

    all_maxs = np.array(all_maxs)
    all_maxs = all_maxs.T

    #logn = lambda x: np.log(n)
    x = np.arange(n)
    fig=plt.figure(figsize=(7,6))
    for row in all_maxs:
        plt.plot(Ns, row, linestyle="dashed")
    plt.plot(Ns, [3*np.log(n)**(3/2) for n in Ns], label="(log(n))^2")
    plt.legend()
    fig.savefig('./imgs/algos/rwm/maxrw.png', dpi=300, bbox_inches='tight')
    plt.show()

elif test == "show-haar":
    j = int(arguments[2])
    n = 10
    x = np.linspace(0, n+0.01, 200)
    psi = data.plot_haar_basis_functions(j,n,x)
    fig=plt.figure(figsize=(7,6))
    plt.plot(x, psi, linewidth=2)
    plt.axvline(x=0, c="black", label="", linewidth=0.8)
    plt.axhline(y=0, c="black", label="", linewidth=0.8)
    k = np.ceil(np.log2(j))
    xt = [float(j*n/2**(k-1)) for j in range(0, int(2**(k-1)+1))]
    print(xt)
    plt.xticks(xt, )
    #plt.xticks(x, [0 for _ in range(n-1)]+[n])
    plt.yticks([-2**(k-1)/n, 0, 2**(k-1)/n], [- 2**(k-1)/n, 0, 2**(k-1)/n])
    fig.savefig('./imgs/algos/rwm/haar'+str(j)+'.png', dpi=300, bbox_inches='tight')
    plt.show()

elif test == "asymptotic-rwm":
    epsilon = 1
    Ns = [5000, 10000, 40000, 70000, 100000, 500000]
    database = data.schoel_balog_tostik_sample(max(Ns),1)
    database = database.reshape((len(database),))
    x = np.linspace(min(database), max(database), 3000)
    m = 80
    all_KS = []
    all_L2 = []
    all_W1 = []
    for rep in range(m):
        if rep%10 == 0:
            print(rep)
        KS, L2, W1 = rwm.asymptotic_rwm(Ns, x, database, epsilon)
        all_KS.append(KS)
        all_L2.append(L2)
        all_W1.append(W1)
    fig=plt.figure(figsize=(7,6))
    if len(arguments)>2 and int(arguments[2]) == 1:
        plt.plot(Ns, np.mean(all_KS, axis=0), label="KS", linestyle="dashed")
        plt.plot(Ns, np.mean(all_L2, axis=0), label="L2", linestyle="dashed")
        plt.plot(Ns, np.mean(all_W1, axis=0), label="W1", linestyle="dotted")
        plt.plot(Ns, (1e-2)*rwm.asymptotic_ub_acc(Ns, epsilon), label="log(n)^(3/2)/alpha")
        plt.legend()
        fig.savefig('./imgs/algos/rwm/asymptotic_rwm.png', dpi=300, bbox_inches='tight')
    else:
        plt.loglog(Ns, np.mean(all_KS, axis=0), label="KS", linestyle="dashed")
        plt.loglog(Ns, np.mean(all_L2, axis=0), label="L2", linestyle="dashed")
        plt.loglog(Ns, np.mean(all_W1, axis=0), label="W1", linestyle="dotted")
        plt.loglog(Ns, (1e-2)*rwm.asymptotic_ub_acc(Ns, epsilon), label="log(n)^(3/2)/alpha")
        plt.legend()
        fig.savefig('./imgs/algos/rwm/asymptotic_rwm_loglog.png', dpi=300, bbox_inches='tight')
    plt.show()

elif test == "show-rwmhist":
    n = int(arguments[2])
    epsilon = 0.1
    database = data.schoel_balog_tostik_sample(n,0)
    database = database.reshape((n,))
    x = np.linspace(min(database), max(database), 3000)
    hist1, hist2 = rwm.hist_rwm(n, x, database, epsilon)
    fig=plt.figure(figsize=(7,6))
    plt.plot(x, hist1.pdf(x), label="hist of std measure")
    plt.plot(x, hist2.pdf(x), label="hist of perturbed measure")
    plt.legend(loc="upper right")
    fig.savefig('./imgs/algos/rwm/hist'+str(n)+'.png')
    plt.show()



elif test == "compare1":
    delta = 0.2
    gamma = 0.2
    epsilon = 1
    # Based on Theorem 2.2 in the vershynin paper
    # n_from_paper = np.ceil(2/(epsilon*delta) * len(test_functions)*np.log(len(test_functions)/gamma)).astype(int)
    #Ns = [int(np.exp(i)) for i in range(int(np.ceil(np.log(1000))), int(np.log(50000)))]
    Ns = [1000, 5000, 10000]
    print(Ns)
    database = data.schoel_balog_tostik_sample(max(Ns),1)
    database = database.reshape((len(database),))
    x = np.linspace(min(database), max(database), 1000)
    d = 1
    reps = 10
    all_RKHS_kme = []
    all_W1_rwm = []
    all_W1_phist = []
    all_W1_shist = []
    all_W1_pmm = []
    for rep in range(reps):
        if rep%3 == 0:
            print(rep)
        _, _, W1_rwm = rwm.asymptotic_rwm(Ns, x, database, epsilon)
        _, _, W1_phist = hist.asymptotic_perturbed(Ns, x, database, epsilon)
        _, _, W1_shist = hist.asymptotic_smooth(Ns, x, database, epsilon)
        rkhs_kme = kme.asymptotic_kme(Ns,1,epsilon)
        W1_pmm = pmm.asymptotic_pmm(Ns, x, database, epsilon)
        all_W1_rwm.append(W1_rwm)
        all_W1_phist.append(W1_phist)
        all_W1_shist.append(W1_shist)
        all_RKHS_kme.append(rkhs_kme)
        all_W1_pmm.append(W1_pmm)

    all_W1_rwm = np.mean(all_W1_rwm, axis=0)
    all_RKHS_kme = np.mean(all_RKHS_kme, axis=0)
    all_RKHS_kme = all_RKHS_kme.reshape((len(all_RKHS_kme),))
    all_W1_phist = np.mean(all_W1_phist, axis=0)
    all_W1_shist = np.mean(all_W1_shist, axis=0)
    all_W1_pmm = np.mean(all_W1_pmm, axis=0)

    fig=plt.figure(figsize=(7,6))
    plt.loglog(Ns, all_W1_rwm, label="Private measure algo in W1", linestyle="solid")
    plt.loglog(Ns, all_RKHS_kme, label="KME Algo in RKHS dist", linestyle="dashed")
    plt.loglog(Ns, all_W1_phist, label="perturbed hist in W1", linestyle="dotted")
    plt.loglog(Ns, all_W1_shist, label="smooth hist in W1", linestyle="dashdot")
    plt.loglog(Ns, all_W1_pmm, label="PMM in W1", linestyle="--")
    plt.legend()
    fig.savefig('./imgs/algos/compareall_loglog.png', dpi=300, bbox_inches='tight')
    plt.show()

elif test == "show_pmm":
    n = 1000
    database = data.schoel_balog_tostik_sample(n, 1)
    r = int(np.log2(n))
    synthetic_data = pmm.pmm(database, depth=r)
    print(database.shape)
    print(synthetic_data.shape)
    x = np.linspace(min(database),max(database),1000)
    database_hist = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(len(database)))))
    sd_hist = stats.rv_histogram(np.histogram(synthetic_data, bins=int(np.sqrt(len(synthetic_data)))))
    fig=plt.figure(figsize=(7,6))
    plt.plot(x, database_hist.pdf(x), label="database")
    plt.plot(x, sd_hist.pdf(x), label="synthetic data")
    plt.legend()
    fig.savefig('./imgs/algos/pmm/hist'+str(r)+'.png', dpi=300, bbox_inches='tight')
    plt.show()

