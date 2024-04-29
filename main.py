import sys
import numpy as np
import matplotlib.pyplot as plt
import histogram as hist
import scipy.stats as stats
import scipy.linalg as linalg
import ot as pot
import data_generator as data
import subsampling_method as vershynin
import rwm
import pmm
import metrics
import utilities

np.random.seed(42)


colors=["blue", "orange", "green"]


#def show_privacy_guarantees(epsilon=1):



def compare_higherdim(Ns, display=True, reps=1, databasetype="SBT", dim=1, with_reference=False, loglog=True, distance="L2", epsilon=1):
    #x = np.linspace(min(database), max(database), 300)
    #all_W1_rwm = []
    all_phist = []
    all_shist = []
    all_pmm = []
    all_rwm = []
    all_subs = []
    for rep in range(reps):
        if databasetype == "ACS":
            database = data.getACS(dim=dim, amt=max(Ns))
            #database = database.T
        elif databasetype == "ACSBIN":
            database = data.getBinACS(dim=dim, amt=max(Ns))
            #database = database.T
        else:
            database = data.schoel_balog_tostik_sample(max(Ns), dim) 
            database = (database - np.min(database)) / (np.max(database) - np.min(database)) 
        #database = data.schoel_balog_tostik_sample(max(Ns),1)
        print("DB SHAPE" + str(database.shape))
        print("NONZERO ETNRIES? -> " + str(np.linalg.norm(database, ord=1)))

        # In DIM 1 execute SRWM ALGO
        if dim == 1:
            print("RWM")
            time_rwm, KS_rwm, L2_rwm, W1_rwm, TF_rwm = rwm.asymptotic_rwm(Ns, database, epsilon=epsilon, dim=dim)
        else: 
            time_rwm, KS_rwm, L2_rwm, W1_rwm, TF_rwm = [], [], [], [], []

        # ON BINARY DATA EXECUTE SUBSAMPLING ALGO
        if databasetype == "ACSBIN":
            time_subs, KS_subs, L2_subs, W1_subs, TF_subs = vershynin.asymptotic_subsampling_bin(Ns, database, epsilon, 0.2, 0.2, dim=dim)
        else:
            time_subs, W1_subs, L2_subs, KS_subs, TF_subs = [], [], [], [], []
        
        print("PHIST")
        time_phist, KS_phist, L2_phist, W1_phist, TF_phist = hist.asymptotic_perturbed(Ns, database=database, epsilon=epsilon, dim=dim)
        print("SHIST")
        time_shist, KS_shist, L2_shist, W1_shist, TF_shist = hist.asymptotic_smooth(Ns, database=database, epsilon=epsilon, dim=dim)
        #_, rkhs_kme, _  = kme.asymptotic_kme(Ks, database, dim=1, epsilon=epsilon)
        print("PMM")
        time_pmm, KS_pmm, L2_pmm, W1_pmm, TF_pmm = pmm.asymptotic_pmm(Ns, database=database, epsilon=epsilon, dim=dim)
        #all_W1_rwm.append(W1_rwm)
        if distance == "all":
            print("DISTANCE == all")
            all_phist.append([time_phist, KS_phist, L2_phist, W1_phist, TF_phist])
            all_shist.append([time_shist, KS_shist, L2_shist, W1_shist, TF_shist])
            all_pmm.append([time_pmm, KS_pmm, L2_pmm, W1_pmm, TF_pmm])
            all_rwm.append([time_rwm, KS_rwm, L2_rwm, W1_rwm, TF_rwm])
            all_subs.append([time_subs, KS_subs, L2_subs, W1_subs, TF_subs])
        elif distance == "L2":
            all_phist.append(L2_phist)
            all_shist.append(L2_shist)
            all_pmm.append(L2_pmm)
            all_rwm.append(L2_rwm)
        elif distance == "KS":
            all_phist.append(KS_phist)
            all_shist.append(KS_shist)
            all_pmm.append(KS_pmm)
            all_rwm.append(KS_rwm)
        elif distance == "W1": 
            all_phist.append(W1_phist)
            all_shist.append(W1_shist)
            all_pmm.append(W1_pmm)
            all_rwm.append(W1_rwm)
        elif distance == "TF": 
            all_phist.append(TF_phist)
            all_shist.append(TF_shist)
            all_pmm.append(TF_pmm)
            all_rwm.append(TF_rwm)
            all_subs.append(TF_subs)
        elif distance == "Time": 
            print("Showing runtime!")
            all_phist.append(time_phist)
            all_shist.append(time_shist)
            all_pmm.append(time_pmm)
            all_rwm.append(time_rwm)
            all_subs.append(time_subs)
        else:
            print("distance not implemented.")
            print("You might have forgotten to specify it!!")

    if dim==1:
        all_rwm = np.mean(all_rwm, axis=0)
    all_phist = np.mean(all_phist, axis=0)
    all_shist = np.mean(all_shist, axis=0)
    all_pmm = np.mean(all_pmm, axis=0)
    all_subs = np.mean(all_subs, axis=0)
    if distance == "all":
        print("DISTANCE == all")
        alldata = all_phist + all_shist + all_pmm
        testinfo = "ALL!! Ns:" + str(Ns) + " ; reps=" + str(reps) + " ; d = " + str(dim)
        utilities.write(alldata, filename="./logs/compareall_"+databasetype+"_"+str(dim)+"_loglog.txt", testinfo=testinfo)
        
        # plot them all:
        distance_types = ["Time", "KS", "L2", "W1"]
        if databasetype == "ACSBIN":
            distance_types.append("TF")

        for i,distance in enumerate(distance_types):
            specdist_phist = all_phist[i]
            specdist_shist = all_shist[i]
            specdist_pmm = all_pmm[i]
            specdist_rwm = []
            specdist_subs = []
            if dim == 1:
                specdist_rwm = all_rwm[i]
            if databasetype == "ACSBIN":
                specdist_subs = all_subs[i]
            if distance == "Time":
                loglog = False
            else:
                loglog = True

            plot_dist(Ns, dists=[specdist_phist, specdist_shist, specdist_pmm, specdist_rwm, specdist_subs], distance=distance, dim=dim, loglog=loglog, databasetype=databasetype, display=False)

    else:
        alldata = all_phist + all_shist + all_pmm
        print(alldata)
        testinfo = "Ns:" + str(Ns) + " ; reps=" + str(reps) + " ; d = " + str(dim)
        utilities.write(alldata, filename="./logs/compareall_"+databasetype+"_"+str(dim)+"_loglog.txt", testinfo=testinfo)
        plot_dist(Ns, [all_phist, all_shist, all_pmm, all_rwm, all_subs], dim=dim, loglog=loglog, databasetype=databasetype, display=display)


def plot_dist(Ns, dists=[], distance="L2", dim=1, loglog=True, databasetype="ACS", display=False):
    if len(dists) < 5:
        print("not enough distances supplied!")
        print("error")
        return
    all_phist = dists[0]
    all_shist = dists[1]
    all_pmm = dists[2]
    all_rwm = dists[3]
    all_subs = dists[4]
    fig=plt.figure(figsize=(7,6))
    #plt.loglog(Ns, all_W1_rwm, label="Random Walk Perturbed in W1", linestyle="solid")
    if loglog:
        plt.loglog(Ns, all_phist, label="P-Hist "+distance, linestyle="dotted", color="blue")
        plt.loglog(Ns, all_shist, label="S-Hist "+distance, linestyle="dashed", color="orange")
        plt.loglog(Ns, all_pmm, label="PMM "+distance, linestyle="solid", color="green")
        if dim==1: plt.loglog(Ns, all_rwm, label="RWM "+distance, linestyle="dashdot", color="lightgreen")
        if databasetype == "ACSBIN": plt.loglog(Ns, all_subs, label="Subsampling "+distance, linestyle=(0,(5,1)), color="red")
        #if with_reference:
            #plt.loglog(Ns, [n**(-1/(3+dim)) for n in Ns], label="n^(-1/(3+dim))", linestyle="dotted", color="blue")
            #plt.loglog(Ns, [n**(-dim/(2*dim+3)) for n in Ns], label="n^(-dim/(2*dim+3))", linestyle="dashed", color="orange")
            #plt.loglog(Ns, [n**(-1/dim) for n in Ns], label="n^(-1/d)", linestyle="solid", color="green")
    else:
        plt.plot(Ns, all_phist, label="P-Hist "+distance, linestyle="dotted", color="blue")
        plt.plot(Ns, all_shist, label="S-Hist "+distance, linestyle="dashed", color="orange")
        plt.plot(Ns, all_pmm, label="PMM "+distance, linestyle="solid", color="green")
        if dim==1: plt.plot(Ns, all_rwm, label="RWM "+distance, linestyle="dashdot", color="lightgreen")
        if databasetype == "ACSBIN": plt.plot(Ns, all_subs, label="Subsampling "+distance, linestyle=(0,(5,1)), color="red")
        #if with_reference:
        #    plt.plot(Ns, [n**(-dim/(2+dim)) for n in Ns], label="n^(-dim/(2+dim))", linestyle="dotted", color="blue")
        #    plt.plot(Ns, [n**(-dim/(2*dim+3)) for n in Ns], label="n^(-dim/(2*dim+3))", linestyle="dashed", color="orange")
        #    plt.plot(Ns, [n**(-1/dim) for n in Ns], label="n^(-1/d)", linestyle="solid", color="green")

    plt.xlabel("size of dataset")
    if distance == "Time":
        plt.ylabel(distance + " in seconds")
    else:
        plt.ylabel(distance + " distance between real and synthetic data")
    plt.title("dimension d="+str(dim))
    plt.legend()
    if loglog:
        fig.savefig('./imgs/algos/'+distance+'/comparehigherdim_loglog_'+databasetype+'_'+str(dim)+'_'+distance+'_woref.png', dpi=130, bbox_inches='tight')
    else:
        fig.savefig('./imgs/algos/'+distance+'/comparehigherdim_'+databasetype+'_'+str(dim)+'_'+distance+'_woref.png', dpi=130, bbox_inches='tight')
    if display:
        plt.show()
    else:
        plt.close()


# Reading and interpreting the command line arguments in the following code
arguments = sys.argv
if len(arguments) < 2:
    print("Too few arguments. Specify the test you want to do. E.g. 'histogram'")
    sys.exit()


test = str(arguments[1])
print(test)
if test == "show_regular":
    hist.show_histogram()

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

elif test == "asymptotic-smooth":
    Ns = [100, 1000, 6000, 10000, 20000] 
    databasetype = ""
    dim = 1
    epsilon = 1
    #reps = 5
    #print(database.shape)
    #print(database)
    #print(np.linalg.norm(database))
    for dim in range(1,4):
        database = data.getACS(dim=dim)
        database = database[:max(Ns),:]
        time_shist, KS_shist, L2_shist, W1_shist, TF_shist = hist.asymptotic_smooth(Ns, x=[], database=database, epsilon=epsilon, dim=dim)
        plt.plot(Ns, time_shist, label="L2, dim="+str(dim))
        #plt.plot(Ns, [2*n**(-1/(3+dim)) for n in Ns], label="n^(-1/(3+"+str(dim)+"))", linestyle="dotted")
    plt.legend()
    plt.show()

elif test == "s-hist-on-normal":
    hist.smooth_on_normal()

elif test == "p-hist-on-normal":
    hist.show_perturbed()

elif test == "show-subsampling":
    delta = 0.05
    gamma = 0.2
    epsilon = 1
    # Based on Theorem 2.2 in the vershynin paper
    # n_from_paper = np.ceil(2/(epsilon*delta) * len(test_functions)*np.log(len(test_functions)/gamma)).astype(int)
    d = 1
    n = 2000
    #database = data.schoel_balog_tostik_sample(n, d)
    database = data.getBinACS(d)
    database = database[:n]
    print(database)
    print("Shape of data")
    print(database.shape)
    #database = database.reshape((n,d))
    test_functions = vershynin.get_binary_testfunctions_upto(d)
    print(test_functions)
    #test_functions = vershynin.get_testfunctions_binary(d)
    print("|F|="+str(len(test_functions)))
    #
    vershynin.show_subsampling_mechanism(database, test_functions, delta, gamma)

elif test == "asymptotic-subsampling-bin":
    delta = 0.2
    gamma = 0.2
    epsilon = 1
    # Based on Theorem 2.2 in the vershynin paper
    # n_from_paper = np.ceil(2/(epsilon*delta) * len(test_functions)*np.log(len(test_functions)/gamma)).astype(int)
    Ns = [int(i*1e3) for i in range(1,8)]
    #Ns = [100, 500, 1000]
    print(Ns)
    dim = 1
    W1_dists = []
    L2_dists = []
    KS_dists = []
    TF_dists = []
    reps = 1
    for rep in range(reps):
        print("rep: " + str(rep))
        #database = data.binary_binomial_sample(max(Ns), d)
        database = data.getBinACS(dim=dim)
        database = database[:max(Ns)]
        W1, L2, KS, TF = vershynin.asymptotic_subsampling_bin(Ns, database, epsilon, delta, gamma, dim=dim)
        W1_dists.append(W1)
        L2_dists.append(L2)
        KS_dists.append(KS)
        TF_dists.append(TF)

    fig=plt.figure(figsize=(7,6))
    #plt.loglog(Ns, np.mean(W1_dists, axis=0), label="W1")
    print("------DISTS------")
    print(W1_dists)
    print(KS_dists)
    print(L2_dists)
    print(TF_dists)
    plt.loglog(Ns, np.mean(W1_dists, axis=0), label="W1")
    plt.loglog(Ns, np.mean(L2_dists, axis=0), label="L2")
    plt.loglog(Ns, np.mean(KS_dists, axis=0), label="KS")
    plt.loglog(Ns, np.mean(TF_dists, axis=0), label="TF")
    #plt.yticks([10**k for k in range(-5,1)])
    plt.legend()
    fig.savefig('./imgs/algos/subsampling/asymptotic_'+str(dim)+'.png', dpi=200, bbox_inches='tight')
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
    epsilon = 5
    #igN = 4000
    Ns = [100, 600, 1000, 2000, 5000]
    #database = data.getACS(1)
    #database = data.schoel_balog_tostik_sample(max(Ns),1)
    #database = database.reshape((len(database),))
    reps = 15
    all_KS = []
    all_L2 = []
    all_W1 = []
    for rep in range(reps):
        database = data.schoel_balog_tostik_sample(max(Ns),1)
        if rep%10 == 0:
            print(rep)
        x = np.linspace(min(database), max(database), 1000)
        _, KS, L2, W1 = rwm.asymptotic_rwm(Ns, x, database, epsilon)
        all_KS.append(KS)
        all_L2.append(L2)
        all_W1.append(W1)
    fig=plt.figure(figsize=(7,6))
    if len(arguments) > 2 and int(arguments[2]) == 1:
        #plt.plot(Ns, np.mean(all_KS, axis=0), label="KS", linestyle="dashed")
        #plt.plot(Ns, np.mean(all_L2, axis=0), label="L2", linestyle="dashed")
        plt.plot(Ns, np.mean(all_W1, axis=0), label="W1", linestyle="dotted")
        plt.plot(Ns, rwm.asymptotic_ub_acc(Ns, epsilon), label="log(n)^(3/2)/alpha")
        plt.legend()
        fig.savefig('./imgs/algos/rwm/asymptotic_rwm.png', dpi=300, bbox_inches='tight')
    else:
        #plt.loglog(Ns, np.mean(all_KS, axis=0), label="KS", linestyle="dashed")
        #plt.loglog(Ns, np.mean(all_L2, axis=0), label="L2", linestyle="dashed")
        plt.loglog(Ns, np.mean(all_W1, axis=0), label="W1", linestyle="dotted")
        plt.loglog(Ns, 100*rwm.asymptotic_ub_acc(Ns, epsilon), label="log(n)^(3/2)/alpha")
        plt.legend()
        fig.savefig('./imgs/algos/rwm/asymptotic_rwm_loglog_2.png', dpi=300, bbox_inches='tight')
    plt.show()

elif test == "show-rwmhist":
    n = int(arguments[2])
    epsilon = 1/n
    database = data.schoel_balog_tostik_sample(n,0)
    #database = data.getACS(dim=1)[:n]
    database = database.reshape((n,))
    x = np.linspace(min(database), max(database), 3000)
    hist1, hist2 = rwm.hist_rwm(n, x, database, epsilon)
    fig=plt.figure(figsize=(7,6))
    plt.plot(x, hist1.pdf(x), label="hist of std measure")
    plt.plot(x, hist2.pdf(x), label="hist of perturbed measure")
    plt.legend(loc="upper right")
    fig.savefig('./imgs/algos/rwm/hist'+str(n)+'.png')
    plt.show()


elif test == "show-rwm":
    n = int(arguments[2])
    rwm = data.get_superregular_rw(n)
    fig=plt.figure(figsize=(7,6))
    plt.plot(np.arange(0,n), rwm, label="hist of std measure")
    #plt.plot(x, hist2.pdf(x), label="hist of perturbed measure")
    plt.legend(loc="upper right")
    fig.savefig('./imgs/algos/rwm/rw_'+str(n)+'.png')
    plt.show()


elif test == "show_pmm":
    n = 1000
    database = data.schoel_balog_tostik_sample(n, 1)
    r = int(np.log2(n))
    time, synthetic_data = pmm.pmm(database, depth=r)
    print(database.shape)
    print(synthetic_data.shape)
    x = np.linspace(min(database),max(database),1000)
    database_hist = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(len(database)))))
    sd_hist = stats.rv_histogram(np.histogram(synthetic_data, bins=int(np.sqrt(len(synthetic_data)))))
    fig=plt.figure(figsize=(7,6))
    plt.plot(x, database_hist.pdf(x), label="database")
    plt.plot(x, sd_hist.pdf(x), label="synthetic data")
    plt.legend()
    fig.savefig('./imgs/algos/reference/pmm_hist'+str(r)+'.png', dpi=300, bbox_inches='tight')
    plt.show()


elif test == "compare_higherdim":
    dim = 1
    n = 20000
    reps = 2
    distance = "W1"
    databasetype = "SBT"
    #Ns = [int(np.exp(i)) for i in np.linspace(6, int(np.log(n)), 6)]
    Ns = [int(i) for i in np.linspace(1000, n, 8)]
    compare_higherdim(Ns, display=True, reps=reps, databasetype=databasetype, dim=dim, loglog=True, with_reference=True, distance=distance)

elif test == "perturbed_acc":
    hist.display_perturbed_accuracy()


elif test == "smooth_acc":
    hist.display_smooth_accuracy()


elif test == "newsmooth":
    hist.display_smooth_histogram()


elif test == "privacy":
    #
    #show_privacy_guarantees(epsilon=1)
    print("empty")


elif test == "rw_on_others":
    distances = ['L2', 'KS', 'W1']
    Ns = [100, 500, 800, 1000, 3000, 6000]
    epsilon = 1
    dim = 1
    database = data.getACS(dim=dim)
    time_shist, KS_shist, L2_shist, W1_shist, TF_shist = hist.asymptotic_smooth(Ns, database=database, epsilon=epsilon, dim=dim)
    time_phist, KS_phist, L2_phist, W1_phist, TF_phist = hist.asymptotic_perturbed(Ns, database=database, epsilon=epsilon, dim=dim)
    time_rwm_phist, KS_rwm_phist, L2_rwm_phist, W1_rwm_phist, TF_rwm_phist = hist.asymptotic_perturbed(Ns, database=database, epsilon=epsilon, dim=dim, rwm=True)
    time_pmm, KS_pmm, L2_pmm, W1_pmm, TF_pmm = pmm.asymptotic_pmm(Ns, database=database, epsilon=epsilon, dim=dim)
    time_rwm, KS_rwm, L2_rwm, W1_rwm, TF_rwm = rwm.asymptotic_rwm(Ns, database=database, epsilon=epsilon, dim=dim)
    for distance in distances:
        fig=plt.figure(figsize=(7,6))
        if distance == "KS":
            plt.loglog(Ns, KS_shist, label="SHist")
            plt.loglog(Ns, KS_phist, label="PHist")
            plt.loglog(Ns, KS_rwm_phist, label="Phist with rwm")
            plt.loglog(Ns, KS_pmm, label="PMM")
            plt.loglog(Ns, KS_rwm, label="SRWM")
            #plt.loglog(Ns, [min(np.log(n)/n**(2/(2+dim)), np.sqrt(np.log(n)/n)) for n in Ns], label="min(log(n)/n^(2/(2+d)), sqrt(log(n)/n))", linestyle="dotted")
        elif distance == "L2":
            plt.loglog(Ns, L2_shist, label="SHist")
            plt.loglog(Ns, L2_phist, label="PHist")
            plt.loglog(Ns, L2_rwm_phist, label="Phist with rwm")
            plt.loglog(Ns, L2_pmm, label="PMM")
            plt.loglog(Ns, L2_rwm, label="SRWM")
        elif distance == "W1":
            plt.loglog(Ns, W1_shist, label="SHist")
            plt.loglog(Ns, W1_phist, label="PHist")
            plt.loglog(Ns, W1_rwm_phist, label="Phist with rwm")
            plt.loglog(Ns, W1_pmm, label="PMM")
            plt.loglog(Ns, W1_rwm, label="SRWM")
            #plt.loglog(Ns, [n**(-2/(2+dim)) for n in Ns], label="n^(-2/(2+d))", linestyle="dotted")
        plt.legend()
        fig.savefig('./imgs/algos/rwm_phist/'+distance+'.png', dpi=130, bbox_inches='tight')
        plt.close()


elif test == "all_convergence":
    display = False
    reps = 5
    ''''''
    #Ns = [int(np.exp(i)) for i in np.linspace(6, int(np.log(n)), 8)]
    maxNs = [15000, 8000, 4000, 700]
    types = ["ACS", "SBT", "ACSBIN"]
    for databasetype in types:
        print("--------------")
        print("--------------")
        print("SWITCH TO "+databasetype)
        print("--------------")
        print("--------------")
        for dim in [1,3]:
            n = maxNs[dim-1]
            #Ns = [int(i) for i in np.linspace(100, n, 6)]
            Ns = [int(np.exp(i)) for i in np.linspace(np.log(300), np.log(n), 6)]
            print("--------------")
            print("dim = " +str(dim))
            print("--------------")
            compare_higherdim(Ns, display=display, reps=reps, databasetype=databasetype, dim=dim, loglog=True, with_reference=False, distance="all")


elif test == "all_supplementary":
    display = False

    # Visualisations
    #hist.display_perturbed_accuracy(display=display)
    #hist.display_smooth_accuracy(display=display)
    #hist.display_smooth_histogram(display=display)
    
    maxNs = [10000, 6000, 4000, 800]
    databasetype = "SBT"
    epsilon = 1
    reps = 5
    distances = ['L2', 'KS', 'W1']
    
    for dim in range(1,3):
        print("dim = " + str(dim))
        n = maxNs[dim-1]
        Ns = [int(np.exp(i)) for i in np.linspace(np.log(200), np.log(n), 6)]
        #database = database[:max(Ns),:]
        times, KSs, L2s, W1s, TFs = [], [], [], [], []
        for rep in range(reps):
            print(rep)
            database = data.getACS(amt=max(Ns), dim=dim)
            time_shist, KS_shist, L2_shist, W1_shist, TF_shist = hist.asymptotic_smooth(Ns, database=database, epsilon=epsilon, dim=dim)
            times.append(time_shist)
            KSs.append(KS_shist)
            L2s.append(L2_shist)
            W1s.append(W1_shist)
        L2 = np.mean(L2s, axis=0)
        KS = np.mean(KSs, axis=0)
        W1 = np.mean(W1s, axis=0)
        for distance in distances:
            fig=plt.figure(figsize=(7,6))
            if distance == "KS":
                plt.loglog(Ns, KS, label=distance+", dim="+str(dim))
                plt.loglog(Ns, [np.sqrt(np.log(n))/(n**(2/(6+dim))) for n in Ns], label="n^(-1/(3+"+str(dim)+"))", linestyle="dotted")
            elif distance == "L2":
                plt.loglog(Ns, L2, label=distance+", dim="+str(dim))
                plt.loglog(Ns, [2*n**(-2/5) for n in Ns], label="n^(-1/(3+"+str(dim)+"))", linestyle="dotted")
            else:
                plt.loglog(Ns, W1, label=distance+", dim="+str(dim))
                plt.loglog(Ns, [n**(-2/(7)) for n in Ns], label="n^(-2/7)", linestyle="dotted")
            plt.legend()
            fig.savefig('./imgs/algos/reference/SHist/'+distance+'_'+str(dim)+'.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("SWITCH TO PHIST")
    distances = ['L2', 'KS', 'W1']
    Ns = [100, 1000, 6000]
    for dim in range(1,3):
        print("dim = " + str(dim))
        database = data.getACS(dim=dim)
        database = database[:max(Ns),:]
        time_shist, KS_shist, L2_shist, W1_shist, TF_shist = hist.asymptotic_perturbed(Ns, database=database, epsilon=epsilon, dim=dim)
        for distance in distances:
            fig=plt.figure(figsize=(7,6))
            if distance == "KS":
                plt.loglog(Ns, KS_shist, label=distance+", dim="+str(dim))
                plt.loglog(Ns, [min(np.log(n)/n**(2/(2+dim)), np.sqrt(np.log(n)/n)) for n in Ns], label="min(log(n)/n^(2/(2+d)), sqrt(log(n)/n))", linestyle="dotted")
            else:
                plt.loglog(Ns, L2_shist, label="L2, dim="+str(dim))
                plt.loglog(Ns, [n**(-2/(2+dim)) for n in Ns], label="n^(-2/(2+d))", linestyle="dotted")
            plt.legend()
            fig.savefig('./imgs/algos/reference/PHist/'+distance+'_'+str(dim)+'.png', dpi=130, bbox_inches='tight')
            plt.close()
    '''
    print("SWITCH TO PMM")
    Ns = [100, 300, 500, 700]
    distance="W1"
    databasetype="ACS"
    reps = 2
    for dim in range(4,5):
        dists = []
        for rep in range(reps):
            print(rep)
            print("dim = " + str(dim))
            #database = data.schoel_balog_tostik_sample(max(Ns), dim) 
            #database = (database - np.min(database)) / (np.max(database) - np.min(database))
            database = data.getACS(dim=dim, amt=max(Ns))
            #database = database[:max(Ns),:]
            time_shist, KS_shist, L2_shist, W1_shist, TF_shist = pmm.asymptotic_pmm(Ns, database=database, epsilon=1, dim=dim)
            dists.append(W1_shist)
        dists = np.mean(dists, axis=0)
        fig=plt.figure(figsize=(7,6))
        plt.loglog(Ns, W1_shist, label=distance+", dim="+str(dim))
        if dim == 1:
            plt.loglog(Ns, [np.log2(n)**2/n for n in Ns], label="log2(n)**2/n", linestyle="dotted")
        elif dim == 2:
            plt.loglog(Ns, [n**(-1/2) for n in Ns], label="n^(-1/2)", linestyle="dotted")
        else:
            plt.loglog(Ns, [n**(-1/dim) for n in Ns], label="n^(-1/"+str(dim)+")", linestyle="dotted")
        plt.legend()
        fig.savefig('./imgs/algos/reference/PMM/'+distance+'_'+str(dim)+'.png', dpi=300, bbox_inches='tight')
        plt.close()
        '''


