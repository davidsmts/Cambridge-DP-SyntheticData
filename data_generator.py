import numpy as np
import scipy.stats as stats
import folktables
import itertools



ACSIncomeBin = folktables.BasicProblem(
    features=[
        "SEX", "DIS", "NATIVITY","DEYE","RAC1P"
    ],
    target='PINCP',
    target_transform=lambda x: x > 25000,    
    group='RAC1P',
    preprocess=folktables.adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSPubCov_Manual = folktables.BasicProblem(
    features=[
        "AGEP", "COW", "SCHL","RELP","RAC1P"
    ],
    target='PUBCOV',
    target_transform=lambda x: x == 1,    
    group='RAC1P',
    preprocess=folktables.public_coverage_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)


def getACS(dim=1, amt=100000):
    data_source = folktables.ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA", "TX"], download=True)
    features, label, group = ACSPubCov_Manual.df_to_numpy(acs_data)
    features = features[:,:dim]
    #maxs = np.max(features, axis=0)
    rescaled_features = (features - np.min(features)) / (np.max(features) - np.min(features))
    rng = np.random.default_rng()
    rng.shuffle(rescaled_features, axis=1)
    #rescaled_features = rescaled_features.T
    return rescaled_features[:amt]

def getBinACS(dim=1, amt=100000):
    data_source = folktables.ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA","TX"], download=True)
    #bin_features = ["SEX", "DEAR", "DEYE","NATIVITY"]
    features, label, group = ACSIncomeBin.df_to_numpy(acs_data)
    if dim <= 4:
        features = features[:,:dim]
    else: 
        print("dim too high !")
        return
    rng = np.random.default_rng()
    rng.shuffle(features, axis=1)
    features = features - 1
    return features[:amt]

#
def laplacian_noise(loc, sigma, shape=None):
    return np.random.laplace(loc, sigma, shape)


#
def manual_discrete_laplacian_noise(sigma, true_count=2):
    p = np.exp(-1/sigma)
    f = lambda x: (1-p)/(1+p)*np.exp(-np.abs(x)/sigma)
    values = [z for z in range(-true_count, true_count+1, 1)]
    #print(values)
    probabilities = f(values)
    #print(probabilities)
    probabilities = probabilities / np.sum(probabilities)
    choice = np.random.choice(values, 1, p=probabilities)
    return choice[0]


#
def gaussian_noise(N, L2_sensitivity, epsilon, delta):
    variance = 2*np.log(1.25 / delta) * L2_sensitivity**2 / (epsilon**2 * N**2) 
    mean = 0
    gaussian_noise = np.random.normal(mean, variance, (N,1))
    return gaussian_noise


# Draw N vectors from the distribution made up of exponential mechanism with
# utility function "utility".
# Due to computatioal limits we have to discretize the space of possible values to choose, to R.
def exponential_mechanism(database, R, utility, N, sensitivity, epsilon):
    n = len(database)
    # compute g_x(r) for each r \in R
    rv_for_B = stats.rv_histogram(np.histogram(database, bins=int(np.sqrt(n)), density=False))
    B = max(rv_for_B.pdf(np.linspace(min(database), max(database), 100)))
    print("B = " + str(B))
    N = int(n**(2/3) * (3 * epsilon / B )**(2/3) )
    print(n)
    print(n**(2/3))
    probabilities = np.zeros(len(R))
    for i,r in enumerate(R):
        probabilities[i] = np.exp(-(epsilon*utility(database, r))/(2*sensitivity))
    probabilities = probabilities / np.sum(probabilities)
    R = R.reshape(probabilities.shape)

    rv1 = stats.rv_histogram(np.histogram(R, bins=int(np.sqrt(len(R))), weights=probabilities))
    # Since g_x(r) now represents the probability that a fixed r \in R will be picked,
    # we simply use np.random.chose
    #
    probabilities_on_R = rv1.pdf(R)
    probabilities_on_R = probabilities_on_R / np.sum(probabilities_on_R)
    indices = np.arange(0, len(R))
    #print(probabilities_on_R.shape)
    #print(indices.shape)
    synthetic_data_indices = np.random.choice(indices, size=(int(N),), p=probabilities_on_R)
    synthetic_data = R[synthetic_data_indices]
    return synthetic_data

# See Wassermann&Zhou paper chapter 4.
def WZ_Lap_noise(epsilon, amt, dim=1):
    if dim == 1:
        return laplacian_noise(0, 8/epsilon**2, (amt,))
    else:
        return laplacian_noise(0, 8/epsilon**2, tuple([amt for _ in range(dim)]))

#
def get_superregular_rw(n, noise=[]):
    if len(noise)==0 and n>0:
        noise = laplacian_noise(0,np.log(n), (n,))
    Z = eval_psis(n, noise)
    res = []
    summe = 0
    for i in range(0, n):
        summe += Z[i]
        res.append(summe)
    #return np.array(res)
    return Z
        

def eval_psis(n, noise):
    #ts = np.arange(1,n+1,0)
    base_vals = np.ones(n)
    DW_n = noise[0] * 1/n * base_vals
    DW_n = DW_n + noise[1] * 1/n * base_vals
    if n == 1: return DW_n
    maxpow = int(np.ceil(np.log2(n)))
    for j in range(2,maxpow):
        two_to_j = 2**j
        psi_j_of_t = base_vals*two_to_j/(2*n)
        interval_length = np.ceil(n/two_to_j).astype(int)
        for i in range(2*two_to_j):
            index_from = i*interval_length
            index_until = (i+1)*interval_length
            sign = 1
            if i%2 != 0:
                sign = -1
            psi_j_of_t[index_from:index_until] = psi_j_of_t[index_from:index_until] * (sign)
        
        psi_j_of_t = psi_j_of_t * noise[j]
        DW_n += psi_j_of_t
    return DW_n


def plot_haar_basis_functions(j, n, ts):
    j = 2**np.ceil(np.log2(j))
    #print(j)
    j = int(j)
    psi_j = np.ones(len(ts))
    if j == 1:
        return psi_j*1/n
    elif j == 2:
        half = np.ceil(len(ts)/2).astype(int)
        psi_j[0:half] = psi_j[0:half]*1/n
        psi_j[half:] = -psi_j[half:]*1/n
        return psi_j
    
    psi_j =   psi_j * j/(2*n)
    interval_length = np.ceil(len(ts)/j).astype(int)
    for i in range(j):
        index_from = i*interval_length
        index_until = (i+1)*interval_length
        sign = 1
        if i%2 != 0:
            sign = -1
        else:
            psi_j[index_from] = np.NaN
            index_from += 1
        psi_j[index_from:index_until] = psi_j[index_from:index_until] * (sign)
        #        index_from = index_until
    return psi_j


def schoel_balog_tostik_sample(amt, dim=0):
    #np.random.multivariate_normal
    C = 10
    D = dim
    var_formeans = 200
    var_fordata = 30
    mu_means = 100.0 * np.ones(D)
    mus = np.random.normal(loc=mu_means, scale=var_formeans, size=(C, D))
    ws = np.array([1.0 / n for n in range(1, C+1)])
    ws /= np.sum(ws)
    memberships = np.random.choice(range(C), size=amt, p=ws)
    database = []
    for n in range(amt):
        mu = mus[memberships[n]]
        sample = np.random.normal(loc=mu, scale=var_fordata)
        database.append(sample)
    database = np.array(database)
    database = (database - np.min(database)) / (np.max(database) - np.min(database)) 
    return database


def binary_binomial_sample(amt, dim=0):
    p = np.random.uniform(0,1,(1,))
    return np.random.binomial(1, p, (amt, dim))


def get_binary_testfunctions_upto(dimension=1, order=2, max_order=False):
    if max_order:
        order = dimension
    else:
        if dimension >= 2:
            order = 2
        else:
            order = dimension

    F = [[1 for _ in range(dimension)], [0 for _ in range(dimension)]]
    for i in range(1, order):
        to_permute = [1 for _ in range(i)] + [0 for _ in range(dimension - i)]
        marginals_of_order_i = list(itertools.permutations(to_permute))
        without_reps = np.unique(marginals_of_order_i, axis=0)
        F = np.concatenate((F, without_reps), axis=0)
        #print("F:")
        #print(to_permute)
    return np.array(F)


