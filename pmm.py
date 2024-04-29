import numpy as np
from binary_tree import Node
import metrics
import time
from histogram import Histogram
import data_generator as data

def pmm(database, depth=3, dim=1, noise_levels=[], epsilon=1):
    start_time = time.time()
    boundaries = [[0,1] for _ in range(dim)]
    #mins = np.min(database, axis=0)
    #maxs = np.max(database, axis=0)
    #for d in range(dim):
    #    boundaries.append((float(mins[d]), float(maxs[d])))

    boundaries = np.array(boundaries)
    
    #noise_levels = manual_noise_levels(depth, epsilon, boundaries)
    #Create binary partition
    root = Node(
        index = [],
        bounds = boundaries,
        level=0,
    )
    #print(boundaries)
    # Partition (and create) all
    root.partition(depth=depth)
    # Count the amount of points top-down for each node
    count = root.count_true(database, dim=dim)
    #print(str(count) + " vs. " + str(len(database)))
    # ...
    # Create noise from those boundaries
    if noise_levels == []:
        all_boundaries = root.collect_boundaries()
        #print("set noise levels")
        #if dim == 1:
        #    noise_levels = get_noise_levels(depth, epsilon)
        #else:
        noise_levels = manual_noise_levels(depth, epsilon, all_boundaries, dim=dim)
    #print("-----NOISE-----")
    #print(noise_levels)
    #print(len(noise_levels))
    #print(depth)
    #print("-----END-----")
    # Add noise top-down for each node
    #print("Add noise")
    root.add_noise(noise_levels)
    # Enforce consistency of the vectors
    #print("Enforce consistency")
    root.enforce_consistency()
    # get all consistent counts
    # Note: this step is useless!
    #counts = root.get_all_leaf_counts(r)
    # sample uniformly from these bins

    synthetic_data = root.sample_from_leaves(depth=depth)
    #print("SYNTHETIC DATA")
    #print(synthetic_data)
    stop_time = time.time() - start_time
    return stop_time, synthetic_data

def get_noise_levels(r, epsilon):
    S = 2*np.sqrt(2)
    deltas = [np.sqrt(2), np.sqrt(2)]
    for j in range(2, r+1):
        if j%2 == 0:
            del_j = np.sqrt( 2**j * np.sqrt(2**(-2*(j-2)) + 2**(-2*(j-1))) ) 
            deltas.append(del_j)
            S += del_j
        else:
            del_j = np.sqrt(2**j * np.sqrt(2*2**(-2*(j-2))))
            deltas.append(del_j)
            S += del_j

    noise_levels = []
    for j in range(r+1):
        sigma_j = S / (epsilon * deltas[j])
        noise_levels.append(sigma_j)
    
    return noise_levels


def manual_noise_levels(r, epsilon, all_bounds, dim=1):
    deltas = [1, 1]
    S = sum(deltas)
    for j in range(2, r+1):
        bounds = all_bounds[j-1]
        #print(np.linalg.norm(bounds[:,1]-bounds[:,0], ord=np.inf))
        delta_j = np.sqrt(2**(j-1) * np.linalg.norm(bounds[:,1]-bounds[:,0], ord=np.inf))
        S += delta_j
        deltas.append(delta_j)

    noise_levels = []
    for j in range(0, r+1):
        sigma_j = S / (epsilon * deltas[j])
        noise_levels.append(sigma_j)

    #print(np.sum(np.reciprocal(noise_levels)))

    return noise_levels

def asymptotic_pmm(Ns, database=[], epsilon=0, dim=1):
    W1_dists = []
    L2_dists = []
    KS_dists = []
    times = []
    TF_dists = []
    database = database.reshape((len(database), dim))
    for n in Ns:
        if epsilon == 0:
            epsilon = np.log2(n)
        if dim == 1:
            r = int(np.ceil(np.log2(epsilon*n)))-1
        else:
            r = int(np.ceil(np.log2(epsilon*n)))
        #print("Depth = " + str(r))
        # Stop the time
        n_time, synthetic_data = pmm(database[:n], depth=r, dim=dim, epsilon=epsilon)
        times.append(n_time)
        #print(synthetic_data.shape)
        # Measure
        w1 = metrics.multivW1(database[:n], synthetic_data, metric="chebyshev")
        #w1 = metrics.multivW1(database[:n], synthetic_data)
        W1_dists.append(w1)
        # INITIALISE HISTOGRAMS TO COMPUTE L2 AND KS DISTANCES and initialise m
        m = int(np.sqrt(n))
        Hist_DB = Histogram(database[:n], bin_amt=m, dim=dim, delta=0)
        Hist_SD = Histogram(synthetic_data, bin_amt=m, dim=dim, delta=0)
        # COMPUTE L2 DISTANCE
        L2 = metrics.smartL2_hypercube(Hist_DB.probabilities, Hist_SD.probabilities, m)
        L2_dists.append(L2)
        # COMPUTE KS DISTANCE
        KS = metrics.smartKS_hypercube(Hist_DB.probabilities, Hist_SD.probabilities, m, dim=dim)
        KS_dists.append(KS)
        # 
        test_functions = data.get_binary_testfunctions_upto(dimension=dim, max_order=False)
        TF = metrics.wrt_marginals(test_functions=test_functions, sample1=database[:n], sample2=synthetic_data, dim=dim)
        TF_dists.append(TF)

    #print(TF_dists)

    return times, KS_dists, L2_dists, W1_dists, TF_dists



def transform_vector(noisy_children=(0,0)):
    consistent_children = ()

    return consistent_children
