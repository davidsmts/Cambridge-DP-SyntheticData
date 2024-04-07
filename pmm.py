import numpy as np
from binary_tree import Node
import metrics

def pmm(database, depth=3, dim=1):
    boundaries = []
    for d in range(dim):
        boundaries.append((float(np.min(database, axis=d))-1e-4, float(np.max(database, axis=d))+1e-4))
    #Create binary partition
    r = depth # depth
    root = Node(
        index = [],
        bounds = boundaries,
        level=0,
    )
    #print(boundaries)
    # Partition (and create) all
    root.partition(depth=r)
    # Count the amount of points top-down for each node
    root.count_true(database)
    # Add noise top-down for each node
    print("Add noise")
    noise_levels = [(r+1-i)*(2/3) for i in range(r+1)]
    root.add_noise(noise_levels)
    # Enforce consistency of the vectors
    print("Enforce consistency")
    root.enforce_consistency()
    # get all consistent counts
    # Note: this step is useless!
    counts = root.get_all_leaf_counts(r)
    # sample uniformly from these bins
    synthetic_data = root.sample_from_leaves(depth=r)
    return synthetic_data

def asymptotic_pmm(Ns, x, database, epsilon, dim=1):
    W1_dists = []
    database = database.reshape((len(database), dim))
    for n in Ns:
        r = int(np.log2(n))
        synthetic_data = pmm(database[:n], depth=r)
        w1 = metrics.multivW1(database[:n], synthetic_data)
        W1_dists.append(w1)

    return W1_dists



def transform_vector(noisy_children=(0,0)):
    consistent_children = ()

    return consistent_children
