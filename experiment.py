from scipy.optimize import linprog
from scipy.stats import wasserstein_distance
from scipy.optimize import minimize

import numpy as np
'''
Omega = np.arange(1,11,1)
#Omega_star = np.random.choice(Omega, size=4)
Omega_star = [4,5,6,7]
width = max(Omega_star) - min(Omega_star)
mOS = [-x for x in Omega_star]
OS = [x for x in Omega_star]
var = [(x-np.mean(Omega_star))**2 for x in Omega_star]
mvar = [-(x-np.mean(Omega_star))**2 for x in Omega_star]
one = [1 for _ in Omega_star]
print(Omega)
print(np.var(Omega))
print(np.var(Omega_star))

c = np.array([1]+[0 for _ in range(len(Omega_star))])

A_ub = np.array([[-1]+OS,[-1]+mOS, [-1]+one])
b_ub = np.array([np.mean(Omega), -np.mean(Omega), 1])
print(A_ub)
A_eq = np.array([[0]+[1 for _ in range(len(Omega_star))]])
b_eq = np.array([1])

minimum = 1/(2*len(Omega_star))
individual_bounds = [(0, None)] + [(minimum,1-minimum) for _ in range(len(Omega_star))]

result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=individual_bounds, method="highs-ipm")
print("----------------")
print(result.fun)
print(result.x)

print("----------------")
print(np.abs(np.mean(Omega)-np.mean(Omega_star)))
hstar = result.x[1:]
print(np.dot(Omega_star,hstar))
print(np.abs(np.dot(Omega_star,hstar)-np.mean(Omega)))
print("----------------")

print("Optimal W1: "+str(wasserstein_distance(Omega, Omega_star)))
print("W1 by algo: "+str(wasserstein_distance(Omega, Omega_star, v_weights=hstar)))

def w1(h):
    return wasserstein_distance(Omega, Omega_star, v_weights=h)

guess = [1/len(Omega_star) for _ in range(len(Omega_star))]
bnds = [(0,1) for _ in enumerate(Omega_star)]
cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
res = minimize(w1, guess, tol=1e-6, bounds=bnds, constraints=cons)
print(res.x)
print(res.fun)
print(sum(res.x))
'''

import numpy as np
import matplotlib.pyplot as plt

def haar_wavelet(x, k, j, s=1):
    """
    Computes the Haar wavelet function at point x for index (k, j) at scale s.
    k and j are integers representing translation and dilation respectively.
    """
    x_scaled = x / s
    return 2**(j/2) * (1 - 2*np.floor(2*x_scaled - k) + np.floor(2*x_scaled - k))

def haar_basis_functions(x, n):
    """
    Generates the Haar basis functions up to level n at the point x.
    """
    basis = []
    for j in range(n+1):
        for k in range(2**j):
            basis.append(haar_wavelet(x, k, j))
    return basis

def plot_haar_basis_functions(n):
    """
    Plots the Haar basis functions up to level n.
    """
    x = np.linspace(0, 1, 1000)
    basis = haar_basis_functions(x, n)

    plt.figure(figsize=(10, 6))
    for idx, func in enumerate(basis):
        plt.plot(x, func, label=f'Haar_{idx+1}(x)')
    plt.title(f'Haar Basis Functions up to level {n}')
    plt.xlabel('x')
    plt.ylabel('Haar(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
#level = 4
#plot_haar_basis_functions(level)

from itertools import product

#def KS(probabilities1, probabilities2, dim):
n = 10
dim=1
probabilities = np.full((n,dim),1/n)
probabilities = np.full((n,dim),1/n)
orig_space = np.arange(start=0, stop=n+1)
orig_space = orig_space/n
spaces = [orig_space for _ in range(dim)]
space = list(product(*spaces))
space_arr = np.array(space)
#for locations in space_arr:
#    if dim == 1:


m = 10
KS = 0
currsum = 0
i1, i2, i3 = 0, 0, 0
for i1 in range(m):
    for i2 in range(m):
        for i3 in range(m):
            currsum += probabilities1[i1,i2,i3] - probabilities2[i1,i2,i3]
            if currsum >= KS:
                KS = currsum

