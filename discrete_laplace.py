import numpy as np
from scipy.stats import rv_discrete


class DiscreteLaplace(rv_discrete):
    "Discrete Laplace distribution"
    def _pmf(self, z, sigma):
        #print("inpdf: "+ str(sigma))
        p = np.exp(-1/sigma)
        return (1-p)/(1+p)*np.exp(-np.abs(z)/sigma)