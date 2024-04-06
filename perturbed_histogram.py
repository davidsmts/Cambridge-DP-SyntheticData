import numpy as np
import scipy.stats as stats

class PerturbedHistogram:

    def __init__(self, data, noise=[], bin_amt=-1) -> None:
        if bin_amt == -1:
            print("Setting a standard bin amount based on square root rule.")
            bin_amt = int(len(data)**(1/2))

        self.data = data
        self.bin_amt = bin_amt

        # check for correct noise
        if len(noise) != bin_amt:
            print("Provide as much noise as bins. Choosing Laplacian Noise with epsilon=1 according to the Wasserstein&Zhou paper now.")
            self.noise = np.random.laplace(0, 8, (bin_amt, 1))
        else:
            self.noise = noise

        # Calculate the perturbed histogram
        D, P = self.perturbBins()
        self.bins = D
        self.probabilities = P

        # Create a scipy.stats.rv_histogram to use its functions and properties
        self.rv = stats.rv_histogram((self.probabilities, self.boundaries))


        


    # Algorithm could be sped up by removing all elements that are already assigned to other bins
    # Then the extra step would boil down to adding all which are left to the last one
    def perturbBins(self):
        a = min(self.data)
        b = max(self.data)
        n = len(self.data)
        bin_width = (b-a)/self.bin_amt

        # Count number of sample points in bin j and find the bin of x
        bin_counts = []   # C_j in the Wassermann & Zhou paper
        boundaries = [a]
        for j in range(self.bin_amt-1):
            # count amount of sample points in this bin
            bin_count = 0
            for x_i in self.data:
                if a+j*bin_width <= x_i < a+(j+1)*bin_width:
                    bin_count += 1

            boundaries.append(a+(j+1)*bin_width)
            bin_counts.append(bin_count)

        # The following case is needed because the last step of for loop would otherwise not account for the case x_i==b 
        bin_count = 0
        for x_i in self.data:
            if a+(self.bin_amt-1)*bin_width <= x_i <= a+self.bin_amt*bin_width:
                bin_count += 1
        boundaries.append(a+self.bin_amt*bin_width)
        bin_counts.append(bin_count)

        # save the boundaries to use them in scipy.stats.rv_histogram later on
        self.boundaries = boundaries

        # perturb the histogram, get bin counts, get bin probabilities (both perturbed, of course).
        D = []  # these are the perturbed histogram counts
        p = []
        for j in range(self.bin_amt):
            D_j = bin_counts[j] + self.noise[j]
            D_j = D_j[0]
            D_j_tilde = max(D_j, 0)
            D.append(D_j_tilde)
            p_j = D_j_tilde / sum(bin_counts)
            p.append(p_j)
        
        return D, p
        
    def evaluate(self, x):
        return self.rv.pdf(x)
    
    def sample(self, amt):
        return self.rv.rvs(size=amt)