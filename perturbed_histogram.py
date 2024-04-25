import numpy as np
import scipy.stats as stats

class PerturbedHistogram:

    def __init__(self, data, noise=[], bin_amt=-1) -> None:
        if bin_amt == -1:
            print("Setting a standard bin amount based on square root rule.")
            bin_amt = int(len(data)**(1/2))

        self.data = data
        self.bin_amt = bin_amt
        self.dim = data.shape[1]

        # check for correct noise
        if len(noise) != bin_amt:
            print("Provide as much noise as bins. Choosing Laplacian Noise with epsilon=1 according to the Wasserstein&Zhou paper now.")
            if self.dim == 1:
                self.noise = np.random.laplace(0, 8, (bin_amt,))
            else:
                self.noise = np.random.laplace(0, 8, (bin_amt for _ in range(self.dim)))
        else:
            self.noise = noise

        # Calculate the perturbed histogram
        D, P = self.perturbBins()
        self.bins = D
        self.probabilities = P

        # Create a scipy.stats.rv_histogram to use its functions and properties

        #self.rv = stats.rv_histogram((self.probabilities, self.boundaries))

        


    # Algorithm could be sped up by removing all elements that are already assigned to other bins
    # Then the extra step would boil down to adding all which are left to the last one
    def perturbBins(self):
        '''
        a = np.min(self.data, axis=0)
        b = np.max(self.data, axis=0)
        n = len(self.data)
        bin_widths = (b-a)/self.bin_amt

        
        # Count number of sample points in bin j and find the bin of x
        bin_counts = []   # C_j in the Wassermann & Zhou paper
        boundaries = [a]
        for j in range(self.bin_amt-1):
            # count amount of sample points in this bin
            bin_count = 0
            for x_i in self.data:
                all_in = True
                for l in range(len(a)):
                    if not (a[l]+j*bin_widths[l] <= x_i < a[l]+(j+1)*bin_widths[l]):
                        
                if all_in:
                    bin_count += 1

            boundaries.append(a+(j+1)*bin_widths)
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
        '''
        bin_amts = np.full((self.dim,), fill_value=self.bin_amt)
        true_counts, edges = np.histogramdd(self.data, bins=bin_amts)
        self.boundaries = np.array(edges)
        
        
        '''
        # perturb the histogram, get bin counts, get bin probabilities (both perturbed, of course).
        D = []  # these are the perturbed histogram counts
        p = []
        for d in range(self.dim-1):
            for j in range(self.bin_amt):
                print(self.dim)
                print(self.bin_amt)
                print(len(true_counts))
                print(self.noise.shape)
                print(true_counts.shape)
                print("d: "+str(d)+ " ; " + str(j))
                #xxeprint(true_counts[d][j])
                print(self.noise[j][d])
                print("")
                D_j = true_counts[d][j] + self.noise[j][d]
                D_j = D_j[0]
                D_j_tilde = np.max(D_j, 0)
                D.append(D_j_tilde)
                p_j = D_j_tilde / np.sum(true_counts)
                p.append(p_j)
        '''
        D = true_counts + self.noise
        D[D<0] = 0
        p = D / np.sum(D)
        
        return D, p
        
    def evaluate(self, x):
        flat_p = self.probabilities.flatten()
        choices = np.random.choice(np.arange(len(flat_p)), size=1)
        return self.rv.pdf(x)
    
    def sample(self, amt, dim=1):
        initial_shape = self.probabilities.shape
        flat_p = self.probabilities.flatten()
        choices = np.random.choice(np.arange(len(flat_p)), size=amt, p=flat_p)
        ddim_indices = self.get_indices(choices, dim=len(self.boundaries))
        points = self.get_points_from_indices(ddim_indices, dim)
        return points
    
    def get_indices(self, choices, dim):
        indices = []
        for choice in choices:
            index_k = []
            for k in range(0, dim):
                index_k.append(int(choice/(self.bin_amt**k))%self.bin_amt)
            #index_k.append(choice%self.bin_amt)
            indices.append(index_k)

        return indices
    
    def get_points_from_indices(self, indices, dim=1):
        points = []
        for j, index in enumerate(indices):
            #lb = []
            #ub = []
            point = []
            for i, ind_dim in enumerate(index):
                #lb.append(self.boundaries[i][ind_dim])                    
                #ub.append(self.boundaries[i][ind_dim+1])
                point_coord = np.random.uniform(self.boundaries[i][ind_dim], self.boundaries[i][ind_dim+1], 1)
                point.append(point_coord[0])
            # draw random point from that histogram bin!!!
            points.append(point)
        return np.array(points)
                