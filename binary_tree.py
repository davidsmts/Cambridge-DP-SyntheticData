import numpy as np
import data_generator as data

class Node: 
    def __init__(self, index=[], bounds=[], true_count=0, noisy_count=0, consistent_count=0, level=0, depth=0):
        # this is not enough.
        # we need a name tag
        self.left = None
        self.right = None
        self.bounds = bounds
        self.index = index
        self.true_count = true_count
        self.noisy_count = noisy_count
        self.consistent_count = consistent_count
        self.level = level
        self.depth = depth


    def insert(self, data):
        # this is not enough.
        # need to pass name tags
        if self.data:
            if data < self.data:
                if self.left is None:
                    self.left = Node(data)
                else:
                    self.left.insert(data)
            elif data > self.data:
                if self.right is None:
                    self.right = Node(data)
                else:
                    self.right.insert(data)
        else:
            self.data = data


    def count_true(self, database, dim=1):
        dim_of_bounds = len(self.bounds)
        count = 0
        if dim != dim_of_bounds:
            print("ERROR! DIMENSION MISMATCH!")
            return False
        for x in database:
            inside = True
            for i in range(dim):
                if not (self.bounds[i][0] <= x[i] < self.bounds[i][1]):
                    inside = False
                    break
            if inside:
                count += 1

        self.true_count = count

        if self.left:
            self.left.count_true(database)
        if self.right:
            self.right.count_true(database)


    def add_noise(self, sigma):
        noise = data.discrete_laplacian_noise(sigma[0], shape=(1,), bounds=(-self.true_count-1 , self.true_count+1))
        #print("sigma: "+str(sigma[0])+"; noise: "+str(noise))
        self.noisy_count = max(0,self.true_count+noise)
        if self.left:
            self.left.add_noise(sigma[1:])
        if self.right:
            self.right.add_noise(sigma[1:])


    def halve_boundaries(self, left=True):
        j = self.level + 1
        dimensions_of_data = len(self.bounds)
        dim_to_halve = j % dimensions_of_data
        new_boundaries = []
        for dims in range(dimensions_of_data):
            if dims == dim_to_halve:
                halfway = (self.bounds[dims][1] - self.bounds[dims][0])/2
                if left:
                    new_boundaries.append( (self.bounds[dims][0], self.bounds[dims][1]-halfway) )
                else:
                    new_boundaries.append( (self.bounds[dims][0]+halfway, self.bounds[dims][1]) )
            else:
                new_boundaries.append( (self.bounds[dims][0], self.bounds[dims][1]) )

        return new_boundaries


    def partition(self, depth):
        self.depth = depth
        if depth > self.level:
            left_boundaries = self.halve_boundaries(left=True)
            right_boundaries = self.halve_boundaries(left=False)
            #print(left_boundaries)
            #print(right_boundaries)
            self.left = Node(self.index + [0], bounds = left_boundaries, level = (self.level + 1), depth = self.depth)
            self.right = Node(self.index + [1], bounds = right_boundaries, level = (self.level + 1), depth = self.depth)
            self.left.partition(depth=depth)
            self.right.partition(depth=depth)

    def find_closest_pair(self, m1, m2, m):
        possibilities = []
        possibilities_norm = []
        diff = m - m1 - m2
        sign = np.sign(diff)
        k = np.abs(diff)
        for i in range(np.abs(diff)+1):
            pair = (m1 + sign*i, m2 + sign*(k-i))
            #print(pair)
            #print("sum: "+ str(m - pair[0] - pair[1]))
            if pair[0] + pair[1] != m: print("Made a mistake, mate")
            possibilities.append(pair)
            possibilities_norm.append(np.linalg.norm(pair))
        minpair_index = np.argmin(possibilities_norm)
        return possibilities[minpair_index]

    def enforce_consistency(self):
        if self.level == 0:
            self.consistent_count = self.noisy_count
        if not ( self.left and self.right ): return
        c1 = self.left.noisy_count
        c2 = self.right.noisy_count
        diff = self.consistent_count - c1 - c2
        if diff != 0:
            #print("c1: "+str(c1)+ " ; c1: "+str(c2)+" ; ccount: "+str(self.consistent_count))
            c1, c2 = self.find_closest_pair(c1, c2, self.consistent_count)
        self.left.consistent_count = c1
        self.right.consistent_count = c2
        
        self.left.enforce_consistency()
        self.right.enforce_consistency()
        
    def get_all_leaf_counts(self, depth):
        res = []
        if depth == self.level+1:
            return [self.left.consistent_count, self.right.consistent_count]
        else: 
            res1 = self.left.get_all_leaf_counts(depth)
            res2 = self.right.get_all_leaf_counts(depth)
            return res1 + res2
        
    def sample_from_leaves(self, depth):
        if depth == self.level:
            sample = []
            for bounds in self.bounds:
                sample_for_dim = np.random.uniform(bounds[0], bounds[1], size=(self.consistent_count,))
                sample.append(sample_for_dim)
            sample = np.array(sample)
            return sample.T
        else: 
            s1 = self.left.sample_from_leaves(depth)
            s2 = self.right.sample_from_leaves(depth)
            return np.concatenate((s1, s2))