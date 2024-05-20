# Scripts that implements the Hopfield Network Class
# from which to create instances of an Hopfield Network either with
# Orthogonal binary memory patterns
# Random binary memory patterns

# Author: Simone Betteti (2023)
# Paper: Stimulus-Driven Dynamics for Robust Memory Retrieval in Hopfield Networks

import numpy as np

class HNN:
    # Initializer for the Hopfield Class
    def __init__(self, C, eps):
        # condition for the generation of the memories as 
        # Orthogonal : C = O
        # Random : C = R
        self.Co = C
        # Definition of the population size as a power of 2 or by MNIST dimensionality
        self.N = 2**(np.random.randint(low=10,high=11))
        # Definition of the admissible number of memories
        #self.P = np.int(np.floor(self.N/10))
        self.P = 10
        #self.P = np.int(np.floor(self.N/(10*np.log(self.N))))
        # Definition of the magnitude of the perturbation of the initial condition
        self.eps = eps

    # Function for the generation of the population and the associated memories
    def net(self):
        if self.Co == 'O':
            C = np.ones((2,2))
            C[1,1] = -1
            C0 = np.copy(C)
            for j in range(np.int(np.log2(self.N/2))):
                C = np.kron(C,C0)
            C = C[:,1:]
        elif self.Co == 'R':
            C = np.random.choice([-1,1], (self.N,self.N))

        # Extraction of P patterns
        # Selection of P random non-repeated indices
        idx_b = np.random.permutation(range(self.N-1))
        idx = idx_b[:self.P]

        # Generation of the memories
        self.mems = C[:,idx]

    # Function that generates the initial condition for the dynamics
    def y0_gen(self):
        # random extraction of an index
        #idx = np.random.randint(low=0,high=self.P)
        
        # generation of the initial condition
        #self.y0 = self.mems[:,idx] + self.eps*np.random.randn(np.shape((self.mems[:,idx]))[0],)
        self.y0 = np.random.randn(self.N)