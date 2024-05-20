# Scripts that implements the Hopfield Network Class
# from which to create instances of an Hopfield Network either with
# Orthogonal binary memory patterns
# Random binary memory patterns

# Author: Simone Betteti (2023)
# Paper: Stimulus-Driven Dynamics for Robust Memory Retrieval in Hopfield Networks

import numpy as np

class HNN:
    # Initializer for the Hopfield Class
    def __init__(self, p, eps):
        # Definition of the population size as a power of 2
        self.N = 2**(np.random.randint(low=10,high=11))
        # Definition of the probability of activation
        self.p = p
        # Definition of the admissible number of memories
        #self.P = np.int(np.floor(self.N/10))
        self.P = 10
        # Definition of the magnitude of the perturbation of the initial condition
        self.eps = eps

    # Function for the generation of the population and the associated memories
    def net(self):
        
        C = np.random.choice([0,1], (self.N,self.N), p=[1-self.p, self.p])

        # Extraction of P patterns
        # Selection of P random non-repeated indices
        idx_b = np.random.permutation(range(self.N-1))
        idx = idx_b[:self.P]

        # Generation of the memories
        self.mems = C[:,idx]


    # Function that directly builds the coupling matrix given the memories
    def W_block(self):
        
        self.W = (1/(self.p*(1-self.p)*self.N))*(self.mems-self.p)@np.transpose(self.mems-self.p)
        idx = range(self.N)
        self.W[idx,idx] = 0

    # Function that generates the initial condition for the dynamics
    def y0_gen(self):
        # random extraction of an index
        idx = np.random.randint(low=0,high=self.P)
        # generation of the initial condition
        #self.y0 = self.mems[:,idx] + self.eps*np.random.randn(np.shape((self.mems[:,idx]))[0],)
        self.y0 = np.multiply(self.mems[:,0],np.random.choice([0,1], size=(self.N,), p=[0.5, 0.5]))