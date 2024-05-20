# Script that, given an Hopfield Model, implements the (stochastic)
# Euler-Mayorama method of integration for the solution of a given SDE
# time [0,T]

# Author: Simone Betteti (2023)
# Paper: Stimulus-Driven Dynamics for Robust Memory Retrieval in Hopfield Networks

import numpy as np
from tqdm import tqdm

class EM:
    # Initialization of the integrator
    def __init__(self, HN, t_ini, t_end, dt, C, ini):
        # definition of the initial integration time
        self.t0 = t_ini
        # definition of the final integration time
        self.T = t_end
        # definition of the step size
        self.dt = dt
        # slope of the activation function
        self.delta = 10
        # Tolerance for the equilibrium condition
        self.tol = 10**(-9)
        # Definition of the timescales for the two layers
        self.tau1 = 1
        self.tau2 = 1
        # Integration condition
        # C == 'M' --> Multiplicative
        # C == 'A' --> Additive
        self.C = C
        self.y0 = ini
    
    # Function that implements the Euler Mayorama method given an SDE
    def Eu_Ma_Test(self,HN, sigma, u):
        
        # Perturbation
        self.sigma = sigma
        # Definition of the integration interval, appropriately partitioned
        x = np.arange(start=self.t0, stop=self.T, step=self.dt)
        # defintion of the solution vector
        self.y = np.zeros((HN.N,np.size(x)))
        # definition of the activation vector
        self.z = np.zeros((HN.N,np.size(x)))


        # setting of the initial condition
        self.y[:,0] = self.y0
        # setting of the transformed initial condition
        self.z[:,0] = self.g_fun_sig(self.y[:,0], HN)
        # Dynamic Threshold parameter
        c = np.zeros((HN.P,))
        # Solution of the system using Euler-Mayorama method
        for t in tqdm(range(np.size(x)-1)):
            if self.C == 'M':
                try:
                    I = u[:,t]
                except:
                    I = u
            elif self.C == 'A':
                I = np.zeros((HN.N,))
                try:
                    if x[t]<2 or (x[t]>10 and x[t]<14) or (x[t]>20 and x[t]<22):
                        I = u[:,t]
                except:
                    if x[t]<2:
                        I = u
            # Computation of the value field
            Hf = self.hop_field_test(self.y[:,t],HN, I)
            # Computation of the Brownian increments
            dW = np.random.randn(np.shape(self.y[:,t])[0],)
            # Computation of the field updated value
            self.y[:,t+1] = self.y[:,t] + Hf*self.dt + np.sqrt(self.dt)*sigma*dW
            self.z[:,t+1] = self.g_fun_sig(self.y[:,t+1], HN)


    # Function that implements the positive activation function
    def g_fun_sig(self, y, HN):
    
        # Inhibition by mean activation value
        if np.size(y)>1:
            v = (1/HN.p)*np.ones((int(HN.N),))*np.mean(y)
            z = 1/(1+np.exp(self.delta*(-2*y+1+0*v)))
        else:
            z = 1/(1+np.exp(self.delta*(-2*y+1)))

        return z
    
    # Function that implements the positive activation function
    def g_fun_sig_bis(self,y, HN):

        y[y<0] = 0
        z = 1/(1+np.exp(self.delta*(-2*y+1)))

        return z
    
    # Function that implements the computation of the Hopfield-field (hehe)
    def hop_field_test(self, y, HN, u):
        # Filtered activation
        z = self.g_fun_sig(y, HN)
        # Construction of the interaction matrix
        B = range(HN.N)
        if self.C == 'M':
            Dalph = (1/(HN.N*HN.p))*np.diagflat(np.dot(np.transpose(HN.mems-HN.p),u))
            #W = (alpha/(HN.N))*(HN.mems-beta)@Dalph@np.transpose(HN.mems-beta)+(gamma/HN.N)*np.ones((HN.N,HN.N))
            W = (1.25/(HN.N*HN.p*(1-HN.p)))*(HN.mems-HN.p)@Dalph@np.transpose(HN.mems-HN.p)-(1/(HN.N*HN.p))*np.ones((HN.N,HN.N))
            W[B,B] = 0
            # Value of the field
            val = -y + self.g_fun_sig((W@y), HN)
        elif self.C == 'A':
            # Value of the field
            val = -y + self.g_fun_sig((HN.W@y + u), HN)

        return val

