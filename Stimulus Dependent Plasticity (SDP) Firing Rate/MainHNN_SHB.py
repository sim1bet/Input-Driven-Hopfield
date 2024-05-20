# Script for the implementation of an input driven firing rate model with 
# either only short term synaptic plasticity or both short and long
# term synaptic components

# Author: Simone Betteti (2023)
# Paper: Stimulus-Driven Dynamics for Robust Memory Retrieval in Hopfield Networks

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from HNN_Gen import HNN
from Eul_May import EM
from HNPlot import PlotOverlap

params = {'ytick.labelsize': 20,
          'xtick.labelsize': 35,
          'axes.labelsize' : 30,
          'font.size' : 20}
plt.rcParams.update(params)


os.system('cls')

# Define the perturbation around one of the memories
eps = 0.5
# Defines the scalar amplitude of the perturbations
sigma = 1*0.45

# Definition of the integration interval
t_ini = 0
t_end = 20
# Definition of the time step
dt = 0.01
# Probability of activation
pp = 0.22

# Generation of the single input window
x = np.arange(start=0,stop=t_end,step=dt)
T = np.size(x)

# Generation of the Hopfield model
HN = HNN(pp, eps)
# Generation of the memories
HN.net()
# Generation of the block matrix
HN.W_block()
# Generation of the initial condition
HN.y0_gen()

# Define the length of the input sequence
n = 3

# Definition of the inputs
u_proto = np.random.uniform(low=0.5,high=1.5,size=(HN.P,n))#low=0.01, high=0.08
for j in range(n):
        u_proto[j,j] = np.random.uniform(low=9,high=10)#low=2,high=3.5
        #if j>0:
        #     u_proto[j-1,j] = np.random.uniform(low=0,high=0.1)
        #nor = np.sum(u_proto[:,j])
        #u_proto[:,j] = u_proto[:,j]*np.sqrt(HN.P*HN.N)/(nor)

u_df = pd.DataFrame(u_proto)
u_df.to_csv('Alphas.csv', index=False)

# Plotting of the weigths for the inputs input
lbl = []
for p in range(HN.P):
    lbl.append(str(p))

# Definition of the colormap for the weights
col = ['cyan' for k in range(HN.P)]
col[0] = 'red'
col[1] = 'green'
col[2] = 'blue'
#col[3] = 'black'

#u = np.zeros((HN.N,n))
u_tran = np.zeros((HN.N,n))
# binary mask for the sparseness of activation
m = np.random.choice([0, 1], size=(HN.N,n), p=[0, 1])
for h in range(n):
    for l in range(HN.P):
        #u[:,h] += u_proto[l,h]*HN.mems[:,l]
        u_tran[:,h] += np.multiply(u_proto[l,h]*HN.mems[:,l],m[:,h])


for j in range(n):
    txt = "Weights Input: "+str(j+1)+'.eps'
    fig, ax = plt.subplots(figsize=(15,10))
    cntr, values, bars = ax.hist(x=lbl,bins=range(HN.P+1),weights=u_proto[:,j],align='left',color='orange',alpha=0.6,linewidth=2.0,edgecolor='black')
    bars[j].set_facecolor(col[j])
    bars[j].set_alpha(1)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\alpha_{\mu}$')
    plt.savefig(txt, bbox_inches='tight',format='eps')
    plt.close()

# Definition of the entire solution vector
Y = np.zeros((HN.N,T*n))
Y_add = np.zeros((HN.N,T*n))

for j in range(n):
    if j>0:
         buff = EMa.y[:,-1]
         buff_add = EMa_add.y[:,-1]
    else:
         buff = HN.y0
         buff_add = HN.y0
    # Generation of the Euler-Mayorama Integrator for the SDE
    EMa = EM(HN, t_ini, t_end, dt, 'M', buff)               # SDP firing rate model
    EMa_add = EM(HN, t_ini, t_end, dt, 'A', buff_add)       # Classic firing rate model
    # Integration of the system over the time interval
    EMa.Eu_Ma_Test(HN, sigma, u_tran[:,j])
    EMa_add.Eu_Ma_Test(HN, sigma, u_tran[:,j])

    Y[:,j*T:(j+1)*T] = EMa.z
    Y_add[:,j*T:(j+1)*T] = EMa_add.z

# Plotting of the overlap during training
PlotOverlap(HN, EMa, Y, n*t_end, col, n, sigma)
PlotOverlap(HN, EMa_add, Y_add, n*t_end, col, n, sigma)