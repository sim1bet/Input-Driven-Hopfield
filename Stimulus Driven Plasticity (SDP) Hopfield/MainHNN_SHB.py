# Script for the implementation of an input driven Hopfield model with 
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

# Define the condition of memory generation
# C = O : Orthogonal Binary
# C = R : Random Binary
C = 'R'
# Define the perturbation around one of the memories
eps = 0.5
# Defines the scalar amplitude of the perturbations
sigma = 8

# Definition of the integration interval
t_ini = 0
t_end = 10
# Definition of the time step
dt = 0.01

# Generation of the single input window
x = np.arange(start=0,stop=t_end,step=dt)
T = np.size(x)

# Generation of the Hopfield model
HN = HNN(C,eps)
# Generation of the memories
HN.net()
# Generation of the initial condition
HN.y0_gen()
# Storing of the original value 
y_buff = HN.y0

# Define the length of the input sequence
n = 3

# Definition of the inputs
u_proto = np.random.uniform(low=0.8,high=1.5,size=(HN.P,n))
for j in range(n):
        u_proto[j,j] = np.random.uniform(low=2,high=3.5)
        if j>0:
             u_proto[j-1,j] = np.random.uniform(low=0.2,high=0.6)
        nor = np.sum(u_proto[:,j])
        u_proto[:,j] = u_proto[:,j]*np.sqrt(HN.P*HN.N)/(nor)

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
col[3] = 'black'

#u = np.zeros((HN.N,n))
u_tran = np.zeros((HN.N,n))
for h in range(n):
    for l in range(HN.P):
        #u[:,h] += u_proto[l,h]*HN.mems[:,l]
        u_tran[:,h] += u_proto[l,h]*HN.mems[:,l]


# Plotting of the weights alphas
for j in range(n):
    txt = "Weights Input: "+str(j+1)+'.pdf'
    fig, ax = plt.subplots(figsize=(15,10))
    cntr, values, bars = ax.hist(x=lbl,bins=range(HN.P+1),weights=u_proto[:,j],align='left',color='orange',alpha=0.6,linewidth=2.0,edgecolor='black')
    bars[j].set_facecolor(col[j])
    bars[j].set_alpha(1)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\alpha_{\mu}$')
    plt.savefig(txt, bbox_inches='tight',format='pdf')
    plt.close()

# Definition of the entire solution vector
Y = np.zeros((HN.N,T*n))
Y_add = np.zeros((HN.N,T*n))
Y_add_c = np.zeros((HN.N,T*n))

for j in range(n):
    # Generation of the Euler-Mayorama Integrator for the SDE
    EMa = EM(HN, t_ini, t_end, dt, 'M','c')                 # SDP Hopfield
    EMa_add = EM(HN, t_ini, t_end, dt, 'A','m')             # Classic Hopfield - Modulated Stimulus
    EMa_add_c = EM(HN, t_ini, t_end, dt, 'A','c')           # Classic Hopfield - Constant Stimulus
    # Integration of the system over the time interval
    if j>0:
         HN.y0 = y[:,-1]
    y = EMa.Eu_Ma_Test(HN, sigma, u_tran[:,j])
    if j>0:
         HN.y0 = y_add[:,-1]
    y_add = EMa_add.Eu_Ma_Test(HN, sigma, u_tran[:,j])
    if j>0:
         HN.y0 = y_add_c[:,-1]
    y_add_c = EMa_add_c.Eu_Ma_Test(HN, sigma, u_tran[:,j])

    Y[:,j*T:(j+1)*T] = y
    Y_add[:,j*T:(j+1)*T] = y_add
    Y_add_c[:,j*T:(j+1)*T] = y_add_c

# Plotting of the overlap during training
PlotOverlap(HN, EMa, Y, n*t_end, col, n, sigma, 'c')
PlotOverlap(HN, EMa_add, Y_add, n*t_end, col, n, sigma, 'm')
PlotOverlap(HN, EMa_add, Y_add_c, n*t_end, col, n, sigma, 'c')

# Definition of the integration interval
t_ini = 0
t_end = 25

# Generation of the single input window
x = np.arange(start=0,stop=t_end,step=dt)
T = np.size(x)

# Definition of the input
u_tot = np.zeros((HN.N,T))
u_tot[:,:800] = np.diag(u_tran[:,0])@np.ones((HN.N,800))
u_tot[:,800:1000] = np.diag(u_tran[:,1])@np.ones((HN.N,200))
u_tot[:,1000:1500] = np.diag(u_tran[:,0])@np.ones((HN.N,500))
u_tot[:,1500:1900] = np.diag(u_tran[:,1])@np.ones((HN.N,400))
u_tot[:,1900:2500] = np.diag(u_tran[:,2])@np.ones((HN.N,600))


# Re-definition of n for the continuous input
n = 1

# Definition of the entire solution vector
Y = np.zeros((HN.N,T*n))
Y_add = np.zeros((HN.N,T*n))
HN.y0 = y_buff

for j in range(n):
    # Generation of the Euler-Mayorama Integrator for the SDE
    EMa = EM(HN, t_ini, t_end, dt, 'M', 'c')
    EMa_add = EM(HN, t_ini, t_end, dt, 'A', 'm')
    # Integration of the system over the time interval
    y = EMa.Eu_Ma_Test(HN, sigma, u_tot)
    y_add = EMa_add.Eu_Ma_Test(HN, sigma, u_tot)

    Y[:,j*T:(j+1)*T] = y
    Y_add[:,j*T:(j+1)*T] = y_add


# Plotting of the overlap during training
PlotOverlap(HN, EMa, Y, n*t_end, col, n, sigma, 'c')
PlotOverlap(HN, EMa_add, Y_add, n*t_end, col, n, sigma, 'm')