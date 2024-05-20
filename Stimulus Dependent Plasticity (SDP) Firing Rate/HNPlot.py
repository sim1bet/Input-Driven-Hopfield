# Script for the generation of the plots of activity for both the tensions/currents and
# the brownian motion associated to the timescales

# Author: Simone Betteti (2023)
# Paper: Stimulus-Driven Dynamics for Robust Memory Retrieval in Hopfield Networks

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

params = {'ytick.labelsize': 20,
          'xtick.labelsize': 35,
          'axes.labelsize' : 30,
          'font.size' : 20,
          'axes.titlesize': 20}
plt.rcParams.update(params)

# Function for the generation of the overlap graph
def PlotOverlap(HN, EMA, y, T,col, n, sigma):

    # Definition of the matrix for memory-trajectory overlap
    M = np.zeros((HN.P,np.size(y[0,:])))

    # Definition of the time axis
    x = np.linspace(start=0, stop=T, num=np.size(y[0,:]))

    # Computation of the single overlap values
    for p in range(HN.P):
        for dt in range(np.size(y[0,:])):
            M[p,dt] = (1/(HN.N*HN.p))*np.abs(np.dot(HN.mems[:,p],y[:,dt]))

    # Saving of the correlation data
    if n > 1:
        if EMA.C == 'M':
            if sigma == 0:
                title = 'TrajectoriesMult_'+str(n)+'.csv'
            else:
                title = 'TrajectoriesMult_noise_'+str(n)+'.csv'
        else:
            if sigma == 0:
                title = 'TrajectoriesAdd_'+str(n)+'.csv'
            else:
                title = 'TrajectoriesAdd_noise_'+str(n)+'.csv'
    else:
        if EMA.C == 'M':
            title = 'TrajectoriesMult_PP'+str(n)+'.csv'
        else:
            title = 'TrajectoriesAdd_PP_'+str(n)+'.csv'

    M_df = pd.DataFrame(M)
    M_df.to_csv(title, index=False)

    # Plotting of the trajectory overlap
    if EMA.C == 'M':
        title = 'TrajectoryOverlap_Mult_'+str(n)+'.pdf'
    elif EMA.C == 'A':
        title = 'TrajectoryOverlap_Add_'+str(n)+'.pdf'
    if n > 1:
        fig = plt.figure(figsize=(35,13))
        gs = fig.add_gridspec(13,1)
        ax1 = fig.add_subplot(gs[:9,:])
        for p in range(HN.P-1,-1,-1):
            if p>2:
                txt = 'Other Memories'
                ax1.plot(x[1:],M[p,1:], color = 'orange', alpha=0.5, linewidth=4.0, linestyle='dotted', markeredgecolor='black')
            else:
                #txt = "Memory"+str(p+1)
                ax1.plot(x[1:],M[p,1:], color=col[p], linewidth=6.0)  
        #ax1.set_ylim(bottom=0,top=1) 
        ax1.set_xlabel(r'$t$')
        ax1.set_ylabel(r'$(\xi,y(t))$')
        ax2 = fig.add_subplot(gs[11:,:])
        ax2.plot(x[1:2000],0.15*np.ones((1999,))+0.01*np.random.randn(1999),color=col[0], linewidth=10.0)
        ax2.plot(x[2000:4000],0.15*np.ones((2000,))+0.01*np.random.randn(2000),color=col[1],linewidth=10.0)
        ax2.plot(x[4000:6000],0.15*np.ones((2000,))+0.01*np.random.randn(2000),color=col[2],linewidth=10.0)
        ax2.set_ylim(bottom=0,top=0.3)
        ax2.set_axis_off()
        ax2.set_title('Dominant Memory', fontweight='bold')
    else:
        fig = plt.figure(figsize=(35,13))
        gs = fig.add_gridspec(13,1)
        ax1 = fig.add_subplot(gs[:9,:])
        for p in range(HN.P-1,-1,-1):
            if p>3:
                txt = 'Other Memories'
                ax1.plot(x[1:],M[p,1:], color = 'orange', alpha=0.5, linewidth=4.0, linestyle='dotted', markeredgecolor='black')
            else:
                #txt = "Memory"+str(p+1)
                ax1.plot(x[1:],M[p,1:], color=col[p], linewidth=6.0)  
        #ax1.set_ylim(bottom=0,top=1) 
        ax1.set_xlabel(r'$t$')
        ax1.set_ylabel(r'$(\xi,y(t))$')
        ax2 = fig.add_subplot(gs[11:,:])
        ax2.plot(x[1:1000],0.15*np.ones((999,))+0.01*np.random.randn(999),color=col[0], linewidth=10.0)
        ax2.plot(x[1000:1200],0.15*np.ones((200,))+0.01*np.random.randn(200),color=col[1],linewidth=10.0)
        ax2.plot(x[1200:2000],0.15*np.ones((800,))+0.01*np.random.randn(800),color=col[0],linewidth=10.0)
        ax2.plot(x[2000:2500],0.15*np.ones((500,))+0.01*np.random.randn(500),color=col[1],linewidth=10.0)
        ax2.plot(x[2500:3150],0.15*np.ones((650,))+0.01*np.random.randn(650),color=col[2],linewidth=10.0)
        ax2.plot(x[3150:3500],0.15*np.ones((350,))+0.01*np.random.randn(350),color=col[3],linewidth=10.0)
        ax2.plot(x[3500:],0.15*np.ones((500,))+0.01*np.random.randn(500),color=col[2],linewidth=10.0)
        ax2.set_ylim(bottom=0,top=0.3)
        ax2.set_axis_off()
        ax2.set_title('Dominant Memory', fontweight='bold')
    plt.savefig(title, bbox_inches = 'tight', format='pdf')
    plt.close()