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
def PlotOverlap(HN, EMA, y, T,col, n, sigma, In):

    # Definition of the matrix for memory-trajectory overlap
    M = np.zeros((HN.P,np.size(y[0,:])))

    # Definition of the time axis
    x = np.linspace(start=0, stop=T, num=np.size(y[0,:]))

    sig = np.tanh(10*y)

    # Computation of the single overlap values
    for p in range(HN.P):
        for dt in range(np.size(y[0,:])):
            M[p,dt] = (1/HN.N)*np.abs(np.dot(HN.mems[:,p],sig[:,dt]))

    # Saving of the correlation data
    if n > 1:
        if EMA.C == 'M':
            if sigma == 0:
                title = 'TrajectoriesMult_'+str(n)+'_'+In+'.csv'
            else:
                title = 'TrajectoriesMult_noise_'+str(n)+'_'+In+'.csv'
        else:
            if sigma == 0:
                title = 'TrajectoriesAdd_'+str(n)+'_'+In+'.csv'
            else:
                title = 'TrajectoriesAdd_noise_'+str(n)+'_'+In+'.csv'
    else:
        if EMA.C == 'M':
            title = 'TrajectoriesMult_PP'+str(n)+'_'+In+'.csv'
        else:
            title = 'TrajectoriesAdd_PP_'+str(n)+'_'+In+'.csv'

    M_df = pd.DataFrame(M)
    M_df.to_csv(title, index=False)

    # Plotting of the trajectory overlap
    if EMA.C == 'M':
        if sigma == 0:
            title = 'TrajectoryOverlap_Mult_'+str(n)+'_'+In+'.pdf'
        else:
            title = 'TrajectoryOverlap_Mult_noise_'+str(n)+'_'+In+'.pdf'
    elif EMA.C == 'A':
        if sigma == 0:
            title = 'TrajectoryOverlap_Add_'+str(n)+'_'+In+'.pdf'
        else:
            title = 'TrajectoryOverlap_Add_noise_'+str(n)+'_'+In+'.pdf'
    if n > 1:
        if EMA.C == 'M':
            fig = plt.figure(figsize=(35,13))
            gs = fig.add_gridspec(13,1)
            ax1 = fig.add_subplot(gs[:9,:])
            for p in range(HN.P-1,-1,-1):
                if p>2:
                    txt = 'Other Memories'
                    ax1.plot(x[1:],M[p,1:], color = 'orange', alpha=0.5, linewidth=4.0, linestyle='dotted', markeredgecolor='black')
                else:
                    ax1.plot(x[1:],M[p,1:], color=col[p], linewidth=6.0)   
            ax1.set_xlabel(r'$t$')
            ax1.set_ylabel(r'$(\xi,y(t))$')
            ax2 = fig.add_subplot(gs[11:,:])
            ax2.plot(x[1:1000],0.15*np.ones((999,))+0.01*np.random.randn(999),color=col[0], linewidth=10.0)
            ax2.plot(x[1000:2000],0.15*np.ones((1000,))+0.01*np.random.randn(1000),color=col[1],linewidth=10.0)
            ax2.plot(x[2000:3000],0.15*np.ones((1000,))+0.01*np.random.randn(1000),color=col[2],linewidth=10.0)
            ax2.set_ylim(bottom=0,top=0.3)
            ax2.set_axis_off()
            ax2.set_title('Dominant Memory', fontweight='bold')
        else:
            fig = plt.figure(figsize=(35,8))
            for p in range(HN.P-1,-1,-1):
                if p>2:
                    txt = 'Other Memories'
                    plt.plot(x[1:],M[p,1:], color = 'orange', alpha=0.5, linewidth=4.0, linestyle='dotted', markeredgecolor='black')
                else:
                    plt.plot(x[1:],M[p,1:], color=col[p], linewidth=6.0)  
            plt.ylim(bottom=0,top=1) 
            plt.xlabel(r'$t$')
            plt.ylabel(r'$(\xi,y(t))$')
    else:
        if sigma !=0:
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
            ax1.set_ylim(bottom=0,top=1) 
            ax1.set_xlabel(r'$t$')
            ax1.set_ylabel(r'$(\xi,y(t))$')
            ax2 = fig.add_subplot(gs[11:,:])
            ax2.plot(x[1:800],0.15*np.ones((799,))+0.01*np.random.randn(799),color=col[0], linewidth=10.0)
            ax2.plot(x[800:1000],0.15*np.ones((200,))+0.01*np.random.randn(200),color=col[1],linewidth=10.0)
            ax2.plot(x[1000:1500],0.15*np.ones((500,))+0.01*np.random.randn(500),color=col[0],linewidth=10.0)
            ax2.plot(x[1500:1900],0.15*np.ones((400,))+0.01*np.random.randn(400),color=col[1],linewidth=10.0)
            ax2.plot(x[1900:2500],0.15*np.ones((600,))+0.01*np.random.randn(600),color=col[2],linewidth=10.0)
            ax2.set_ylim(bottom=0,top=0.3)
            ax2.set_axis_off()
            ax2.set_title('Dominant Memory', fontweight='bold')
        else:
            fig = plt.figure(figsize=(35,13))
            gs = fig.add_gridspec(13,1)
            ax1 = fig.add_subplot(gs[:,:])
            for p in range(HN.P-1,-1,-1):
                if p>2:
                    txt = 'Other Memories'
                    ax1.plot(x[1:],M[p,1:], color = 'orange', alpha=0.5, linewidth=4.0, linestyle='dotted', markeredgecolor='black')
                else:
                    ax1.plot(x[1:],M[p,1:], color=col[p], linewidth=6.0)  
            ax1.set_ylim(bottom=0,top=1) 
            ax1.set_xlabel(r'$t$')
            ax1.set_ylabel(r'$(\xi,y(t))$')
    plt.savefig(title, bbox_inches = 'tight')
    plt.close()

