###############################################################################
# ICTP-SAIFR Minicourse on Machine Learning for Many-Body Physics             #
# Roger Melko, Juan Carrasquilla and Lauren Hayward Sierens                   #
# Tutorial 1: Monte Carlo for the Ising lattice gauge theory                  #
###############################################################################

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import random
import argparse

### Input parameters:
parser = argparse.ArgumentParser(description='input parameters')
parser.add_argument('-t', '--temp', metavar='', type=float, nargs='+', default=np.linspace(5.0, 0.5,19), help='temperature of the system')
parser.add_argument('-L', metavar='', type=int, default=4, help='linear size of the system')
parser.add_argument('-eq', metavar='', type=int, default=1000, help='number of equilibration sweeps')
parser.add_argument('-b', '--n_bins', metavar='', type=int, default=500, help='total number of measurement bins')
parser.add_argument('-s', '--sweepsPerBin', metavar='', type=int, default=50, help='number of sweeps performed in one bin')
parser.add_argument('-a', '--animate',action='store_true', help='enable animations')

args = parser.parse_args()

### Ising model parameters:
T_list = args.temp    # temperature list
L = args.L    # linear size of lattice
N_sites = L**2    # total number of lattice sites
N_spins = 2*L**2    # total number of spins (one spin on each link)
J = 1    # coupling parameter
Tc = 1.0 # TODO: needs to be used to tell difference between T=0 and T=\inf configs


### Monte Carlo parameters:
n_eqSweeps = args.eq    # number of equilibration sweeps
n_bins = args.n_bins    #total number of measurement bins
n_sweepsPerBin = args.sweepsPerBin    # number of sweeps performed in one bin

### Files to write training and testing spin configurations (x) and phases (y):
train_frac = 0.7    # fraction of data to be used for training
file_xtrain = open('data/xtrain.txt', 'w')
file_ytrain = open('data/ytrain.txt', 'w')
file_xtest  = open('data/xtest.txt', 'w')
file_ytest  = open('data/ytest.txt', 'w')

### Parameters needed to show animation of spin configurations:
bw_cmap = colors.ListedColormap(['black', 'white'])

### Initially, the spins are in a random state (a high-T phase):
spins = np.zeros(N_spins,dtype=np.int)
for i in range(N_spins):
    spins[i] = 2*random.randint(0,1) - 1    # either +1 or -1

### Store each lattice site's four nearest neighbours in a neighbours array (using periodic boundary conditions):
neighbours = np.zeros((N_sites,4),dtype=np.int)
for i in range(N_sites):
    #neighbour to the right:
    neighbours[i,0] = i + 1
    if i%L==(L-1):
        neighbours[i,0] = i + 1 - L
    #upwards neighbour:
    neighbours[i,1] = i + L
    if i >= (N_sites-L):
        neighbours[i,1] = i + L - N_sites
    #neighbour to the left:
    neighbours[i,2] = i - 1
    if i%L==0:
        neighbours[i,2] = i - 1 + L
    #downwards neighbour:
    neighbours[i,3]= i - L
    if i <= (L-1):
        neighbours[i,3] = i - L + N_sites
#end of for loop

def getEnergy():
    """Return energy of spin configuration."""
    currEnergy = 0
    for i in range(N_sites):
        currEnergy += -J*getPlaquetteProduct(i)
    return currEnergy
#end of getEnergy() function

def getPlaquetteProduct(i):
    """Return product of spins in plaquette i"""
    return spins[2*i]*spins[2*i+1]*spins[2*neighbours[i,1]]*spins[2*neighbours[i,0]+1]

def sweep():
    """Do a Monte Carlo sweep (update)."""
    #do one sweep (N_spins local updates):
    for i in range(N_spins):
        #randomly choose which spin to consider flipping:
        spinLoc = random.randint(0,N_spins-1)
 
        #calculate the change in energy of proposed move by considering the two plaquettes it will affect:
        plaq1 = spinLoc//2

        #get plaq2 based on whether the spin is on a horizontal or vertical link
        if (spinLoc%2)==0:
            plaq2 = neighbours[plaq1,3]
        else:
            plaq2 = neighbours[plaq1,2]

        # energy change by the proposed spin flip
        deltaE = 2*J*(getPlaquetteProduct(plaq1) + getPlaquetteProduct(plaq2))
    
        # accept the flip according to detailed balance
        if (deltaE <= 0) or (random.random() < np.exp(-deltaE/T)):
            spins[spinLoc] = -spins[spinLoc]
    #end loop over i
#end of sweep() function

def writeConfigs(num,T):
    """Function to write the training/testing data to file."""
    # determine whether the current configuration will be used for training or testing:
    if num < (train_frac*n_bins):
        file_x = file_xtrain
        file_y = file_ytrain
    else:
        file_x = file_xtest
        file_y = file_ytest
    # multiply the configuration by +1 or -1 to ensure we generate configurations with both     positive and negative magnetization:
    flip = 2*random.randint(0,1) - 1 # TODO: is this gauge invariant?
    # loop to write each spin to a single line of the X data file:
    for i in range(N_spins):
        currSpin = flip*spins[i]
        # replace -1 with 0 (to be consistent with the desired format):
        if currSpin == -1:
            currSpin = 0
        file_x.write('%d  ' %(currSpin) )
    # end loop over i
    file_x.write('\n')

    y = 0
    if T>Tc:
        y = 1
    file_y.write('%d \n' %y)
 # end of writeConfigs(num,T) function


#################################################################################
########## Loop over all temperatures and perform Monte Carlo updates: ##########
#################################################################################
for T in T_list:
    print('T = %f' %T)

    # open a file where observables will be recorded:
    fileName = 'data/gaugeTheory2d_L%d_T%.4f.txt' %(L,T)
    file_observables = open(fileName, 'w')
 
    # equilibration sweeps:
    for i in range(n_eqSweeps):
        sweep()

    #start doing measurements:
    for i in range(n_bins):
        for j in range(n_sweepsPerBin):
            sweep()
        #end loop over j

        # Write observables to file:
        energy = getEnergy()
        file_observables.write('%d \t %.8f \n' %(i, energy))

        # write the x, y data to file:
        writeConfigs(i,T)

        if args.animate:
            # Display the current spin configuration:
            plt.clf()
            plt.imshow( spins.reshape((L,L)), cmap=bw_cmap, norm=colors.BoundaryNorm([-1,0,    1], bw_cmap.N) )
            plt.xticks([])
            plt.yticks([])
            plt.title('%d x %d gauge Ising model, T = %.3f' %(L,L,T))
            plt.pause(0.01)
        # end if


        if (i+1)%50==0:
            print('  %d bins complete' %(i+1))
    #end loop over i

    file_observables.close()
#end loop over temperature
