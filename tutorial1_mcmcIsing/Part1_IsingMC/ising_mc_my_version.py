##############################################################################
# ICTP-SAIFR Minicourse on Machine Learning for Many-Body Physics            #
# Roger Melko, Juan Carrasquilla and Lauren Hayward Sierens                  #
# Tutorial 1: Monte Carlo for the Ising model                                #
##############################################################################

"""
code to generate a markov chain monte carlo sampling of the square lattice
ising model at a give temperature
"""

import random
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

### Input parameters:
parser = argparse.ArgumentParser(description='input parameters')
parser.add_argument('-t', '--temp', metavar='', type=float, nargs='+',
                    default=np.linspace(5.0, 0.5, 19),
                    help='temperature of the system')
parser.add_argument('-L', metavar='', type=int, default=4,
                    help='linear size of the system')
parser.add_argument('-eq', metavar='', type=int, default=1000,
                    help='number of equilibration sweeps')
parser.add_argument('-b', '--n_bins', metavar='', type=int, default=500,
                    help='total number of measurement bins')
parser.add_argument('-s', '--sweepsPerBin', metavar='', type=int, default=50,
                    help='number of sweeps performed in one bin')
parser.add_argument('-a', '--animate', action='store_true',
                    help='enable animations')

args = parser.parse_args()

### Ising model parameters:
T_list = args.temp
L = args.L
N_spins = L**2    # total number of spins
J = 1    # coupling parameter
Tc = 2.0/np.log(1.0 + np.sqrt(2))*J    # critical temperature

### Monte Carlo parameters
n_eqSweeps = args.eq    # number of equilibration sweeps
n_bins = args.n_bins    # total number of measurement bins
n_sweepsPerBin = args.sweepsPerBin    # number of sweeps performed in one bin

### Files to write spin configurations (x) and phases (y):
train_frac = 0.7    # fraction of data to be used for training
file_xtrain = open('data/xtrain.txt', 'w')
file_ytrain = open('data/ytrain.txt', 'w')
file_xtest = open('data/xtest.txt', 'w')
file_ytest = open('data/ytest.txt', 'w')

### Parameters needed to show animation of spin configurations:
bw_cmap = colors.ListedColormap(['black', 'white'])

### Initially, the spins are in a random state (a high-T phase):
spins = np.zeros(N_spins, dtype=np.int)
for i in range(N_spins):
    spins[i] = 2*random.randint(0, 1) - 1    # either +1 or -1

### Store each spin's four nearest neighbours in a neighbours array
### (using periodic boundary conditions):
neighbours = np.zeros((N_spins, 4), dtype=np.int)

for i in range(N_spins):
    neighbours[i, 0] = i + 1    # neighbour to the right
    if i%L == (L-1):
        neighbours[i, 0] = i + 1 - L

    neighbours[i, 1] = i + L    # upwards neighbour
    if i >= (N_spins-L):
        neighbours[i, 1] = i + L - N_spins

    neighbours[i, 2] = i - 1    # neighbour to the left
    if i%L == 0:
        neighbours[i, 2] = i - 1 + L

    neighbours[i, 3] = i - L    # downwards neighbour
    if i <= (L-1):
        neighbours[i, 3] = i - L + N_spins
# end of for loop

def getEnergy():
    """Return energy of the spin configuration."""
    currEnergy = 0
    for i in range(N_spins):
        currEnergy += -J*(spins[i]*spins[neighbours[i, 0]] +
                          spins[i]*spins[neighbours[i, 1]])
    return currEnergy
# end of getEnergy() function

def getMag():
    """Return total magnetization of spin configuration."""
    return np.sum(spins)
# end of getMag() function

def sweep():
    """Perform one spin flip."""
    # do one sweep (N_spins local updates):
    for i in range(N_spins):
        deltaE = 0
        # randomly choose which spin to consider flipping:
        site = random.randint(0, N_spins-1)
         # calculate the change in energy of the proposed move
         # by considering only the nearest neighbours:
        for j in range(4):
            deltaE += 2*J*spins[site]*spins[neighbours[site, j]]
        # flip the spin according to detailed balance conditions
        if (deltaE <= 0) or (random.random() < np.exp(-deltaE/T)):
            spins[site] = -spins[site]
    # end loop over i
# end of sweep() function

def writeConfigs(num, T):
    """Function to write the training/testing data to file."""
    # determine whether the current configuration will be
    # used for training or testing:
    if num < (train_frac*n_bins):
        file_x = file_xtrain
        file_y = file_ytrain
    else:
        file_x = file_xtest
        file_y = file_ytest
    # multiply the configuration by +1 or -1 to ensure we
    # generate configurations with both positive and negative magnetization:
    flip = 2*random.randint(0, 1) - 1
    # loop to write each spin to a single line of the X data file:
    for i in range(N_spins):
        currSpin = flip*spins[i]
        # replace -1 with 0 (to be consistent with the desired format):
        if currSpin == -1:
            currSpin = 0
        file_x.write('%d  ' %(currSpin))
    # end loop over i
    file_x.write('\n')

    y = 0
    if T > Tc:
        y = 1
    file_y.write('%d \n' %y)
# end of writeConfigs(num,T) function

##############################################################################
#   Loop over all temperatures and perform Monte Carlo updates:              #
##############################################################################

for T in T_list:
    print('\nT = %f' %T)
    # File where observables will be recorded:
    fileName = 'data/ising2d_L%d_T%.4f.txt' %(L, T)
    with file_observales as open(fileName, 'w'):
        # Perform equilibration sweeps:
        for i in range(n_eqSweeps):
            sweep()

        # Start doing measurements:
        for i in range(n_bins):
            for j in range(n_sweepsPerBin):
                sweep()
            # end loop over j

            # Write the observables to file:
            energy = getEnergy()
            mag = getMag()
            file_observables.write('%d \t %.8f \t %.8f \n' %(i, energy, mag))

            # Write the x, y data to file:
            writeConfigs(i, T)

            # Display animation if args.animate is True
            if args.animate:
                # Display the current spin configuration:
                plt.clf()
                plt.imshow(spins.reshape((L, L)), cmap=bw_cmap,
                           norm=colors.BoundaryNorm([-1, 0, 1], bw_cmap.N))
                plt.xticks([])
                plt.yticks([])
                plt.title('%d x %d Ising model, T = %.3f' %(L, L, T))
                plt.pause(0.01)
            # end if

            if (i+1)%500 == 0:
                print('  %d bins complete' %(i+1))
        # end loop over i
# end loop over temperature
