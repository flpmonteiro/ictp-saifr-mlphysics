################################################################################
#ICTP-SAIFR Minicourse on Machine Learning for Many-Body Physics               #
#Roger Melko, Juan Carrasquilla and Lauren Hayward Sierens                     #
#Tutorial 1: Print observables as a function of temperature for the Ising model#
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import sys

### Input parameters (these should be the same as in ising_mc.py):
T_list = np.linspace(5.0,0.5,19)    # temperature list
L_list = np.array([2,4,8,16])    # linear size of the lattice
J = 1    # coupling parameter

### Critical temperature:
Tc = 2.0/np.log(1.0 + np.sqrt(2))*J

### Observables to plot as a function of temperature:
energy = [[] for i in L_list]
mag = [[] for i in L_list]
specHeat = [[] for i in L_list]
susc = [[] for i in L_list]

for i in range(len(L_list)):
    L = L_list[i]
    N_spins = L**2    #  total number of spins
    ### Loop to read in data for each temperature:
    for T in T_list:
        file = open('data/ising2d_L%d_T%.4f.txt' %(L,T), 'r')
        data = np.loadtxt(file)

        E = data[:,1]
        M = abs(data[:,2])

        energy[i].append(np.mean(E)/(1.0*N_spins))
        mag[i].append(np.mean(M)/(1.0*N_spins))
        specHeat[i].append((np.mean(E**2) - np.mean(E)**2)/(1.0*T**2*N_spins))
        susc[i].append((np.mean(M**2) - np.mean(M)**2)/(1.0*T*N_spins))
    # end loop over T
# end loop over L

plt.figure(figsize=(8,6))

plt.subplot(221)
plt.axvline(x=Tc, color='k', linestyle='--', label='_nolegend_')
plt.plot(T_list, energy[0], 'o-')
plt.plot(T_list, energy[1], 'o-')
plt.plot(T_list, energy[2], 'o-')
plt.plot(T_list, energy[3], 'o-')
plt.xlabel('$T$')
plt.ylabel('$<E>/N$')
plt.legend(['L = 2', 'L = 4', 'L = 8', 'L = 16'], loc='upper left')

plt.subplot(222)
plt.axvline(x=Tc, color='k', linestyle='--', label='_nolegend_')
plt.plot(T_list, mag[0], 'o-')
plt.plot(T_list, mag[1], 'o-')
plt.plot(T_list, mag[2], 'o-')
plt.plot(T_list, mag[3], 'o-')
plt.xlabel('$T$')
plt.ylabel('$<M>/N$')
plt.legend(['L = 2', 'L = 4', 'L = 8', 'L = 16'], loc='upper left')

plt.subplot(223)
plt.axvline(x=Tc, color='k', linestyle='--', label='_nolegend_')
plt.plot(T_list, specHeat[0], 'o-')
plt.plot(T_list, specHeat[1], 'o-')
plt.plot(T_list, specHeat[2], 'o-')
plt.plot(T_list, specHeat[3], 'o-')
plt.xlabel('$T$')
plt.ylabel('$C/N$')
plt.legend(['L = 2', 'L = 4', 'L = 8', 'L = 16'], loc='upper left')

plt.subplot(224)
plt.axvline(x=Tc, color='k', linestyle='--', label='_nolegend_')
plt.plot(T_list, susc[0], 'o-')
plt.plot(T_list, susc[1], 'o-')
plt.plot(T_list, susc[2], 'o-')
plt.plot(T_list, susc[3], 'o-')
plt.xlabel('T')
plt.ylabel('$\chi/N$')
plt.legend(['L = 2', 'L = 4', 'L = 8', 'L = 16'], loc='upper left')

plt.suptitle('2d Ising model')

plt.savefig('neural_network_comparison.png')
plt.show()
