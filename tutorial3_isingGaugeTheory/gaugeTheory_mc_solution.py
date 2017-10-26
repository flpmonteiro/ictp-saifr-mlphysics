########## ICTP-SAIFR Minicourse on Machine Learning for Many-Body Physics ##########
### Roger Melko, Juan Carrasquilla and Lauren Hayward Sierens
### Tutorial 1: Monte Carlo for the Ising lattice gauge theory
#####################################################################################

import numpy as np
import random

### Input parameters: ###
#T_list = np.linspace(5.0,0.5,19) #temperature list
T_list = np.array([1.0e-3,1e3])
L = 4          #linear size of the lattice
N_sites = L**2  #total number of lattice sites
N_spins = 2*L**2 #total number of spins (one spin on each link)
J = 1            #coupling parameter
Tc = 1
### Monte Carlo parameters: ###
n_eqSweeps = 100   #number of equilibration sweeps
n_bins = 10000    #total number of measurement bins
n_sweepsPerBin=50 #number of sweeps performed in one bin

### Files to write training and testing spin configurations (X) and phases (y): ###
train_frac   = 5.0/7.0 #fraction of data to be used for training
file_Xtrain = open('Xtrain.txt', 'w')
file_ytrain = open('ytrain.txt', 'w')
file_Xtest  = open('Xtest.txt', 'w')
file_ytest  = open('ytest.txt', 'w')

### Initially, the spins are in a random state (a high-T phase): ###
spins = np.zeros(N_spins,dtype=np.int)
for i in range(N_spins):
  spins[i] = 2*random.randint(0,1) - 1 #either +1 or -1

### Store each lattice site's four nearest neighbours in a neighbours array (using periodic boundary conditions): ###
neighbours = np.zeros((N_sites,4),dtype=np.int)
for i in range(N_sites):
  #neighbour to the right:
  neighbours[i,0]=i+1
  if i%L==(L-1):
    neighbours[i,0]=i+1-L
  
  #upwards neighbour:
  neighbours[i,1]=i+L
  if i >= (N_sites-L):
    neighbours[i,1]=i+L-N_sites
  
  #neighbour to the left:
  neighbours[i,2]=i-1
  if i%L==0:
    neighbours[i,2]=i-1+L
  
  #downwards neighbour:
  neighbours[i,3]=i-L
  if i <= (L-1):
    neighbours[i,3]=i-L+N_sites
#end of for loop

### Function to calculate the total energy: ###
def getEnergy():
  currEnergy = 0
  for i in range(N_sites):
    currEnergy += -J*getPlaquetteProduct(i)
  return currEnergy
#end of getEnergy() function

### Function to calculate the product of spins on plaquette i: ###
def getPlaquetteProduct(i):
  # ************************************************************** #
  # *** FILL IN: CALCULATE THE PRODUCT OF SPINS ON PLAQUETTE i *** #
  # ************************************************************** #
  return spins[2*i]*spins[(2*i)+1]*spins[2*neighbours[i,1]]*spins[(2*neighbours[i,0])+1]

### Function to perform one Monte Carlo sweep ###
def sweep():
  #do one sweep (N_spins local updates):
  for i in range(N_spins):
    #randomly choose which spin to consider flipping:
    spinLoc = random.randint(0,N_spins-1)
    
    # ************************************************************ #
    # *** FILL IN: CALCULATE deltaE FOR THE PROPOSED SPIN FLIP *** #
    # ************************************************************ #
    #calculate the change in energy of the proposed move by considering the two plaquettes it will affect:
    plaq1 = spinLoc//2
    #get plaq2 based on whether the spin is on a horizontal or vertical link:
    if (spinLoc%2)==0:
      plaq2 = neighbours[plaq1,3]
    else:
      plaq2 = neighbours[plaq1,2]
    
    deltaE = 2*J*( getPlaquetteProduct(plaq1) + getPlaquetteProduct(plaq2) )
  
    if (deltaE <= 0) or (random.random() < np.exp(-deltaE/T)):
      #flip the spin:
      spins[spinLoc] = -spins[spinLoc]
  #end loop over i
#end of sweep() function

### Function to perform one Monte Carlo sweep ###
def sweep_gauge():
  #do one sweep (N_spins local updates):
  for i in range(N_spins):
      spinLoc = random.randint(0,N_sites-1)
      #flip the spins:
      spins[2*spinLoc] = -spins[2*spinLoc]#right
      spins[2*spinLoc+1] = -spins[2*spinLoc+1] #up
      spins[2*neighbours[spinLoc,2]] = -spins[2*neighbours[spinLoc,2]] #left
      spins[2*neighbours[spinLoc,3]+1] = -spins[2*neighbours[spinLoc,3]+1] #down
  #end loop over i
#end of sweep() function

#################################################################################
########## Loop over all temperatures and perform Monte Carlo updates: ##########
#################################################################################

### Function to write the training/testing data to file: ###
def writeConfigs(num,T):
  #determine whether the current configuration will be used for training or testing:
  if num < (train_frac*n_bins):
    file_X = file_Xtrain
    file_y = file_ytrain
  else:
    file_X = file_Xtest
    file_y = file_ytest

  #multiply the configuration by +1 or -1 to ensure we generate configurations with both positive and negative magnetization:
  flip = 2*random.randint(0,1) - 1 
    
  #loop to write each spin to a single line of the X data file:
  for i in range(N_spins):
    currSpin = spins[i]
    
    #replace -1 with 0 (to be consistent with the desired format):
    #if currSpin == -1: 
     # currSpin = 0 
    
    file_X.write('%d  ' %(currSpin) )
  #end loop over i
  file_X.write('\n')
  
  y = 0 
  if T>Tc:
    y = 1 
  file_y.write('%d \n' %y) 
#end of writeConfigs(num,T) function


for T in T_list:
  print('T = %f' %T)
  
  #open a file where observables will be recorded:
  #fileName         = 'gaugeTheory2d_L%d_T%.4f.txt' %(L,T)
  #file_observables = open(fileName, 'w')
  
  #equilibration sweeps:
  for i in range(n_eqSweeps):
    sweep()

  #start doing measurements:
  for i in range(n_bins):
    for j in range(n_sweepsPerBin):
      sweep_gauge()
    #end loop over j
    
    #energy = getEnergy()
    #file_observables.write('%d \t %.8f \n' %(i, energy))

    writeConfigs(i,T)
    
    if (i+1)%50==0:
      print '  %d bins complete' %(i+1)
  #end loop over i

  #file_observables.close()
#end loop over temperature
