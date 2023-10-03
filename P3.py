import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import pandas as pd     #to load .txt
from matplotlib.cbook import get_sample_data
from numpy.lib.function_base import copy 
import os.path
import sys
import time
import os

simulation='P3'

#DEFINITION OF FUNCTIONS

def define_lattice_sc(M, L_cell): #We define the initial positions of sc lattice
    pos = []
    for nx in range(M):
        for ny in range(M):
            for nz in range(M):
                pos.append([nx,  ny, nz])
    return np.array(pos) * L_cell


def therm_Andersen(v, nu, sigma,dt): #Andersen thermostat
    N = len(v)
    for n in range(N):
        if np.random.ranf() < nu*dt:
            v[n] = np.random.normal(0,sigma,3)
    return v


def pbc(vec, L_box): #Periodic boundary conditions
    vec = vec - np.rint(vec / L_box) * L_box
    return np.array(vec)
        

def find_force_LJ(pos, L_box): #Calculate force and potential energy
    N = len(pos)
    F = np.zeros((N,3))
    cutoff2 = cutoff*cutoff
    pot = 0.0
    for i in range(N):
        for j in range(i+1, N):
                rij = pbc(pos[i]-pos[j], L_box)
                d2 = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]
                d4 = d2*d2; d6 = d4*d2; d8 = d6*d2; d12 = d6*d6; d14 = d8*d6
                if d2<cutoff2:
                    F[i] = F[i] + (48 / d14 - 24 /d8)*rij
                    F[j] = F[j] - (48 / d14 - 24 /d8)*rij
                    pot = pot + 4.*( 1/ d12 - 1 /d6)
    
    return np.array(F), pot


def time_step_vVerlet(pos, vel, L_box,F): #Velocity Verlet algorithm
    pos = pos + vel * dt + 0.5* F * dt*dt
    pos = pbc(pos, L_box)   #Apply Periodic Boundary Conditions to place particles inside de simulation box
    vel += F* 0.5 * dt
    F, pot = find_force_LJ(pos, L_box)
    vel += F* 0.5 * dt
    kin = 0.5 * np.sum(vel**2)
    return np.array(pos), np.array(vel), np.array(F), pot, kin


# ------------ INITIALIZATION -------------------------


print(os.getcwd()) #Create new folder if doesn't exist
if not os.path.exists('./'+simulation+'/'):
    os.makedirs('./'+simulation+'/')
    print('new folder '+simulation+'')

#First we define all parameters which we will use
rho = 0.8
Nsc = 125
Lsc = np.double((np.double(Nsc/rho))**(1. / 3.))
Msc = 5
asc = np.double(Lsc/Msc)

#Now we obtain sc lattice
r = define_lattice_sc(asc,Msc)

cutoff = 0.95*(Lsc/2.)
dt = 0.0001
Ntimesteps1 = 10000
N = 125
v = np.zeros((N,3))

#---------- 1st Part--------------

#Parameters for obtain a disordered position of the atoms
Temp1 = 100.0
nu = 0.1
pas = 10
sigma1 = np.sqrt(Temp1)

#We apply velocity verlet
F, pot = find_force_LJ(r, Lsc)
for t in range(Ntimesteps1):
    r, v, F, pot, kin = time_step_vVerlet(r, v, Lsc,F)
    v = therm_Andersen(v, nu, sigma1,dt)

#Print the disordered position
file1 = os.path.join('./'+simulation+'/trajectory.xyz')
with open(file1,'w+') as f:
        f.write('%i\n\n'%N)
        for i in range(0,N):
            f.write('A  %14.8f  %14.8f  %14.8f\n'%(r[i][0], r[i][1], r[i][2]))

f.close()

##---------- 2nd Part --------------

#Parameters for 2nd part
Ntimesteps2 = 100000
Temp2 = 10.0
sigma2 = np.sqrt(Temp2)

#We apply velocity verlet
F, pot = find_force_LJ(r, Lsc)
file2 = os.path.join('./'+simulation+'/thermodynamics.dat')
with open(file2,'w+') as f:
    for t in range(Ntimesteps2):
        r, v, F, pot, kin = time_step_vVerlet(r, v, Lsc,F)
        v = therm_Andersen(v, nu, sigma2,dt)
        T = (kin*2.0)/(3.0*N)

        if (t%pas==0):  #Every pas timesteps save the energies and T
                        
            Etot = pot + kin
            f.write('%14.8f %14.8f %14.8f %14.8f %14.8f\n'%(t, pot, kin, Etot,T))

f.close()

#Load the data for the plots
df = pd.read_csv('./'+simulation+'/thermodynamics.dat', header=None,delim_whitespace=True)
df.columns = ["t", "pot", "kin", "total", "T"]

#Do the plots
f, axarr = plt.subplots(2,2)
axarr[0,0].plot(df['t'],df['pot'])
axarr[0,0].set_xlabel(r'$t$',fontsize=12)
axarr[0,0].set_ylabel(r'$Potential\,energy$',fontsize=12)
axarr[0,0].get_yaxis().get_major_formatter().set_useOffset(False)


axarr[0,1].plot(df['t'],df['kin'])
axarr[0,1].set_xlabel(r'$t$',fontsize=12)
axarr[0,1].set_ylabel(r'$Kinetic\,energy$',fontsize=12)
axarr[0,1].get_yaxis().get_major_formatter().set_useOffset(False)


axarr[1,0].plot(df['t'],df['total'])
axarr[1,0].set_xlabel(r'$t$',fontsize=12)
axarr[1,0].set_ylabel(r'$Total\,energy$',fontsize=12)
axarr[1,0].get_yaxis().get_major_formatter().set_useOffset(False)

axarr[1,1].plot(df['t'],df['T'])
axarr[1,1].set_xlabel(r'$t$',fontsize=12)
axarr[1,1].set_ylabel(r'$T$',fontsize=12)
axarr[1,1].get_yaxis().get_major_formatter().set_useOffset(False)

figure = plt.gcf()  
figure.set_size_inches(9, 9,forward=True)
plt.tight_layout()
file4=os.path.join('./'+simulation+'/energies_temperature.png')
f.savefig(file4, dpi = 200)

plt.close()


