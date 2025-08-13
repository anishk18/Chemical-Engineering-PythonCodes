#Importing the requisite libraries
import numpy as np
import math as m
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Consecutive reactions: A -> B -> C
#The batch reactor is at a constant temperature and volume
#CAo present initially in the reactor, no B or C present

#Let's first solve the system of ODEs to get the concentration profiles

def batch(C,t):
    dCAdt = -k1*C[0]
    dCBdt = k1*C[0] - k2*C[1]
    dCCdt = k2*C[1]  
    dCdt = [dCAdt, dCBdt, dCCdt]
    return dCdt

#Solving & plotting the analytical solutions & ODEs
Co = [20,1,1] #M 
t = np.linspace(0,50,100) #0-25 minutes, 50 increments
k1 = 0.1 #1/min
k2 = 0.07 #1/min

conc = odeint(batch,Co,t)

#ODE solutions
CA = conc[:,0]
CB = conc[:,1]
CC = conc[:,2]

#Plots
plt.plot(t, CA, "k", label = '$C_A$')
plt.plot(t, CB, "b", label = '$C_B$')
plt.plot(t, CC, "r", label = '$C_C$')
plt.xlabel("Time (minutes)")
plt.ylabel("Concentration (M)")
plt.legend()
plt.show()

#Let's find the time when B is at a maximum

maxCB = CB[0]
index = 0

for i in range(0, len(CB)):    
    #Compare elements of CB with maxCB
    #If an element > maxCB, designate it as the new maxCB    
   if(CB[i] > maxCB):    
       maxCB = CB[i];
       index = i;

topt = t[index]
print('The maximum concentration for CB is',round(maxCB,2),'M')
print('This concentration is attained at',round(topt,2),'minutes')

#Analytical confirmation
topt_analytical = (1/(k1-k2))*m.log(k1/k2)
print('Analytical solution gives',round(topt_analytical,2),'minutes')

#Let's consider the ratio of B to C

ratio = CB/CC
plt.plot(t, ratio, "g", label = r'$\frac{C_B}{C_C}$')
plt.plot(t, CB, "b", label = '$C_B$')
plt.xlabel("Time (minutes)")
plt.ylabel("Concentration (M)")
plt.legend()
plt.show()

maxratio = ratio[0]
index2 = 0

for i in range(0, len(ratio)):    
   if(ratio[i] > maxratio):    
       maxratio = ratio[i];
       index2 = i;

topt2 = t[index2]
print('The maximum ratio of B to C occurs at',round(topt2,2),'minutes')

#What is truly the best time to stop the reactor?
#This depends on how expensive it is to separate C from B