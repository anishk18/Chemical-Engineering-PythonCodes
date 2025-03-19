#Importing the requisite libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Consecutive reactions: A -> B -> C
#The batch reactor is at a constant temperature and volume
#CAo present initially in the reactor, no B or C present


#Analytical solution
def analytical(conc_ini,time):
    CA = conc_ini[0]*np.exp(-k1*time)
    CB = (k1*conc_ini[0]*(np.exp(-k2*time) - np.exp(-k1*time)))/(k1 - k2)
    CC = conc_ini[0] + (conc_ini[0]*(k2*np.exp(-k1*time) - k1*np.exp(-k2*time)))/(k1 - k2)
    conc = [CA, CB, CC]
    return conc


#Solving the system of ODEs
def sysode(C,t):
    dCAdt = -k1*C[0]
    dCBdt = k1*C[0] - k2*C[1]
    dCCdt = k2*C[1]  
    dCdt = [dCAdt, dCBdt, dCCdt]
    return dCdt


#Solving & plotting the analytical solutions & ODEs
Co = [100,0,0] #M 
t = np.linspace(0,25,50) #0-25 minutes, 50 increments
k1 = 0.1 #1/min
k2 = 0.07 #1/min
conc_analytical = analytical(Co, t)
conc_ode = odeint(sysode,Co,t)


#Analytical solutions
CA_analytical = conc_analytical[0]
CB_analytical = conc_analytical[1]
CC_analytical = conc_analytical[2]

#ODE solutions
CA_ode = conc_ode[:,0]
CB_ode = conc_ode[:,1]
CC_ode = conc_ode[:,2]


#Plots
plt.plot(t, CA_analytical, "ko", label = "A analytical")
plt.plot(t, CB_analytical, "ro", label = "B analytical")
plt.plot(t, CC_analytical, "co", label = "C analytical")
plt.plot(t, CA_ode, "m", label = "A ODE")
plt.plot(t, CB_ode, "y", label = "B ODE")
plt.plot(t, CC_ode, "g", label = "C ODE")
plt.xlabel("Time (minutes)")
plt.ylabel("Concentration (M)")
plt.legend()
plt.show()