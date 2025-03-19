#Importing the requisite libraries
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt

#We have a first order liquid phase reaction

#The reaction rate expression is -rA = k.CA

#We have derived expressions for XA from the mole and energy balances

#Reaction stoichiometry
a = 1
b = 2
delta = b - a

#Inlet conditions
CAo = 0.5 # M
vo = 2.5 # L/s
V = 100 #Reactor volume, L
thetaB = 0 #No B in inlet

#Reaction rate data
A = 100000 # L/mol.s
EA = 45000 #J/mol.K
R = 8.314 #J/mol.K

#Thermodynamic data
Hrxno = -35000 #J/mol
CpA = 130 #J/mol.K
CpB = 160 #J/mol.K
deltaCp = ((b*CpB) - (a*CpA))/a
To = 300 #K
Tref = 298 #K
U = 100 #W/sqm.K
SA = 0.05 #sqm
Tc = 277 #K, one can also calculate this using the LMTD formula

#Define the XEB and XMB functions

def energy(T):
    X = (-U*SA*(T - Tc) - CAo*vo*CpA*(T - To))/(CAo*vo*Hrxno + CAo*vo*deltaCp*(T-Tref))
    return X

def moles(T):
    k = A*np.exp(-EA/(R*T))
    tau = V/vo
    X = (k*tau)/(1 + k*tau)
    return X
                                                                                                      
#Graphical approach
T = np.linspace(280, 420, 100) #K
XEB = energy(T)
XMB = moles(T)

#Preparing the plots
plt.plot(T, XEB, "k", label ='$X_{EB}$')
plt.plot(T, XMB, "r",label ='$X_{MB}$')
plt.xlabel("Temperature (K)")
plt.ylabel("Conversion")
plt.legend()
plt.show()

#Solving the problem using optimization

def solution(T):
    XEB = energy(T)
    XMB = moles(T)
    sol = abs(XMB - XEB)
    return sol

#Running the optimization
Tsol = fmin(solution, 360, xtol=1e-8)
Xsol = moles(Tsol)
print('The exit temperature is',np.round(Tsol[0],1),'K')
print('The exit conversion is',np.round(Xsol[0],2))