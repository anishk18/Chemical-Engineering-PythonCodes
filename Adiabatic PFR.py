#Importing the requisite libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#We have a first order reaction

#The reaction rate expression is -rA = k.CA

#Reaction stoichiometry
a = 1
b = 1
delta = b - a

#Inlet conditions
yAo = 1 #pure A in inlet
CAo = 0.5 # M
vo = 2.5 # L/s
V = 20 #Reactor volume, L
thetaB = 0 #No B in inlet
eps = (yAo*delta)/a

#Reaction rate data
A = 100000 # 1/s
EA = 45000 #J/mol.K
R = 8.314 #J/mol.K

#Thermodynamic data
Hrxno = -65000 #J/mol
CpA = 130 #J/mol.K
CpB = 190 #J/mol.K
deltaCp = ((b*CpB) - (a*CpA))/a
To = 330 #K
Tref = 298 #K

#Define the functions

#S = [conversion, temperature]

def pfr(S,V):
    k = A*np.exp(-EA/(R*S[1]))
    dXAdV = (k*(1-S[0])*To)/(vo*S[1]*(1+eps*S[0]))
    dTdV = ((deltaCp*(Tref - S[1])-Hrxno)*(k*(1-S[0])*To))/((CpA + S[0]*deltaCp)*(vo*S[1]*(1+eps*S[0])))
    dSdV = [dXAdV, dTdV]
    return dSdV    

#Solving the system of ODEs
V = np.linspace(0, 20, 50)
So = [0,To]
S = odeint(pfr, So, V)
XA = S[:,0]
T = S[:,1]

#Preparing the plots

plt.subplot(2,1,1)
plt.plot(V, XA, "k")
plt.xlabel("Volume (L)")
plt.ylabel("Conversion ($X_A$)")

plt.subplot(2,1,2)
plt.plot(V, T, "r")
plt.xlabel("Volume (L)")
plt.ylabel("Temperature (K)")
plt.show()

print('The exit temperature is',np.round(T[len(T)-1],2),'K')
print('The exit conversion is',np.round(XA[len(XA)-1],2))