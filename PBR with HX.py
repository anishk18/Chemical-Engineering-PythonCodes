#Importing the requisite libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Assuming that we have a first order reaction, A > 2B
#Ideal gases
#Specific heats are assumed to be independent of temperature
#No B in the feed
#The reactor is equipped with a heat exchanger
#There is a pressure drop in the system

#Stoichiometry
a = 1
b = 2
delta = b - a

#Inlet conditions
yAo = 1 #pure A in inlet
CAo = 0.5 # M
vo = 5 # L/s
W = 20 #kg
thetaB = 0 #No B in inlet
eps = (yAo*delta)/a
Tcin = 277#K

#Reaction rate data
A = 100000 # L/kg.s
EA = 32000 #J/mol.K
R = 8.314 #J/mol.K

#Thermodynamic data
Hrxno = -250000 #J/mol
CpA = 2000 #J/mol.K
CpB = 3000 #J/mol.K
deltaCp = ((b*CpB) - (a*CpA))/a
To = 350 #K
Tref = 298 #K

#Heat exchange and coolant data
U = 1000#W/sqm.K
a = 2000 #surface area per unit volume
mc = 0.1 #coolant mass flow rate (kg/s)
CpCool = 50000 #J/kg.K

#Ergun parameters
alpha = 0.01 #1/kg

#Define the functions
# S = [XA, T, y, Tc]

def pbr(S, W):
    k = A*np.exp(-EA/(R*S[1]))
    CA = (CAo*S[2]*To*(1 - S[0]))/(S[1]*(1 + eps*S[0]))
    Hrxn = Hrxno + deltaCp*(S[1] - Tref)
    dXAdW = (k*CA)/(CAo*vo)
    dTdW = -(U*a*(S[1]-S[3]) + k*CA*Hrxn)/(CAo*vo*(CpA + deltaCp*S[0]))
    dydW = -(alpha*S[1]*(1 + eps*S[0]))/(2*S[2]*To)
    dTcdW = (U*a*(S[1] - S[3]))/(mc*CpCool)
    dSdW = [dXAdW, dTdW, dydW, dTcdW]
    return dSdW                                                                                                          
                                                                                                                   
#Solving the system of ODEs
W = np.linspace(0, 20, 30)
S0 = [0,To,1,Tcin]
S = odeint(pbr, S0, W)

#Extracting the solutions
XA = S[:,0]
T = S[:,1]
y = S[:,2]
Tc = S[:,3]

#Preparing the plots

plt.plot(W, XA, "k", label = '$X_{A}$')
plt.plot(W, y, "g", label = 'y')
plt.xlabel("Bed weight (kg)")
plt.ylabel("Conversion or dimensionless pressure")
plt.legend()
plt.show()

plt.plot(W, T, "r--", label = "Reactor")
plt.plot(W, Tc, "b", label = "Coolant")
plt.xlabel("Bed weight (kg)")
plt.ylabel("Temperature (K)")
plt.legend()
plt.show()

print('The exit temperature of the reacting stream is',np.round(T[len(T)-1],2),'K')
print('The exit conversion is',np.round(XA[len(XA)-1],2))
print('The pressure drop is',np.round((1-y[len(y)-1])*100,2),'%')