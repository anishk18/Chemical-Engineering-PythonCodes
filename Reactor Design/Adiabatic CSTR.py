#Importing the requisite libraries
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt

#We have a second order liquid-phase reaction A + B > C that takes place in a SS CSTR
#The reaction rate expression is -rA = k.CA.CB
#We have derived expressions for XA from the mole and energy balances

#Reaction rate data, inlet conditions & thermodynamic parameters
A = 500000 # L/mol.s
EA = 45000 #J/mol.K
CAo = 0.1 #M
V = 100 #Reactor volume, L
vo = 5 #L/s  
a = 1 #Stoichiometric coefficient of A
b = 1 #Stoichiometric coefficient of B
thetaB = 1 #This assumes that the feed is equimolar in A & B
Hrxno = -65000 #J/mol
CpA = 130 #J/mol.K
CpB = 135 #J/mol.K
CpC = 125 #J/mol.K
deltaCp = CpC - CpA - CpB
To = 330 #K
Tref = 298 #K
R = 8.314 # J/mol.K

#Define the functions

def energy(T):
    X = -(CpA*(T - To) + thetaB*CpB*(T - To))/(Hrxno + deltaCp*(T - Tref))
    return X

def moles(T):
    k = A*np.exp(-EA/(R*T))
    tau = V/vo
    X = (2*k*tau*CAo + 1 - np.sqrt(4*k*tau*CAo + 1))/2*k*tau*CAo
    return X
  
def solution(T):
    XEB = energy(T)
    XMB = moles(T)
    sol = abs(XMB - XEB)
    return sol                                                                                                        

#Solving the equations
T = np.linspace(340, 400, 30) #K
XEB = energy(T)
XMB = moles(T)

#Running the optimization
Tsol = fmin(solution, 360, xtol=1e-8)
print('The operating temperature of the CSTR is',np.round(Tsol,2))
Xfinal = energy(Tsol)
print('The exit conversion is',np.round(Xfinal,2))

#Preparing the plots
plt.plot(T, XEB, "b", label = '$X_{EB}$')
plt.plot(T, XMB, "r", label = '$X_{MB}$')
plt.plot(Tsol, Xfinal, "ko")
plt.axhline(y=Xfinal, color='k', linestyle='--')
plt.axvline(x=Tsol, color='k',linestyle='--')
plt.xlabel("Temperature (K)")
plt.ylabel("Conversion ($X_{EB}$ & $X_{MB}$)")
plt.legend()
plt.show()