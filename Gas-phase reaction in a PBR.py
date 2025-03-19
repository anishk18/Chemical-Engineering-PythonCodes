#Importing the requisite libraries
import numpy as np
from scipy.integrate import odeint #Using the Runge-Kutta ODE solvers
import matplotlib.pyplot as plt

#Gas-phase reaction in a PBR
#We are assuming the gas behaves ideally
#The temperature is constant

#Information provided in the problem
#Reaction: aA -> bB
# -rA = kCA**2
a = 2
b = 3
delta = b - a
yAo = 0.4
e = (yAo*delta)/a
k = 100 #L^2/(mol.kg.s)
alpha = 0.03 #kg-1
vo = 10 #L/s
T = 366 #K
P = 1.5 #atm
R = 0.082059 #L.atm/(mol. K)
Fo = (P*vo)/(R*T)
FAo = yAo*Fo
CAo = FAo/vo
Xo = 0 #initial conversion is 0

#Governing equations of the PBR
def pbr(S,W):
    #S is [XA, y]
    dXAdW = ((k*CAo)/vo)*(((1-S[0])/(1+e*S[0]))**2)*(S[1]**2)
    dydW = -(alpha/(2*S[1]))*(1+e*S[0])
    dSdW = [dXAdW, dydW]
    return dSdW

#The beg weight is 20 kg

#Solving the system of ODEs

W = np.linspace(0, 20, 200)
S0 = [0,1]
S = odeint(pbr, S0, W)
XA = S[:,0]
y = S[:,1]

#Plotting the solution
plt.plot(W, XA, "k", label = 'Conversion')
plt.plot(W, y, "b", label = 'Pressure drop')
plt.xlabel("Bed weight (kg)")
plt.ylabel("Conversion & pressure drop")
plt.legend()
plt.show()