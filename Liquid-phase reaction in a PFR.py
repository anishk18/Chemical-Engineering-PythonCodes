#Importing the requisite libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Liquid-phase PFR


#Information provided in the problem
FAo = 5 #mol/min
CAo = 0.1 #M
vo = FAo/CAo
k = 0.5 # 1/(min M)
V = 200 #L

#-rA = kC(A^2)

#Governing equation of the PFR
def pfr(X,V):
    dXdV = (k*CAo*((1-X)**2))/vo
    return dXdV

#Solving the ODE
Xo = 0 #initial conversion is 0
V = np.linspace(0,200,200)
XA = odeint(pfr,Xo,V)
index = np.size(XA) - 1
XAexit = XA[index]
print(XAexit)

#Plots
plt.plot(V, XA, "k")
plt.xlabel("Volume (L)")
plt.ylabel("Conversion")
plt.show()