#Importing the requisite libraries
import numpy as np
import matplotlib.pyplot as plt

#Let's make some plots of FAo/-rA versus XA


CAo = 10 #M
vo = 1 #L/s
FAo = CAo*vo
k1 = 0.1 #1/s, first-order reaction rate constant
k2 = 0.7 #1/(M.s), second-order reaction rate constant

X = np.linspace(0,1,50)
CA = CAo*(1 - X)
rate1 = k1*CA #rate expression for first-order reaction
rate2 = k2*(CA**2) #rate expression for second-order reaction
Y1 = FAo/rate1
Y2 = FAo/rate2

#Plotting the curves
plt.plot(X, Y1,"k", label = 'First-order reaction')
plt.plot(X, Y2,"b", label = 'Second-order reaction')
plt.xlabel("$X_A$")
plt.ylabel(r'$\frac{F_Ao}{-r_A}$')
plt.legend()
plt.show()

