#Importing the requisite libraries
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt

#Root finding is an important exercise, especially for sizing CSTRs

#Let's define the function for conversion
def function(X):
    func = abs(2*(X**2) - 3.75*X + 1.5)
    return func


#Finding the root
Xsol = fmin(function, 0, xtol=1e-8)
print(Xsol)


#Finding the solution by plotting

def function2(X):
    func2 = 2*(X**2) - 3.75*X + 1.5
    return func2

Xplot = np.linspace(0,1,30)
Xfunc = function2(Xplot)

plt.plot(Xplot, Xfunc, "r")
plt.axhline(y=0, color='k')
plt.xlabel("XA")
plt.ylabel("f(XA)")
plt.show()