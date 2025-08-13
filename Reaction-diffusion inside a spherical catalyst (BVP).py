#This code allows you to solve a second-order ODE

#A common situation where you will use this code is to solve the ODE
#for the concetration profile inside a catalyst

#Let's start with a first-order reaction to compare the analytical and numerical solutions

#Importing the requisite libraries
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fmin
import matplotlib.pyplot as plt

#Let us first consider the analytical solutions

#Analytical solution 1 is for C(r=0) = 0
#Analytical solution 2 is for dCdr(r=0) = 0

R = 2# millimeter

def analytical1(r,phi):
    C1 = R*(np.sinh((phi*r)/R))/np.sinh(phi)
    return C1 

def analytical2(r,phi):
    C2 = R*(np.cosh((phi*r)/R))/np.cosh(phi)
    return C2 

#Plotting analytical solutions as a series of solutions

r = np.linspace(0,R,100)
phiarray = np.linspace(0.1, 10,10) #creating a series of phi values to see how the Thiele modulus impacts performance
x = np.size(r)
y = np.size(phiarray)
Canalytical1 = np.zeros((x,y))
Canalytical2 = Canalytical1
i = 0
j = 0
#Plotting the analytical solutions for the C(r=0) = 0 case

while i < y:
    p = phiarray[i]
    Canalytical1[:,i] = analytical1(r,p)
    if i == 0:
        plt.plot(r, Canalytical1[:,i], "b")
    elif i == y-1:
        plt.plot(r, Canalytical1[:,i], "r")
    else:
        plt.plot(r, Canalytical1[:,i], "k")
    i += 1
plt.xlabel("Radius")
plt.ylabel("Scaled concentration " r'$\frac{C_A.r}{C_{A_surf}}$')
plt.title('Analytical solutions: $C_A$ = 0 at r = 0')
plt.show()


#Plotting the analytical solutions for the dCdr(r=0) = 0 case

while j < y:
    q = phiarray[j]
    Canalytical2[:,j] = analytical2(r,q)
    if j == 0:
        plt.plot(r, Canalytical2[:,j], "b")
    elif j == y-1:
        plt.plot(r, Canalytical2[:,j], "r")
    else:
        plt.plot(r, Canalytical2[:,j], "k")
    j += 1
plt.xlabel("Radius")
plt.ylabel("Scaled concentration " r'$\frac{C_A.r}{C_{A_surf}}$')
plt.title("Analytical solutions: " r'$\frac{dC_A}{dr}$' " = 0 at r = 0")
plt.show()


#odeint only solves IVPs
#This means we need to use the shooting method to solve the BVP
#The shooting method is simply an optimization problem
#We will solve the problem for a single value of phi to demonstrate the method

P = 2 #value for phi that we will use to solve the BVP numerically
r = np.linspace(0,R,25) #Re-defining r so that we have a clearer plot

#The solution array is S = [C, J] where dCdr = J

def bvp(S, r):
    dCdr = S[1]
    dJdr = ((P/R)**2)*S[0]
    dSdr = [dCdr, dJdr]
    return dSdr

#Shooting method for case of C = 0 at r = 0

def shooting1(J_guess):
    S0 = [0,J_guess]
    S1 = odeint(bvp, S0, r)
    C_numerical1 = S1[:,0]
    k = np.size(r)-1
    sol1 = abs(C_numerical1[k] - R)
    return sol1

#Running the optimization
J_ini_correct = fmin(shooting1, -20, xtol=1e-8)

#Resolving the ODE with the correct initial value of G
S_ini1 = [0, J_ini_correct]
S_correct1 = odeint(bvp, S_ini1, r)
C_correct1 = S_correct1[:,0]
C_analytical1 = analytical1(r,P)

#Plotting the result
plt.plot(r, C_analytical1, "ko", label = 'Analytical solution')
plt.plot(r, C_correct1, "k--", label = 'Numerical solution')
plt.xlabel("Radius")
plt.ylabel("Scaled concentration " r'$\frac{C_A.r}{C_{A_surf}}$')
plt.title('$C_A$ = 0 at r = 0 and ' r'$\phi$' " = 2")
plt.legend()
plt.show()

#Shooting method for case of dCdr = 0 at r = 0

def shooting2(C_guess):
    S0 = [C_guess,0]
    S2 = odeint(bvp, S0, r)
    C_numerical2 = S2[:,0]
    k = np.size(r)-1
    sol2 = abs(C_numerical2[k] - R)
    return sol2

#Running the optimization
C_ini_correct = fmin(shooting2, 0, xtol=1e-8)

#Resolving the ODE with the correct initial value of G
S_ini2 = [C_ini_correct,0]
S_correct2 = odeint(bvp, S_ini2, r)
C_correct2 = S_correct2[:,0]
C_analytical2 = analytical2(r,P)

#Plotting the result
plt.plot(r, C_analytical2, "bo", label = 'Analytical solution')
plt.plot(r, C_correct2, "b--", label = 'Numerical solution')
plt.xlabel("Radius")
plt.ylabel("Scaled concentration " r'$\frac{C_A.r}{C_{A_surf}}$')
plt.title(r'$\frac{dC_A}{dr}$' " = 0 at r = 0 and " r'$\phi$' " = 2" )
plt.legend()
plt.show()

#Plotting all results
plt.plot(r, C_analytical1, "ko", label = 'Analytical solution for low penetration')
plt.plot(r, C_correct1, "k--", label = 'Numerical solution for low penetration')
plt.plot(r, C_analytical2, "bo", label = 'Analytical solution for finite core')
plt.plot(r, C_correct2, "b--", label = 'Numerical solution for finite core ')
plt.xlabel("Radius")
plt.ylabel("Scaled concentration " r'$\frac{C_A.r}{C_{A_surf}}$')
plt.title("Superposition of both solutions for " r'$\phi$' " = 2")
plt.legend()
plt.show()

#Thiele modulus and effectiveness factors

#Low penetration case (C = 0 at r = 0)

def effectiveness1(phi):
    cosh = np.cosh(phi)
    sinh = np.sinh(phi)
    coth = cosh/sinh
    eta = (3/(phi**2))*(phi*coth - 1)
    return eta

#Finite core case (dCdr = 0 at r = 0)

def effectiveness2(phi):
    cosh = np.cosh(phi)
    sinh = np.sinh(phi)
    tanh = sinh/cosh
    eta = (3/(phi**2))*(phi*tanh - 1)
    return eta

phiarray = np.linspace(0.1, 10,40)
etaarray1 = effectiveness1(phiarray)
etaarray2 = effectiveness2(phiarray)

plt.plot(phiarray, etaarray1, "k", label = '$C_A$ = 0 at r = 0')
plt.plot(phiarray, etaarray2, "k--", label = r'$\frac{dC_A}{dr}$' " = 0 at r = 0")
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\eta$')
plt.ylim([0,1.2])
plt.title('1st order reaction in a spherical catalyst')
plt.legend()
plt.show()