#Importing the requisite libraries
from scipy.integrate import solve_ivp #Using the Runge-Kutta ODE solvers
from scipy.optimize import fsolve #Using Python's in-built rootfinder
import matplotlib.pyplot as plt

#Gas-phase reaction PFR
#We are assuming the PFR behaves ideally
#The pressure and temperature are constant


#Information provided in the problem
#Reaction: aA -> bB
# -rA = kCA**2
a = 2
b = 3
delta = b - a
yAo = 0.2
e = (yAo*delta)/a
k = 0.05 #L/(mol.s)
vo = 1 #L/s
T = 366 #K
P = 1.5 #atm
R = 0.082059 #L.atm/(mol. K)
Fo = (P*vo)/(R*T)
FAo = yAo*Fo
CAo = FAo/vo
Xo = 0 #initial conversion is 0

#Governing equation of the PFR
def pfr(V,X):
    dXdV = ((k*CAo)/vo)*(((1-X)/(1+e*X))**2)
    return dXdV

#In the first type of problem, one could be given a volume of the PFR
#and asked to find the final conversion

#Let's assume that you were given that the reactor volume is 3000 L

#Solving the ODE
sol = solve_ivp(pfr, [0,3000], [0], method='RK23',dense_output=True)

#Plotting the solution
plt.plot(sol.t, sol.y[0],"k")
plt.xlabel("Volume (L)")
plt.ylabel("Conversion")
plt.show()
print('Final conversion is',sol.y[0,len(sol.y[0])-1]) #This is the exit conversion


#In the second type of problem, one could be given a final conversion
#and be asked to find the volume of the reactor that achieves this conversion
#This type of problem is an optimization problem

#Let the final conversion in this instance be 0.5

def findvolume(Vguess):
    Vsolve = [0,Vguess]
    solve = solve_ivp(pfr, Vsolve, [0], method='RK23',dense_output=True)
    Xsolve = solve.y[0,len(solve.y[0])-1]
    result = 0.5 - Xsolve #this is our objective function
    return result

#Finding the solution
volumereqd = fsolve(findvolume,1000,xtol=1e-6)
print('Required volume is', volumereqd)