#STEP 1: Import the data

#Using the Excel file 'Batch experiment' to get the data

import pandas as pd 

data1 = pd.read_excel(r'Batch experiment.xls', sheet_name='Experiment 1')
data2 = pd.read_excel(r'Batch experiment.xls', sheet_name='Experiment 2')

#STEP 2: Convert the Pandas dataframes into arrays

time1 = data1['Time (min)'].values
concA = data1['Concentration of A (M)'].values

time2 = data2['Time (min)'].values
concB = data2['Concentration of B (M)'].values

#Let's start by analyzing the data from experiment 1

#STEP 3: Let's make some plots: [1] CA v t, [2] lnCA v t and [3] 1/CA v t

import numpy as np
import matplotlib.pyplot as plt

lnconcA = np.log(concA) #ln of concentration
recconcA = 1/concA # reciprocal of concentration

plt.plot(time1, concA, "ko")
plt.xlabel("Time (min)")
plt.ylabel("$C_A$")
plt.show()

plt.plot(time1, lnconcA, "b^")
plt.xlabel("Time (min)")
plt.ylabel("$lnC_A$")
plt.show()

plt.plot(time1, recconcA, "gs")
plt.xlabel("Time (min)")
plt.ylabel(r'$\frac{1}{C_A}$')
plt.show()

#Which one of the plots look linear? Possibly lnCA v time

#Let's do linear regression on the lnCA v time dataset

#STEP 4: Data fitting

from sklearn.linear_model import LinearRegression

model = LinearRegression()
time1rs = time1.reshape((-1, 1)) #Transpose of the array
model.fit(time1rs, lnconcA) #Performing the linear regression

#Let's find the R^2 value, slope and intercept

#STEP 5: Parameter estimation

intercept1 = model.intercept_
slope1 = model.coef_
score1 = model.score(time1rs, lnconcA) #R^2 value

#STEP 6: Trendline

#Plotting the fit

fit1 = model.predict(time1rs) #Calculating the trendline
plt.plot(time1, lnconcA, "b^", label = 'Data set')
plt.plot(time1, fit1, "b", label = 'Fit')
plt.xlabel("Time (min)")
plt.ylabel("$lnC_A$")
plt.legend()
plt.show()

#Printing the relevant information

print('Slope:', slope1)
print('Intercept:', intercept1)
print('R^2:',score1)

#Now let's fit data from experiment 2

#STEP 7: Let's make the relevant plots. Remember that we have already
#imported the data earlier

lnconcB = np.log(concB) #ln of concentration
recconcB = 1/concB # reciprocal of concentration

plt.plot(time2, concB, "ko")
plt.xlabel("Time (min)")
plt.ylabel("$C_B$")
plt.show()

plt.plot(time2, lnconcB, "b^")
plt.xlabel("Time (min)")
plt.ylabel("$lnC_B$")
plt.show()

plt.plot(time2, recconcB, "gs")
plt.xlabel("Time (min)")
plt.ylabel(r'$\frac{1}{C_B}$')
plt.show()

#Which one of the plots look linear? Possibly 1/CB v time looks the most likely

#Let's do non-linear regression the 1/CB v time dataset (second order)
#Remember that linear regression works just as well

#STEP 8: Defining the non-linear model 

def secondorder(time2,constant):
    concentration = 1/((1*constant*time2) + 1)
    return concentration

#The initial concentration of B is 1 M

#STEP 9: Define the residual

def residual(constant):
    return secondorder(time2,constant) - concB

#STEP 10: Run the regression function

from scipy.optimize import least_squares 

fit = least_squares(residual, 1) #Fitting, 1 is the initial guess

concBfit = secondorder(time2, fit.x) #Data for trendline, parameters stored in the x-array of fit2

#Plot
plt.plot(time2, concB, "gs", label = 'Experimental data')
plt.plot(time2, concBfit, "g", label = 'Optimized fit')
plt.xlabel("Time (min)")
plt.ylabel('$C_B$')
plt.legend()
plt.show()

print('')
print('The rate constant estimated by least squares is', fit.x, '1/min')
slope2 = fit.x

#We now have the data we need

#From experiment 1, order of A is 1, k' is 0.026 > k is 2.6
#From experiment 2, order of B is 2, k" is 0.089 > k is 2.67

k = ((-slope1/100)+(slope2/30))/2 #This is the rate constant (note the units)

#STEP 11: Sizing the PBR

#We have imported all the functions we need with the exception of odeint

from scipy.integrate import odeint 

#We are assuming the gas behaves ideally
#The temperature is constant
#Reaction: A + B -> C
# -rA = k*CA*CB^2

a = 1
b = 1
c = 1
delta = c - (a + b)
yAo = 0.1
e = (yAo*delta)/a
alpha = 0.003 #kg-1
vo = 0.00001 #L/min
T = 366 #K, note that the batch reactor experiments are conducted at the same temperature as the PBR
P = 1.5 #atm
R = 0.082059 #L.atm/(mol. K)
Fo = (P*vo)/(R*T)
FAo = yAo*Fo
CAo = FAo/vo
Xo = 0 #initial conversion is 0

#Governing equations of the PBR
def pbr(S,W):
    #S is [XA, y]
    dXAdW = ((k*(CAo**2))/vo)*(((1-S[0])/(1+e*S[0]))**3)*(S[1]**3)
    dydW = -(alpha/(2*S[1]))*(1+e*S[0])
    dSdW = [dXAdW, dydW]
    return dSdW

#The units are balanced

#The beg weight is 30 kg

#Solving the system of ODEs

W = np.linspace(0, 50, 200)
S0 = [0,1]
S = odeint(pbr, S0, W) #Solving the ODE system
XA = S[:,0]
y = S[:,1]

#Plotting the solution
plt.plot(W, XA, "k", label = 'Conversion')
plt.plot(W, y, "b", label = 'Dimensionless pressure')
plt.xlabel("Bed weight (kg)")
plt.ylabel("Conversion & dimensionless pressure")
plt.legend()
plt.show()