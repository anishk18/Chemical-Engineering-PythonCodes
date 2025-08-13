#Let's first import the data into Python
import pandas as pd 

data1 = pd.read_excel(r'Sample data.xls', sheet_name='Data set 1')
data2 = pd.read_excel(r'Sample data.xls', sheet_name='Data set 2')

#For analysis, we need to convert the Pandas dataframes into arrays of values

time1 = data1['Time (min)'].values
conc1 = data1['Concentration (M)'].values
time2 = data2['Time (min)'].values
conc2 = data2['Concentration (M)'].values

#Model fitting using Python's in-built linear regression tools

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

lnconc1 = np.log(conc1) #ln of concentration
recconc1 = 1/conc1 # reciprocal of concentration
lnconc2 = np.log(conc2)
recconc2 = 1/conc2

#We first reshape the independent variables

time1rs = time1.reshape((-1, 1)) #I want the transpose of the array
time2rs = time2.reshape((-1, 1))

#Regression (2 datasets, 2 possibilities evaluated for each = 4 fits)
model1_1 = LinearRegression() #Dataset 1, fit 1
model1_2 = LinearRegression() #Dataset 1, fit 2
model2_1 = LinearRegression() #Dataset 2, fit 1
model2_2 = LinearRegression() #Dataset 2, fit 2

#Dataset 1, fit 1
model1_1.fit(time1rs, lnconc1)
r_sq1_1 = model1_1.score(time1rs, lnconc1)
int1_1 = model1_1.intercept_
slope1_1 = model1_1.coef_
fit1_1 = model1_1.predict(time1rs)
plt.plot(time1, lnconc1, "ko", label = 'Data set 1')
plt.plot(time1, fit1_1, "k", label = 'Fit 1')
plt.xlabel("Time (min)")
plt.ylabel("Concentration (M)")
plt.legend()
plt.show()

#Dataset 1, fit 2
model1_2.fit(time1rs, recconc1)
r_sq1_2 = model1_2.score(time1rs, recconc1)
int1_2 = model1_2.intercept_
slope1_2 = model1_2.coef_
fit1_2 = model1_2.predict(time1rs)
plt.plot(time1, recconc1, "ko", label = 'Data set 1')
plt.plot(time1, fit1_2, "k", label = 'Fit 2')
plt.xlabel("Time (min)")
plt.ylabel("Concentration (M)")
plt.legend()
plt.show()

#Dataset 2, fit 1
model2_1.fit(time2rs, lnconc2)
r_sq2_1 = model2_1.score(time2rs, lnconc2)
int2_1 = model2_1.intercept_
slope2_1 = model2_1.coef_
fit2_1 = model2_1.predict(time2rs)
plt.plot(time2, lnconc2, "bo", label = 'Data set 2')
plt.plot(time2, fit2_1, "b", label = 'Fit 1')
plt.xlabel("Time (min)")
plt.ylabel("Concentration (M)")
plt.legend()
plt.show()

#Dataset 2, fit 2
model2_2.fit(time2rs, recconc2)
r_sq2_2 = model2_2.score(time2rs, recconc2)
int2_2 = model2_2.intercept_
slope2_2 = model2_2.coef_
fit2_2 = model2_2.predict(time2rs)
plt.plot(time2, recconc2, "bo", label = 'Data set 2')
plt.plot(time2, fit2_2, "b", label = 'Fit 2')
plt.xlabel("Time (min)")
plt.ylabel("Concentration (M)")
plt.legend()
plt.show()

#Printing the models

print('The coefficient of determination for the fit for dataset 1 is', r_sq1_1)
print('The model for dataset 1 is ln C =', int1_1, '+', slope1_1 ,'* time')

print('The coefficient of determination for the fit for dataset 2 is', r_sq2_2)
print('The model for dataset 1 is 1/C =', int2_2, '+', slope2_2 ,'* time')
