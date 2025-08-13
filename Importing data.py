#Make sure that you have Pandas and XLRD installed on your computers

#Let's first import the data into Python
import pandas as pd 

data1 = pd.read_excel(r'Sample data.xls', sheet_name='Data set 1')
data2 = pd.read_excel(r'Sample data.xls', sheet_name='Data set 2')

#For analysis, we need to convert the Pandas dataframes into arrays of values

time1 = data1['Time (min)'].values
conc1 = data1['Concentration (M)'].values
time2 = data2['Time (min)'].values
conc2 = data2['Concentration (M)'].values

#Let's plot the data 

import matplotlib.pyplot as plt

plt.plot(time1, conc1, "ko", label = 'Data set 1')
plt.xlabel("Time (min)")
plt.ylabel("Concentration (M)")
plt.legend()
plt.show()

plt.plot(time2, conc2, "bo", label = 'Data set 2')
plt.xlabel("Time (min)")
plt.ylabel("Concentration (M)")
plt.legend()
plt.show()