import numpy as np
import math
import csv

x = np.genfromtxt('Results_Rmse/RMSECDF1.csv')
y = np.genfromtxt('Results_Rmse/RMSECDF.csv')

z = np.zeros((96,96))
z[0:75,:] = y[0:75,:]
z[75:,:] = x[75:,:]
print z[:,0]
np.savetxt('Results_Rmse/RMSECDF2.csv',z)
