import numpy as np
import math
import csv
import pandas as pd

RMSEPDF_MAT_ZEROS = np.zeros((96,))
for n in range(96):
	df1 = pd.read_csv('Results_Test/testCum{0}.csv'.format(n+1), header = None)
	test = df1.iloc[:,:]
	test = np.asarray(test)
	test = test.astype('float32')
	ECDF = test[-61*24:]
	#ECDF = ECDF.reshape((ECDF.shape[0],1))
	T = 24; numDay = 61
	EPDF = np.zeros(ECDF.shape)
	for i in range(numDay):
		EPDF[i*T]=ECDF[i*T]
		for j in range(1,T):
			EPDF[i*T+j]=ECDF[i*T+j]-ECDF[i*T+j-1]
	EPDF[[i for i in range(len(EPDF)) if EPDF[i]<0]] = 0
	zero_rmse = np.linalg.norm(EPDF)/np.sqrt(len(EPDF))
	print 'RMSEPDF of zeros at {0} th node = '.format(n+1), zero_rmse
	RMSEPDF_MAT_ZEROS[n-1] = zero_rmse

np.savetxt('Results_Rmse/RMSEPDF_zero.csv', RMSEPDF_MAT_ZEROS)
