import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def hisAver(numEvents, testN):
    train = numEvents[:-testN]
    test = numEvents[-testN:]
    testPredict = np.zeros(test.shape)
    for i in range(testN):
        j = i%24
        testPredict[i] = np.mean(train[range(j,train.shape[0],24)])
    #plt.plot(testPredict)
    #plt.show()
    testRMSE = math.sqrt(mean_squared_error(test, testPredict))
    print 'Historical average = ', testRMSE
    return testRMSE

if __name__ == '__main__':
    HisAver_RMSE = []
    for n_index in range(1,51):
        train_n = n_index
        ndays = 730
        TimeEachDay = 24
        
        #Events
        df1 = pd.read_csv('data.csv', header = None)
        numEvents = df1.iloc[:, n_index-1]
        print 'numEvents Size: ', numEvents.shape
        numEvents = np.asarray(numEvents)
        testN = numEvents.shape[0] - int(round(0.8*numEvents.shape[0]))
        res = hisAver(numEvents, testN)
        print "Node {0} HA = {1}".format(train_n, res) 
        HisAver_RMSE.append(res)
    HisAver_RMSE = np.asarray(HisAver_RMSE)
    np.savetxt('Results_Rmse/HisAver.csv', HisAver_RMSE)
    print 'Historical average mean = ', np.mean(HisAver_RMSE)