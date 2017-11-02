import numpy as np
import matplotlib.pyplot as plt
import time
import math
import scipy.ndimage
import csv
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import argparse
import TrainNodeRNN

parser = argparse.ArgumentParser()

parser.add_argument('--train_n', type=int, default=3,
                    help='index of training node, [1,96]')
parser.add_argument('--test_n', type=int, default=3,
                    help='index of testing node, [1,96]')

FLAGS = parser.parse_args()
#Resize the image
def img_enlarge(img, factor = 2.0, order = 2):
    return scipy.ndimage.zoom(img, factor, order = order)

#Convert data into (feature, label) format

def loadAndPredict(numEvents, connected, Temp, Wind, Events, Holiday, Time, TimeEachDay):
    #Normalize data to (0, 1)
    scaler1 = MinMaxScaler(feature_range = (0, 1))
    numEvents = scaler1.fit_transform(numEvents)

    #Dependence
    numWeek = 3; numDay = 3; numHour = TimeEachDay
    sequence_length1 = numHour + 1
    sequence_length2 = numWeek*7*TimeEachDay + 1
    matrix = TrainNodeRNN.ConvertSeriesToMatrix2(numEvents, connected, Temp, Wind, Events, Holiday, Time, sequence_length1, sequence_length2, numWeek, numDay, TimeEachDay)
    matrix = np.asarray(matrix)
    #Split dataset: 20% for testing
    train_row = int(round(0.8*matrix.shape[0]))
    test_set = matrix[train_row:, :]
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    
    #Transform the testing set into the LSTM format (number of samples, the dim of each elements)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # load json and create model
    json_file = open('Saved_models/model{0}.json'.format(FLAGS.train_n), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("Saved_models/model{0}.h5".format(FLAGS.train_n))
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss = "mse", optimizer = "adam")
    testPredict = loaded_model.predict(x_test)
    print 'Finished prediction.'
    testPredict = scaler1.inverse_transform(testPredict)
    return testPredict
    
#The main function
if __name__ == '__main__':
    ndays = 730
    TimeEachDay = 24
    
    #Events
    df1 = pd.read_csv('data.csv', header = None)
    numEvents = df1.iloc[:, FLAGS.test_n-1]
    print '1st day of test data: ', numEvents[0:24]
    print('numEvents Size: ', numEvents.shape)
    numEvents = np.asarray(numEvents)
    numEvents2 = np.zeros(ndays*TimeEachDay)	#cdf of the events
    for i in range(len(numEvents)):
    	if i%24 == 0:
    	     numEvents2[i] = numEvents[i]
    	else:
    	     numEvents2[i] = numEvents[i]+numEvents2[i-1]
    np.savetxt('Results_Test/testCum{0}.csv'.format(FLAGS.test_n), numEvents2)
    numEvents3 = img_enlarge(numEvents2, factor = 2.0, order = 2)	#Superresolve
    print('numEvents3 size: ', len(numEvents3))
        
    #Weather features
    df2 = pd.read_csv('weather_holiday_Enlarged.csv', header = None)
    Temp = df2.iloc[:, 0]
    Wind = df2.iloc[:, 1]
    Events = df2.iloc[:, 2]
    Holiday = df2.iloc[:, 3]
    Time = df2.iloc[:, 4]
    print('External size: ', Temp.shape, Wind.shape, Events.shape, Holiday.shape, Time.shape)
    Temp2 = np.zeros(ndays*TimeEachDay*2)
    Wind2 = np.zeros(ndays*TimeEachDay*2)
    Events2 = np.zeros(ndays*TimeEachDay*2)
    Holiday2 = np.zeros(ndays*TimeEachDay*2)
    Time2 = np.zeros(ndays*TimeEachDay*2)
    for i in range(ndays*TimeEachDay):
    	Temp2[i*2] = Temp[i]; Temp2[i*2+1] = Temp[i]
    	Wind2[i*2] = Wind[i]; Wind2[i*2+1] = Wind[i]
    	Events2[i*2] = Events[i]; Events2[i*2+1] = Events[i]
    	Holiday2[i*2] = Holiday[i]; Holiday2[i*2+1] = Holiday[i]
    	Time2[i*2] = Time[i]; Time2[i*2+1] = Time[i]
    print('External feature dimensions: ', len(Temp2), len(Wind2), len(Events2), len(Holiday2), len(Time2))
    
    numEvents3 = numEvents3.astype('float32')
    Temp2 = np.asarray(Temp2); Temp2 = Temp2.astype('float32')
    Wind2 = np.asarray(Wind2); Wind2 = Wind2.astype('float32')
    Events2 = np.asarray(Events2); Events2 = Events2.astype('float32')
    Holiday2 = np.asarray(Holiday2); Holiday2 = Holiday2.astype('float32')
    Time2 = np.asarray(Time2); Time2 = Time2.astype('float32')

    connected3 = TrainNodeRNN.getConnectedInfo()
    
    TimeEachDay *= 2	#Superresolve

    #load, and test the model on other nodes.
    testPredict = loadAndPredict(numEvents3, connected3, Temp2, Wind2, Events2, Holiday2, Time2, TimeEachDay) 
    np.savetxt('Results_Test/testPredictCumSupT_train{0}_test{1}.csv'.format(FLAGS.train_n, FLAGS.test_n), testPredict)
    test = numEvents2
    testPredict = testPredict[-61*48:];
    ECDF = test[-61*24:];
    PCDF = testPredict[1::2]
    PCDF = np.floor(PCDF+0.5)
    ECDF = ECDF.reshape((ECDF.shape[0],1))
    ErrCDF = ECDF - PCDF;
    RMSECDF = np.linalg.norm(ErrCDF)/np.sqrt(len(ErrCDF))
    print 'RMSECDF = ', RMSECDF
    T = 24; numDay = 61
    EPDF = np.zeros(ECDF.shape);
    PPDF = np.zeros(PCDF.shape);
    for i in range(numDay):
        EPDF[i*T,0]=ECDF[i*T,0]
        PPDF[i*T,0]=PCDF[i*T,0]
        for j in range(1,T):
            EPDF[i*T+j,0]=ECDF[i*T+j,0]-ECDF[i*T+j-1,0]
            PPDF[i*T+j,0]=PCDF[i*T+j,0]-PCDF[i*T+j-1,0]

    print PCDF[0:24,0]
    print PPDF[0:24,0]
    #print PPDF[0:24]
    EPDF[[i for i in range(len(EPDF)) if EPDF[i]<0]] = 0
    PPDF[[i for i in range(len(PPDF)) if PPDF[i]<0]] = 0
    ErrPDF = EPDF - PPDF;
    RMSEPDF = np.linalg.norm(ErrPDF)/np.sqrt(len(ErrPDF))
    zero_rmse = np.linalg.norm(EPDF)/np.sqrt(len(EPDF))
    print 'RMSEPDF = ', RMSEPDF
    print 'RMSEPDF of zeros = ', zero_rmse
    #compute precision matrix
    print 'START compute precision matrix'
    TestEPDF = np.array(EPDF)
    TestPPDF = np.array(PPDF)
    AnoAcc = np.zeros((3, 3))
    for Thres in range(1,4):
        Index = np.zeros(TestEPDF.shape)
        Index[[i for i in range(len(Index)) if TestEPDF[i]>Thres-0.001]] = 1
        Index = Index.reshape(numDay, 24)
        Index = Index.transpose()
        Index1 = np.zeros(TestPPDF.shape)
        Index1[[i for i in range(len(Index1)) if TestPPDF[i]>Thres-0.001]] = 1
        Index1 = Index1.reshape(numDay, 24)
        Index1 = Index1.transpose()
        for Delay in range(3):
            IndexRes = np.array(Index1)
            for iter_ in range(1,Delay+1):
                Indextmp = np.array(Index1)
                Indextmp[iter_:, :] = Indextmp[:-iter_, :]
                Indextmp[0:iter_, :] = 0
                IndexRes = IndexRes + Indextmp
            TotalAno = np.count_nonzero(Index)
            PredAno = np.count_nonzero(np.multiply(Index,IndexRes))
            if TotalAno == 0:
                Acc = 1.0
            else:
                Acc = float(PredAno)/float(TotalAno)
            AnoAcc[Thres-1, Delay] = Acc;
    print AnoAcc
    
    
