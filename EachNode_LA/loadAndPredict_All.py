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

train_n = 0
test_n = 0

#Resize the image
def img_enlarge(img, factor = 2.0, order = 2):
    return scipy.ndimage.zoom(img, factor, order = order)

#Convert data into (feature, label) format
def ConvertSeriesToMatrix(numEvents, Temp, Wind, Events, Holiday, Time, len1, len2, numWeek, numDay, TimeEachDay):
    matrix = []
    #We need to discard the data 0 ~ len2-1
    for i in range(len(numEvents) - len2):
        tmp = []            #(feature, label) at the time slot i + len2
        tmp.append(Temp[i+len2])    #Temperature
        tmp.append(Wind[i+len2])    #Wind speed
        tmp.append(Events[i+len2])  #Events
        tmp.append(Holiday[i+len2]) #Holiday
        tmp.append(Time[i+len2])    #Time
        
        #Weekly dependence
        for j in range(numWeek):
            tmp.append(numEvents[i+len2-(j+1)*7*TimeEachDay])
        #Daily dependence
        for j in range(numDay):
            tmp.append(numEvents[i+len2-(j+1)*TimeEachDay])
        #Hourly dependence
        for j in range(i+len2-len1, i+len2-1):  #Note: Skip the closest one, due to superresolve
            tmp.append(numEvents[j])
        
        #Label
        tmp.append(numEvents[i+len2])
        matrix.append(tmp)
    return matrix

def loadAndPredict(numEvents, Temp, Wind, Events, Holiday, Time, TimeEachDay):
    #Normalize data to (0, 1)
    scaler1 = MinMaxScaler(feature_range = (0, 1))
    numEvents = scaler1.fit_transform(numEvents)

    #Dependence
    numWeek = 3; numDay = 3; numHour = TimeEachDay
    sequence_length1 = numHour + 1
    sequence_length2 = numWeek*7*TimeEachDay + 1
    matrix = ConvertSeriesToMatrix(numEvents, Temp, Wind, Events, Holiday, Time, sequence_length1, sequence_length2, numWeek, numDay, TimeEachDay)
    matrix = np.asarray(matrix)
    #Split dataset: 20% for testing
    train_row = int(round(0.8*matrix.shape[0]))
    test_set = matrix[train_row:, :]
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    
    #Transform the testing set into the LSTM format (number of samples, the dim of each elements)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # load json and create model
    json_file = open('Saved_models/model{0}.json'.format(train_n), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("Saved_models/model{0}.h5".format(train_n))
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
    
    Temp2 = np.asarray(Temp2); Temp2 = Temp2.astype('float32')
    Wind2 = np.asarray(Wind2); Wind2 = Wind2.astype('float32')
    Events2 = np.asarray(Events2); Events2 = Events2.astype('float32')
    Holiday2 = np.asarray(Holiday2); Holiday2 = Holiday2.astype('float32')
    Time2 = np.asarray(Time2); Time2 = Time2.astype('float32')

    RMSECDF_MAT = np.zeros((96,96))
    RMSEPDF_MAT = np.zeros((96,96))
    RMSEPDF_MAT_ZEROS = np.zeros((96,))

    for train_index in range(75):
        train_n = train_index+1
        # load json and create model
        json_file = open('Saved_models/model{0}.json'.format(train_n), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights("Saved_models/model{0}.h5".format(train_n))
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss = "mse", optimizer = "adam")
        for test_index in range(96):
            test_n = test_index+1
            print 'USE {0} th NODE to predict {1} th NODE'.format(train_n, test_n)
            ndays = 730
            TimeEachDay = 24
            df1 = pd.read_csv('data.csv', header = None)
            numEvents = df1.iloc[:, test_n-1]
            print '1st day of test data: ', numEvents[0:24]
            print('numEvents Size: ', numEvents.shape)
            numEvents = np.asarray(numEvents)
            numEvents2 = np.zeros(ndays*TimeEachDay)    #cdf of the events
            for i in range(len(numEvents)):
                if i%24 == 0:
                     numEvents2[i] = numEvents[i]
                else:
                     numEvents2[i] = numEvents[i]+numEvents2[i-1]
            np.savetxt('Results_Test/testCum{0}.csv'.format(test_n), numEvents2)
            numEvents3 = img_enlarge(numEvents2, factor = 2.0, order = 2)   #Superresolve
            print('numEvents3 size: ', len(numEvents3))
            numEvents3 = numEvents3.astype('float32')
            
            TimeEachDay *= 2    #Superresolve

            #load, and test the model on other nodes.
            #testPredict = loadAndPredict(numEvents3, Temp2, Wind2, Events2, Holiday2, Time2, TimeEachDay) 

            #Normalize data to (0, 1)
            scaler1 = MinMaxScaler(feature_range = (0, 1))
            numEvents3 = scaler1.fit_transform(numEvents3)

            #Dependence
            numWeek = 3; numDay = 3; numHour = TimeEachDay
            sequence_length1 = numHour + 1
            sequence_length2 = numWeek*7*TimeEachDay + 1
            matrix = ConvertSeriesToMatrix(numEvents3, Temp2, Wind2, Events2, Holiday2, Time2, sequence_length1, sequence_length2, numWeek, numDay, TimeEachDay)
            matrix = np.asarray(matrix)
            #Split dataset: 20% for testing
            train_row = int(round(0.8*matrix.shape[0]))
            test_set = matrix[train_row:, :]
            x_test = test_set[:, :-1]
            y_test = test_set[:, -1]
            
            #Transform the testing set into the LSTM format (number of samples, the dim of each elements)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            testPredict = loaded_model.predict(x_test)
            print 'Finished prediction.'
            testPredict = scaler1.inverse_transform(testPredict)
            np.savetxt('Results_Test/testPredictCumSupT_train{0}_test{1}.csv'.format(train_n, test_n), testPredict)
            test = numEvents2
            testPredict = testPredict[-61*48:];
            ECDF = test[-61*24:]
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

            EPDF[[i for i in range(len(EPDF)) if EPDF[i]<0]] = 0
            PPDF[[i for i in range(len(PPDF)) if PPDF[i]<0]] = 0
            ErrPDF = EPDF - PPDF;
            RMSEPDF = np.linalg.norm(ErrPDF)/np.sqrt(len(ErrPDF))
            zero_rmse = np.linalg.norm(EPDF)/np.sqrt(len(EPDF))
            print 'RMSEPDF = ', RMSEPDF
            print 'RMSEPDF of zeros = ', zero_rmse
            RMSECDF_MAT[train_n-1,test_n-1] = RMSECDF
            RMSEPDF_MAT[train_n-1,test_n-1] = RMSEPDF
            RMSEPDF_MAT_ZEROS[test_n-1] = zero_rmse

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
            print train_n, test_n
            np.savetxt('Results_Matrix/AnoAcc_train{0}_test{1}.csv'.format(train_n, test_n), AnoAcc)
            print AnoAcc

    np.savetxt('Results_Rmse/RMSEPDF.csv', RMSEPDF_MAT)
    np.savetxt('Results_Rmse/RMSECDF.csv', RMSECDF_MAT)
    np.savetxt('Results_Rmse/RMSEPDF_zero1.csv', RMSEPDF_MAT_ZEROS)
