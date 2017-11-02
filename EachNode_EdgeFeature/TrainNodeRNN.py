#!/usr/bin/python
'''
This code is used to predict the crime time series for the ZipCode region: 90003.
Diurnal cumulative and temporal superresolution.
Hourly, Daily, Weekly features.
External features.
'''
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
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_n', type=int, default=3,
                    help='index of node, [1,96]')

FLAGS = parser.parse_args()
#Resize the image
def img_enlarge(img, factor = 2.0, order = 2):
    return scipy.ndimage.zoom(img, factor, order = order)

#Convert data into (feature, label) format
def ConvertSeriesToMatrix(numEvents, connected, Temp, Wind, Events, Holiday, Time, len1, len2, numWeek, numDay, TimeEachDay):
    matrix = []
    #We need to discard the data 0 ~ len2-1
    for i in range(len(numEvents) - len2):
    	tmp = []			#(feature, label) at the time slot i + len2    
          
        week_dep0 = numEvents[range(i+len2-7*TimeEachDay, i+len2-4*7*TimeEachDay, -7*TimeEachDay)]
        day_dep0 =  numEvents[range(i+len2-TimeEachDay, i+len2-4*TimeEachDay, -TimeEachDay)]
        hour_dep0 = numEvents[range(i+len2-len1, i+len2-1)]
        #Feature from the connected nodes
        for col in range(connected.shape[1]):
            '''
            # data feature
            for j in range(numWeek):
                tmp.append(connected[i+len2-(j+1)*7*TimeEachDay,col])
            for j in range(numDay):
                tmp.append(connected[i+len2-(j+1)*TimeEachDay,col])
            for j in range(i+len2-6, i+len2-1,2):  #Note: Skip the closest one, due to superresolve
            #for j in range(i+len2-len1, i+len2-1):
                tmp.append(connected[j,col])
            '''
            # covariance feature
            week_dep = connected[range(i+len2-7*TimeEachDay, i+len2-4*7*TimeEachDay, -7*TimeEachDay), col]
            day_dep =  connected[range(i+len2-TimeEachDay, i+len2-4*TimeEachDay, -TimeEachDay),col]
            hour_dep = connected[range(i+len2-len1, i+len2-1),col]
            tmp.append(np.cov(week_dep,week_dep0)[0][1])
            tmp.append(np.cov(day_dep,day_dep0)[0][1])
            tmp.append(np.cov(hour_dep,hour_dep0)[0][1])
            # Graph inference
            tmp.append(A[cindex[col],FLAGS.train_n-1])
                        
    	tmp.append(Temp[i+len2])	#Temperature
    	tmp.append(Wind[i+len2])	#Wind speed
    	tmp.append(Events[i+len2])	#Events
    	tmp.append(Holiday[i+len2])	#Holiday
    	tmp.append(Time[i+len2])	#Time

    	#Weekly dependence
    	for j in range(numWeek):
    	    tmp.append(numEvents[i+len2-(j+1)*7*TimeEachDay])
    	#Daily dependence
    	for j in range(numDay):
    	    tmp.append(numEvents[i+len2-(j+1)*TimeEachDay])
    	#Hourly dependence
    	for j in range(i+len2-len1, i+len2-1):	#Note: Skip the closest one, due to superresolve
    	    tmp.append(numEvents[j])
    	#Label
    	tmp.append(numEvents[i+len2])
        if i<5:
            print len(tmp)
        matrix.append(tmp)
    return matrix

#Convert data into (feature, label) format, combine connected node version.
def ConvertSeriesToMatrix2(numEvents, connected, Temp, Wind, Events, Holiday, Time, len1, len2, numWeek, numDay, TimeEachDay):
    matrix = []
    #We need to discard the data 0 ~ len2-1
    for i in range(len(numEvents) - len2):
        tmp = []            #(feature, label) at the time slot i + len2   
        week_dep0 = numEvents[range(i+len2-7*TimeEachDay, i+len2-4*7*TimeEachDay, -7*TimeEachDay)]
        day_dep0 =  numEvents[range(i+len2-TimeEachDay, i+len2-4*TimeEachDay, -TimeEachDay)]
        hour_dep0 = numEvents[range(i+len2-len1, i+len2-1)]
        #Feature from the connected nodes
        week_dep = np.zeros((numWeek,))
        day_dep = np.zeros((numDay,))
        hour_dep = np.zeros((len1-1,))
        for col in range(connected.shape[1]):
            week_dep = week_dep + A[cindex[col],FLAGS.train_n-1]*connected[range(i+len2-7*TimeEachDay, i+len2-4*7*TimeEachDay, -7*TimeEachDay), col]
            day_dep  = day_dep +  A[cindex[col],FLAGS.train_n-1]*connected[range(i+len2-TimeEachDay, i+len2-4*TimeEachDay, -TimeEachDay),col]
            hour_dep = hour_dep + A[cindex[col],FLAGS.train_n-1]*connected[range(i+len2-len1, i+len2-1),col]
        tmp += list(week_dep)
        tmp += list(day_dep)
        tmp += list(hour_dep)

        '''
        #covariance:
        for col in range(connected.shape[1]):
            tmp.append(np.cov(connected[range(i+len2-7*TimeEachDay, i+len2-4*7*TimeEachDay, -7*TimeEachDay), col],week_dep0)[0][1])
            tmp.append(np.cov(connected[range(i+len2-TimeEachDay, i+len2-4*TimeEachDay, -TimeEachDay),col],day_dep0)[0][1])
            tmp.append(np.cov(connected[range(i+len2-len1, i+len2-1),col], hour_dep0)[0][1])
        '''
            
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
        if i<5:
            print 'feature length: ', len(tmp)-1
        matrix.append(tmp)
    print 'Concatenate features of all connected nodes together.'
    return matrix

#RNN predictor
def RNNPrediction(numEvents, connected, Temp, Wind, Events, Holiday, Time, TimeEachDay):
    #Normalize data to (0, 1)
    scaler1 = MinMaxScaler(feature_range = (0, 1))
    numEvents = scaler1.fit_transform(numEvents)

    #Dependence
    numWeek = 3; numDay = 3; numHour = TimeEachDay
    sequence_length1 = numHour + 1
    sequence_length2 = numWeek*7*TimeEachDay + 1
    matrix = ConvertSeriesToMatrix2(numEvents, connected, Temp, Wind, Events, Holiday, Time, sequence_length1, sequence_length2, numWeek, numDay, TimeEachDay)
    matrix = np.asarray(matrix)
    print 'data shape: ', matrix.shape
    
    #Split dataset: 80% for training and 20% for testing
    train_row = int(round(0.8*matrix.shape[0]))
    train_set = matrix[:train_row, :]
    test_set = matrix[train_row:, :]
     
    x_train = train_set[:, :-1]
    y_train = train_set[:, -1]
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    
    #Transform the training set into the LSTM format (number of samples, the dim of each elements)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    #Build the deep learning model
    model = Sequential()
    #Layer1: LSTM
    model.add(LSTM(input_dim = 1, output_dim = 64, return_sequences = True))
    model.add(Dropout(0.2))
    #Layer2: LSTM
    #model.add(LSTM(input_dim = 64, output_dim = 128, return_sequences = True))
    #model.add(Dropout(0.2))
    #Layer3: LSTM
    model.add(LSTM(output_dim = 128, return_sequences = False))
    model.add(Dropout(0.2))
    #Layer4: fully connected
    model.add(Dense(output_dim = 1, activation = 'sigmoid'))
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = "mse", optimizer = adam)
    
    #Training the model
    model.fit(x_train, y_train, batch_size = 128, nb_epoch = 500, validation_split = 0.2, verbose = 1)

    #save the model
    model_json = model.to_json()
    with open("Saved_models/model{0}.json".format(FLAGS.train_n), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Saved_models/model{0}.h5".format(FLAGS.train_n))
    print("Saved model to disk")

    #Prediction
    trainPredict = model.predict(x_train)
    testPredict = model.predict(x_test)
    
    #Invert the prediction
    trainPredict = scaler1.inverse_transform(trainPredict)
    testPredict = scaler1.inverse_transform(testPredict)
    
    train = scaler1.inverse_transform(np.array(y_train))
    test = scaler1.inverse_transform(np.array(y_test))
    
    #Calculate the error
    trainScore = math.sqrt(mean_squared_error(train, trainPredict))
    trainScore2 = np.average(train - trainPredict)
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(test, testPredict))
    testScore2 = np.average(test-testPredict)
    print('Test Score: %.2f RMSE' % (testScore))
    print train.shape, test.shape

def getConnectedInfo():
    #read all the connected nodes data.
    df3 = pd.read_csv('A.csv', header = None)
    A = df3.iloc[:, :]
    A = np.asarray(A)
    cindex  = []
    for cn in range(96):
        if A[cn,FLAGS.train_n-1] > 0 and cn!=FLAGS.train_n-1:
            cindex.append(cn)
    global A, cindex
    print 'The {0} th node is connected with: '.format(FLAGS.train_n), cindex
    df1 = pd.read_csv('data.csv', header = None)
    connected = df1.iloc[:,cindex]
    connected = np.asarray(connected)
    connected2 = np.zeros(connected.shape)
    for i in range(connected.shape[0]):
        if i%24 == 0:
            connected2[i,:] = connected[i,:]
        else:
            connected2[i,:] = connected[i,:]+connected2[i-1,:]
    connected3 = np.zeros((connected.shape[0]*2,connected.shape[1]))
    for col in range(connected.shape[1]):
        connected3[:,col] = img_enlarge(connected2[:,col], factor = 2.0, order = 2)
    connected3 = connected3.astype('float32')
    return connected3

        
#The main function
if __name__ == '__main__':
    ndays = 730
    TimeEachDay = 24
    
    #Events
    df1 = pd.read_csv('data.csv', header = None)
    numEvents = df1.iloc[:, FLAGS.train_n-1]
    print('numEvents Size: ', numEvents.shape)
    numEvents = np.asarray(numEvents)
    numEvents2 = np.zeros(ndays*TimeEachDay)	#cdf of the events
    for i in range(len(numEvents)):
    	if i%24 == 0:
    	     numEvents2[i] = numEvents[i]
    	else:
    	     numEvents2[i] = numEvents[i]+numEvents2[i-1]
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

    connected3 = getConnectedInfo()

    TimeEachDay *= 2	#Superresolve

    #train, and save the model
    res = RNNPrediction(numEvents3, connected3, Temp2, Wind2, Events2, Holiday2, Time2, TimeEachDay)
