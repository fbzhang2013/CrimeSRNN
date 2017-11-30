#!/usr/bin/python
"""
This code is used to predict the crime time series for the Zipcode region: 90003.
Diurnal cumulative and temporal superresolution.
Hourly, Daily, Weekly features.
External features.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.interpolate import interp1d
import csv
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#Resize the image
def img_enlarge(img):
    length = len(img)
    x = range(1, length+1)
    xx = np.zeros(length*2-1)
    xx[0] = x[0]
    for i in range(1, length):
        xx[2*i] = x[i]
        xx[2*i-1] = float(x[i-1]+x[i])/2.0
    func = interp1d(x, img, kind='linear')
    #func = interp1d(x, img, kind='cubic')
    yy = func(xx)
    return yy
    
#Convert data into (feature, label) format
def ConvertSeriesToMatrix(numEvents, Temp, Wind, Events, Holiday, Time, len1, len2, numWeek, numDay, TimeEachDay):
    matrix = []
    #We need to discard the data 0 ~ len2-1
    for i in range(len(numEvents) - len2):
	tmp = []			#(feature, label) at the time slot i + len2
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
	matrix.append(tmp)
    return matrix

#RNN predictor
def RNNPrediction(numEvents, Temp, Wind, Events, Holiday, Time, TimeEachDay):
    #Normalize data to (0, 1)
    scaler1 = MinMaxScaler(feature_range = (0, 1))
    numEvents = scaler1.fit_transform(numEvents)
    
    #Dependence
    numWeek = 3; numDay = 3; numHour = TimeEachDay
    sequence_length1 = numHour + 1
    sequence_length2 = numWeek*7*TimeEachDay + 1
    matrix = ConvertSeriesToMatrix(numEvents, Temp, Wind, Events, Holiday, Time, sequence_length1, sequence_length2, numWeek, numDay, TimeEachDay)
    matrix = np.asarray(matrix)
    print('Matrix shape: ', matrix.shape)
    np.savetxt('Matrix.csv', matrix, delimiter = ',', fmt = '%f')
    
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
#    model.add(LSTM(input_dim = 64, output_dim = 128, return_sequences = True))
#    model.add(Dropout(0.2))
    #Layer3: LSTM
    model.add(LSTM(output_dim = 128, return_sequences = False))
    model.add(Dropout(0.2))
    #Layer4: fully connected
    model.add(Dense(output_dim = 1, activation = 'sigmoid'))
    model.compile(loss = "mse", optimizer = "adam")
    
    #Training the model
    model.fit(x_train, y_train, batch_size = 128, nb_epoch = 500, validation_split = 0.2, verbose = 1)
    
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
    
    return [train, trainPredict, test, testPredict]


#The main function
if __name__ == '__main__':
    ndays = 365
    TimeEachDay = 24
    
    #Events
    df1 = pd.read_csv('EventsZip48.csv', header = None)
    numEvents = df1.iloc[:, 0]
    print('numEvents Size: ', numEvents.shape)
    numEvents = np.asarray(numEvents)
    numEvents2 = np.zeros(ndays*TimeEachDay)	#cdf of the events
    for i in range(len(numEvents)):
	if i%24 == 0:
	     numEvents2[i] = numEvents[i]
	else:
	     numEvents2[i] = numEvents[i]+numEvents2[i-1]
    numEvents3 = img_enlarge(numEvents2)	#Superresolve
    print('numEvents3 size: ', len(numEvents3))
    
    #Weather features
    df2 = pd.read_csv('ExternalFeatures.csv', header = None)
    Temp = df2.iloc[:, 0]
    Wind = df2.iloc[:, 1]
    Events = df2.iloc[:, 2]
    Holiday = df2.iloc[:, 3]
    Time = df2.iloc[:, 4]
    print('External size: ', Temp.shape, Wind.shape, Events.shape, Holiday.shape, Time.shape)
    
    Temp2 = np.zeros(ndays*(TimeEachDay*2-1))
    Wind2 = np.zeros(ndays*(TimeEachDay*2-1))
    Events2 = np.zeros(ndays*(TimeEachDay*2-1))
    Holiday2 = np.zeros(ndays*(TimeEachDay*2-1))
    Time2 = np.zeros(ndays*(TimeEachDay*2-1))
    for i in range(ndays):
        for j in range(TimeEachDay-1):
            T1 = i*(TimeEachDay*2-1) + 2*j
            T2 = i*(TimeEachDay*2-1) + 2*j+1
            T = i*TimeEachDay + j
            Temp2[T1] = Temp[T]; Temp2[T2] = Temp[T]
            Wind2[T1] = Wind[T]; Wind2[T2] = Wind[T]
            Events2[T1] = Events[T]; Events2[T2] = Events[T]
            Holiday2[T1] = Holiday[T]; Holiday2[T2] = Holiday[T]
            Time2[T1] = Time[T]; Time2[T2] = Time[T]
        Temp2[i*(TimeEachDay*2-1)+(TimeEachDay*2-1-1)] = Temp[i*TimeEachDay+TimeEachDay-1]
        Wind2[i*(TimeEachDay*2-1)+(TimeEachDay*2-1-1)] = Wind[i*TimeEachDay+TimeEachDay-1]
        Events2[i*(TimeEachDay*2-1)+(TimeEachDay*2-1-1)] = Events[i*TimeEachDay+TimeEachDay-1]
        Holiday2[i*(TimeEachDay*2-1)+(TimeEachDay*2-1-1)] = Holiday[i*TimeEachDay+TimeEachDay-1]
        Time2[i*(TimeEachDay*2-1)+(TimeEachDay*2-1-1)] = Time[i*TimeEachDay+TimeEachDay-1]
"""
    for i in range(ndays*TimeEachDay):
	Temp2[i*2] = Temp[i]; Temp2[i*2+1] = Temp[i]
	Wind2[i*2] = Wind[i]; Wind2[i*2+1] = Wind[i]
	Events2[i*2] = Events[i]; Events2[i*2+1] = Events[i]
	Holiday2[i*2] = Holiday[i]; Holiday2[i*2+1] = Holiday[i]
	Time2[i*2] = Time[i]; Time2[i*2+1] = Time[i]
"""
    print('External feature dimensions: ', len(Temp2), len(Wind2), len(Events2), len(Holiday2), len(Time2))
    
    numEvents3 = numEvents3.astype('float32')
    Temp2 = np.asarray(Temp2); Temp2 = Temp2.astype('float32')
    Wind2 = np.asarray(Wind2); Wind2 = Wind2.astype('float32')
    Events2 = np.asarray(Events2); Events2 = Events2.astype('float32')
    Holiday2 = np.asarray(Holiday2); Holiday2 = Holiday2.astype('float32')
    Time2 = np.asarray(Time2); Time2 = Time2.astype('float32')
    
    TimeEachDay = 47	#Superresolve
    numSlots1 = TimeEachDay * 365
    numEvents3 = numEvents3[-numSlots1:]
    Temp2 = Temp2[-numSlots1:]; Wind2 = Wind2[-numSlots1:]
    Events2 = Events2[-numSlots1:]; Holiday2 = Holiday2[-numSlots1:]; Time2 = Time2[-numSlots1:] 
    
    res = RNNPrediction(numEvents3, Temp2, Wind2, Events2, Holiday2, Time2, TimeEachDay)
    
    train = res[0]
    np.savetxt('TrainSupT.csv', train)
#    train = img_enlarge(train, factor = 0.5, order = 2)
    trainPredict = res[1]
    np.savetxt('TrainPredictSupT.csv', trainPredict)
#    trainPredict = img_enlarge(trainPredict, factor = 0.5, order = 2)
    test = res[2]
    np.savetxt('TestSupT.csv', test)
#    test = img_enlarge(test, factor = 0.5, order = 2)
    testPredict = res[3]
    np.savetxt('TestPredictSupT.csv', testPredict)
#    testPredict = img_enlarge(testPredict, factor = 0.5, order = 2)
#    np.savetxt('Train.csv', train)
#    np.savetxt('TrainPredict.csv', trainPredict)
#    np.savetxt('Test.csv', test)
#    np.savetxt('TestPredict.csv', testPredict)
