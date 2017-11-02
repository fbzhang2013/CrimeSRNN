import numpy as np
import math
import csv
import pandas as pd

def Classify_Node(num_class = 3, num_crime = None):
    #Classify the nodes into 3 classes
    n_max = np.max(num_crime)
    n_min = np.min(num_crime)
    types = num_class
    h = (n_max - n_min)/(types+1)

    Type1 = [node for node in xrange(num_crime.shape[0]) if num_crime[node] < n_min+h]
    Type2 = [node for node in xrange(num_crime.shape[0]) if num_crime[node] >= n_min+h and num_crime[node] < n_min+2*h]
    Type3 = [node for node in xrange(num_crime.shape[0]) if num_crime[node] >= n_min+2*h]
    print 'Class 0' , Type1 
    print 'Class 1' , Type2
    print 'Class 2' , Type3 
    nodeType = {}
    for n in Type1:
        nodeType[n] = '0'
    for n in Type2:
        nodeType[n] = '1'
    for n in Type3:
        nodeType[n] = '2'

    return nodeType, Type1, Type2, Type3

if __name__ == '__main__':
    df1 = pd.read_csv('data.csv', header = None)
    data = df1.iloc[:, :]
    data = np.asarray(data); data = data[:,:-1]; data = data.astype('float32')
    num_crime = np.sum(data,axis=0)
    nodeType, Type1, Type2, Type3 = Classify_Node(3, num_crime)
    
    
