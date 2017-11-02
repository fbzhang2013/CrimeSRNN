import numpy as np
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--n', type=int, default=1,
                    help='which type is used for training? 1,2,or 3')

FLAGS = parser.parse_args()

def Classify_Node(num_class = 3, num_crime = None):
    #Classify the nodes into 3 classes
    n_max = np.max(num_crime)
    n_min = np.min(num_crime)
    types = num_class
    h = (n_max - n_min)/(types+3)

    Type1 = [node for node in xrange(num_crime.shape[0]) if num_crime[node] < n_min+h]
    Type2 = [node for node in xrange(num_crime.shape[0]) if num_crime[node] >= n_min+3*h and num_crime[node] < n_min+4*h]
    Type3 = [node for node in xrange(num_crime.shape[0]) if num_crime[node] >= n_min+5*h]
    print 'Type 1' , Type1
    print 'Type 2' , Type2
    print 'Type 3' , Type3
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
    print num_crime
    nodeType, Type1, Type2, Type3 = Classify_Node(3, num_crime)

    if FLAGS.n == 1:
        train_type = Type1 
    elif FLAGS.n == 2:
        train_type = Type2
    elif FLAGS.n == 3:
        train_type = Type3

    RMSECDF = np.genfromtxt('Results_Rmse/RMSECDF.csv')
    RMSEPDF = np.genfromtxt('Results_Rmse/RMSEPDF.csv')
    RMSEPDF_zero = np.genfromtxt('Results_Rmse/RMSEPDF_zero.csv')
    #print RMSEPDF_zero[Type1]
    #print RMSEPDF_zero[Type2]
    #print RMSEPDF_zero[Type3]

    #compute overall average result.
    RMSECDF_overall_aver_test1 = 0
    RMSECDF_overall_aver_test2 = 0
    RMSECDF_overall_aver_test3 = 0
    RMSEPDF_overall_aver_test1 = 0
    RMSEPDF_overall_aver_test2 = 0
    RMSEPDF_overall_aver_test3 = 0
    for n in train_type:
        RMSECDF_overall_aver_test1 += np.sum(RMSECDF[n,Type1])/len(Type1)
        RMSECDF_overall_aver_test2 += np.sum(RMSECDF[n,Type2])/len(Type2)
        RMSECDF_overall_aver_test3 += np.sum(RMSECDF[n,Type3])/len(Type3)
        RMSEPDF_overall_aver_test1 += np.sum(RMSEPDF[n,Type1])/len(Type1)
        RMSEPDF_overall_aver_test2 += np.sum(RMSEPDF[n,Type2])/len(Type2)
        RMSEPDF_overall_aver_test3 += np.sum(RMSEPDF[n,Type3])/len(Type3)

    print 'Use model trained by type3, the overall rmse average: '
    print 'type 1: CDF = {0}, PDF = {1}'.format(RMSECDF_overall_aver_test1/len(train_type), RMSEPDF_overall_aver_test1/len(train_type))
    print 'type 2: CDF = {0}, PDF = {1}'.format(RMSECDF_overall_aver_test2/len(train_type), RMSEPDF_overall_aver_test2/len(train_type))
    print 'type 3: CDF = {0}, PDF = {1}'.format(RMSECDF_overall_aver_test3/len(train_type), RMSEPDF_overall_aver_test3/len(train_type))

    RMSEPDF_aver = np.zeros((50,))
    for n in train_type:
        RMSEPDF_aver = RMSEPDF_aver + RMSEPDF[n,:]
    RMSEPDF_aver = RMSEPDF_aver/len(train_type)

    self_rmse = 0
    for n in train_type:
        self_rmse += RMSEPDF[n,n]

    x = np.arange(1,51)

    #plot distribution of pdf rmse.
    plt.rcParams["figure.figsize"] = [15.0, 10.0]
    plt.bar(x,RMSEPDF_zero,color = 'k', label = 'Zeros Prediction', align = 'center')
    plt.bar(x[Type1],RMSEPDF_aver[Type1],color = 'r', label = 'Type1 Prediction', align = 'center')
    plt.bar(x[Type2],RMSEPDF_aver[Type2],color = 'b', label = 'Type2 Prediction', align = 'center')
    plt.bar(x[Type3],RMSEPDF_aver[Type3],color = 'g', label = 'Type3 Prediction', align = 'center')
    plt.legend()
    plt.ylabel('PDF Rmse')
    plt.xlabel('Node Index')
    plt.title('Average pdf rmse using Class{0} nodes(more crime)'.format(FLAGS.n))
    plt.savefig('train{0}_distribution.png'.format(FLAGS.n))
    plt.close()

    #plot overall average 
    plt.rcParams["figure.figsize"] = [8.0, 6.0]
    Type_x = [1,4,7]
    y = [RMSEPDF_overall_aver_test1/len(train_type), RMSEPDF_overall_aver_test2/len(train_type), RMSEPDF_overall_aver_test3/len(train_type)]
    zeros = [np.sum(RMSEPDF_zero[Type1])/len(Type1), np.sum(RMSEPDF_zero[Type2])/len(Type2), np.sum(RMSEPDF_zero[Type3])/len(Type3)]
    ratio = np.divide(y,zeros)
    plt.bar(Type_x, zeros, color = 'k', label = 'Zeros Prediction', align = 'center')
    plt.bar(Type_x[0],y[0],color = 'r', label = 'Type1 Prediction', align = 'center')
    plt.bar(Type_x[1],y[1],color = 'b', label = 'Type2 Prediction', align = 'center')
    plt.bar(Type_x[2],y[2],color = 'g', label = 'Type3 Prediction', align = 'center')
    plt.legend(loc='upper left')
    plt.ylabel('PDF Rmse')
    plt.title('Class-wise average pdf rmse using Class{0} nodes(more crime)'.format(FLAGS.n))
    plt.text(0.5, 0.8,'ratio: {0}\n, {1}\n, {2}'.format(ratio[0],ratio[1],ratio[2]))
    plt.savefig('train{0}_ClassAver.png'.format(FLAGS.n))
    plt.close()
    
    #compute average matrix.
    Type_list = [Type1, Type2, Type3]
    for train_type_index in range(1,4):
        for test_type_index in range(1,4):
            train_type = Type_list[train_type_index-1]
            test_type = Type_list[test_type_index-1]
            ma = np.zeros((3,3))
            for train_n in train_type:
                for test_n in test_type:
                    ma = ma + np.genfromtxt('Results_Matrix/AnoAcc_train{0}_test{1}.csv'.format(train_n+1, test_n+1))
            ma = ma/len(train_type)/len(test_type)
            np.savetxt('Results_Matrix_Aver/AnoAcc_TrainType{0}_TestType{1}.csv'.format(train_type_index, test_type_index), ma)
    


    
