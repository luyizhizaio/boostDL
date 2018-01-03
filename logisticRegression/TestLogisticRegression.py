__author__ = 'tend'

import numpy as np
import matplotlib.pyplot as plt

import time
from LogisticRegression import *

def loadData():

    train_x = []
    train_y = []
    fileIn = open("D:/sourceSpace/tensorflow_test/logisticRegression/testSet.txt")
    for line in fileIn:
        lineArr= line.strip().split(",")
        train_x.append([1.0,float(lineArr[0].strip()),float(lineArr[1].strip())])
        train_y.append(float(lineArr[2].strip()))


    return np.mat(train_x),np.mat(train_y).transpose()


def main():

    train_x, train_y = loadData()

    test_x = train_x; test_y = train_y


    #training
    #opts= {'alpha':0.01,'maxIter':50,'optimizeType':'stocGradDescent'}
    opts= {'alpha':0.01,'maxIter':50,'optimizeType':'smoothStocGradDescent'}
    optimalWeights = trainLogRegression(train_x,train_y,opts)


    #testing
    accuracy =testLogRegression(optimalWeights,test_x,test_y)

    print 'the classification accuracy is :%.3f%%' %(accuracy * 100)


    showLogRegress(optimalWeights,test_x,test_y)



if __name__ =='__main__':
    main()