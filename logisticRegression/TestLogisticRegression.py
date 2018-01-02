__author__ = 'tend'

from numpy import *
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


    return mat(train_x),mat(train_y).transpose()


train_x, train_y = loadData()

test_x = train_x; test_y = train_y


#training
opts= {'alpha':0.01,'maxIter':20,'optimizeType':'gradDescent'}
optimalWeights = trainLogRegression(train_x,train_y,opts)


#testing
accuracy =testLogRegression(optimalWeights,test_x,test_y)

print 'the classification accuracy is :%.3f%%' %(accuracy * 100)


showLogRegress(optimalWeights,test_x,test_y)