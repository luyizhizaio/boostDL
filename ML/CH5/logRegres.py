# -*- coding: utf-8 -*-
__author__ = 'tend'
from numpy import *

#梯度上升算法

def loadDataSet():

    dataMat =[]; labelMat=[]

    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat



def sigmoid(inX):
    return 1.0/(1+ exp(-inX))



def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn) #数值转成矩阵
    labelMat = mat(classLabels).transpose() #转置成1列
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1)) #n行1列
    for k in range(maxCycles):
        #下列几行为损失函数
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()*error #转置成n行3列的矩阵

    return weights



#EXECUTE
dataArr,labelMat = loadDataSet()

weight = gradAscent(dataArr,labelMat)

print weight

