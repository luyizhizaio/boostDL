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

    return weights  #返回的为矩阵



#EXECUTE
dataArr,labelMat = loadDataSet()

weight = gradAscent(dataArr,labelMat)

print weight


#画出数据集的最佳拟合函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    #weights =wei.getA() #返回数组
    dataMat,labelMat = loadDataSet()
    dataArr=array(dataMat) #转成数组
    n = shape(dataArr)[0] #数据集行数
    xcord1 =[]; ycord1 = []
    xcord2 = [];ycord2=[]
    for i in range(n):
        if int(labelMat[i]) ==1:
            xcord1.append(dataArr[i,1]);
            ycord1.append(dataArr[i,2]);
        else:
            xcord2.append(dataArr[i,1]);
            ycord2.append(dataArr[i,2]);
    fig = plt.figure()
    ax = fig.add_subplot(111) #子图
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s') #算点图
    ax.scatter(xcord2,ycord2,s=30,c='green')

    x = arange(-3.0,3.0,0.1) #返回-3到3的数组
    y = (-weights[0] -weights[1] *x)/weights[2] #ax+by+c =0 的直线
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()



#执行
plotBestFit(weight.getA())




#随机梯度上升，

def stocGradAscent0(dataMatrix,classLabels):
    m,n =shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights)) #sum把此行的值加起来
        error = classLabels[i] -h
        weights = weights + alpha * error * dataMatrix[i]

    return weights #返回的为数组


dataMat,labelMat= loadDataSet()
weights = stocGradAscent0(array(dataMat),labelMat)

print 'stocGradAscent'
print weights
plotBestFit(weights)




#改进的随机梯度上升算法

def stocGradAscent1(dataMatrix,classLabels,numIter = 150):

    m,n = shape(dataMatrix)
    weights = ones(n) #返回长度为n值为1的数组
    for j in range(numIter):
        dataIndex = range(m) #返回0到m-1的数组
        for i in range(m):
            alpha = 4/(1.0+j+i) +0.0001 #步长是递减的
            randIndex = int(random.uniform(0,len(dataIndex))) #随机下标
            h = sigmoid(sum(dataMatrix[randIndex] *weights))

            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex]) #从数组中删除元素
    return weights



dataMat,labelMat = loadDataSet()

weights = stocGradAscent1(array(dataMat),labelMat)

print  weights

plotBestFit(weights)
























