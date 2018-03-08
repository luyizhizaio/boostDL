# -*- coding: utf-8 -*-
__author__ = 'tend'

from math import log


#计算信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts={}  #定义一个字典（map）
    for featVec in dataSet:
        currentLabel = featVec[-1] #获取当前标签
        #计算每个分类标签的数量。
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] =0
        labelCounts[currentLabel] +=1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -=prob * log(prob ,2)
    return shannonEnt


#生成数据集
def createDataSet():
    dataSet= [[1,1,'yes'],
              [1,1,'yes'],
              [1,0,'no'],
              [0,1,'no'],
              [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#执行
myDat,labels = createDataSet()

Ent = calcShannonEnt(myDat)
print Ent
#修改一个标签
#myDat[0][-1] = 'maybe'
Ent = calcShannonEnt(myDat)
print Ent


#划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet = [] #list
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis] #添加前半部分值
            reduceFeatVec.extend(featVec[axis +1:]) #添加后半部分值
            retDataSet.append(reduceFeatVec)
    return retDataSet

#执行
myDat,labels = createDataSet()

print myDat
#划分出第1列值为1的记录
print splitDataSet(myDat,0,1)

print splitDataSet(myDat,0,0)



#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) -1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range (numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i ,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy -newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#execute
myDat,labels = createDataSet()
print chooseBestFeatureToSplit(myDat)













