# -*- coding: utf-8 -*-
__author__ = 'tend'

from math import log
import operator


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



#选择最好的数据集划分方式, 返回特征下标
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) -1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range (numFeatures):
        featList = [example[i] for example in dataSet] #第一列属性值
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



#获取数量最大的分类名称
def majorityCnt(classList):

    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    #排序，降序
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse =True)
    return sortedClassCount[0][0]


#创建树

#参数列表，数据集和标签列表;返回嵌套字典
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #类别相同停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #遍历完所有特征返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet) #计算信息增益，选择出最好的特征
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


def createTree2(dataSet,labels):
    classList = [example[-1] for example in dataSet] #统计每个分类列
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    if len(dataSet[0]) ==1:
        return majorityCnt(classList) #计算数量最多的分类
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])  #删除列表中元素
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues) #排重
    for value in uniqueVals: #得到每个特征值
        subLabels=labels[:] #复制标签,为了不改变参数列表中的labels
        myTree[bestFeatLabel][value] = createTree2(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


#执行
myDat ,labels = createDataSet()
mytree = createTree(myDat,labels)

print mytree



#使用决策树的分类函数
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]  #获取字典的key
    secondDict=inputTree[firstStr]  #根据key获取value
    featIndex = featLabels.index(firstStr) #第一个key的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ =='dict': #
                classLabel = classify(secondDict[key],featLabels,testVec)
            else: classLabel = secondDict[key]
    return classLabel


#execute

import treePlotter

myDat,labels = createDataSet()

myTree = treePlotter.retrieveTree(0)

#执行分类算法
#print classify(myTree,labels,[0,1])




#存储tree ，使用pickle序列化对象
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

#execute

storeTree(myTree,'classifierStorage,txt')

tree1 = grabTree('classifierStorage,txt')
print tree1



