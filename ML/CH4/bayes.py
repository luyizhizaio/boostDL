# -*- coding: utf-8 -*-
__author__ = 'tend'
from numpy import *



#从文本构建词向量

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0,1,0,1,0,1] #1
    return postingList,classVec


#包含所有单词的列表
def createVocabList(dataSet):
    vocabSet = set([]) #创建一个空集里面是数组
    for document in dataSet:
        vocabSet = vocabSet | set(document) #并集,Set是排重的
    return list(vocabSet)


#输出文档向量
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList) #生成长度为len的列表
    for word in  inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] =1  #把单词相应的位置标示为1.
        else: print "the word: %s is not in my Vocabulary !" % word
    return returnVec


#执行
#listOPosts,listClasses = loadDataSet()

#myVocabList = createVocabList(listOPosts)

#print myVocabList


#print setOfWords2Vec(myVocabList,listOPosts[0])



#bayes 分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    #训练样本的数量
    numTrainDocs = len(trainMatrix)
    #样本的特征数
    numWords = len(trainMatrix[0])
    #正样本的比例
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #创建数组
    p0Num=zeros(numWords)
    p1Num = zeros(numWords)
    #每个类别的总词数
    p0Denom =0.0;p1Denom =0.0
    for i in range(numTrainDocs):
        if trainCategory[i] ==1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vec = p1Num/p1Denom #类型为1时各个词的概率。
    p0Vec = p0Num/p0Denom
    return p0Vec,p1Vec,pAbusive


#执行
listOPosts,listClasses = loadDataSet()

myVocabList = createVocabList(listOPosts)

trainMat = [] #创建一个矩阵
for postInDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList,postInDoc))

p0V,p1V,pAb = trainNB0(trainMat,listClasses)

print(pAb)

print(p0V)





























