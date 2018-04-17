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
listOPosts,listClasses = loadDataSet()

myVocabList = createVocabList(listOPosts)

print myVocabList


print setOfWords2Vec(myVocabList,listOPosts[0])

































