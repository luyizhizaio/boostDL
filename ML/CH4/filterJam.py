# -*- coding: utf-8 -*-
__author__ = 'tend'
import re
from numpy import *
from bayes import *


mySent = 'this book is the best book on Python or ML.I have ever laid eyes upon.'

print mySent.split()



#把标点符号分开
regEx = re.compile('\\W*')

listOfTokens = regEx.split(mySent)

print listOfTokens


#去掉空字符串和变成小写
print [tok.lower() for  tok in listOfTokens if len(tok) > 0]




emailText = open('email/ham/6.txt').read()

print  emailText

listOfTokens = regEx.split(emailText)

print listOfTokens


#解析字符串
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for  tok in listOfTokens if  len(tok) > 2]

def spamTest():

    docList=[];classList =[]; fullText=[]
    for i  in  range(1,26):
        wordList = textParse(open('email/spam/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50);testSet=[]

    #生成测试集  (留存交叉校验)
    for i in  range(10):
        #生成随机数
        randIndex = int(random.uniform(0,len(trainingSet)))
        print 'random number:', randIndex
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses=[]
    for  docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0

    #对测试集分类
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is :' ,float(errorCount) / len(testSet)





#执行
spamTest()


