# -*- coding: utf-8 -*-
__author__ = 'tend'
from numpy import *
import operator
from os import listdir  #读取目录

#create dataset
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group ,labels



def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis =1)
    distances = sqDistances **0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel]  = classCount.get(voteIlabel,0) +1
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]






#execute
group,labels = createDataSet()

print group

print labels


# execute classify method
c =classify0([0,0],group,labels,3)
print c #B


################################################################################
#2.2
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index ,:] = listFromLine[0:3] #选择3个字段
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return returnMat,classLabelVector






#
datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')

print datingDataMat

print datingLabels[0:20]

#输出散点图
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0 * array(datingLabels))
#plt.show()


#散点图2
fig = plt.figure()
ax = fig.add_subplot(111)
#里程数和百分比作图
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0 * array(datingLabels))
#plt.show()




#归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) #生成shape 与dataSet一样的0矩阵
    m = dataSet.shape[0]
    normDataSet = dataSet-tile(minVals ,(m,1)) #tile表示复制数组
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals



normMat,ranges,minVals = autoNorm(datingDataMat)



#测试算法效果

def datingClassTest():

    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0] #行数
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],3)
        print "the calssifier came back with: %d,the real answer is: %d" % (classifierResult,datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
        print "the total error rate is : %f" % (errorCount /float(numTestVecs))



#执行
#datingClassTest()





def classifyPersion():
    resultList = ['not at all','in small dases','in large doses']
    percentTats = float(raw_input(
        "percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat ,ranges,minVals  = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats,iceCream])
    classifierResult = classify0((inArr - minVals)/ranges,normMat,datingLabels,3)
    print "You weill probably like this persion:",resultList[classifierResult -1]



#classifyPersion()



#图片转成向量
def img2vector(filename):

    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32 * i + j] = int(lineStr[j])
    return returnVect

#执行
testVector = img2vector('testDigits/0_13.txt')
print testVector[0,0:31]



def handwritingClassTest():
    hwLabels = [] #定义一个向量
    #获取目录内容
    trainingFileList = listdir('trainingDigits') #获取文件名称列表
    m = len(trainingFileList)
    trainingMat = zeros((m,1024)) #m行1024列矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr) #向量里增加值
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)

        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print "the classifier came back with :%d,the real answer is :%d" % (classifierResult,classNumStr)

        if (classifierResult != classNumStr): errorCount +=1.0

    print  "\nthe totla number of errors is : %d" % errorCount
    print "\nthe total error rate is : %f" % (errorCount /float(mTest))

def  handwritingClassTest2():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr =  int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList= listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr =int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)

        classifierResult = classify0(vectorUnderTest, trainingMat,hwLabels,5)
        print "the classifier came back with :%d , the real answer is : %d" % (classifierResult,classNumStr)

        if ( classifierResult != classNumStr) : errorCount +=1.0
    print "\n the total number of errors is :%d" % errorCount
    print "\n the total error rate is : %f" %(errorCount/float(mTest))



#执行
handwritingClassTest()

handwritingClassTest2()

















