__author__ = 'tend'



import numpy as np
from pylab import scatter,show,legend,xlabel,ylabel


#加载数据画图方法
data = np.loadtxt('data2.txt', delimiter=',')

X = data[:,0:2]
y = data[:,2]

pos = np.where(y == 1) #select
neg = np.where(y == 0)

scatter(X[pos,0],X[pos,1],marker='o',c='b')
scatter(X[neg,0],X[neg,1],marker='x',c='r')
xlabel('Featrue1/Exam 1 score')
ylabel('Featrue2/Exam 2 score')
legend(['Fail','Pass'])
show()


#多项式 特征转换，
def map_feature(x1,x2):
    '''''
    Maps the two input featrues to polomomial features.
    Returns a new feature array with more features of
    x1,x2, x1**2,x1x2,x2**2,...,x2**6
    '''''

    x1.shape=(x1.size,1)
    x2.shape=(x2.size,1)
    degree =6
    mapped_fea=np.ones(shape=(x1[:,0].size,1))
    m,n = mapped_fea.shape
    for i in range(1,degree+1):
        for j in range(j+1):
            r = (x1 ** (i-j))* (x2**j)



